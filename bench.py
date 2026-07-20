#!/usr/bin/env python3
"""llm-bench-2x4090 — download, test, document, discard HF models on a 2xRTX 4090 rig.

Each run produces results/<model-slug>.json (success OR structured failure — both are
data), regenerates the README leaderboard, and commits. Weights are discarded after
testing unless kept. Stdlib only.

Usage:
  bench.py run <hf-repo> [--keep] [--image IMG] [--flags "..."] [--skip-download-check]
  bench.py queue          # process every pending entry in models.json
  bench.py report         # regenerate README table from results/
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "hfcache")
RESULTS = os.path.join(HERE, "results")
PORT = 8199
NAME = "llmbench-vllm"
BASE = f"http://127.0.0.1:{PORT}"
# (gpu_memory_utilization, max_model_len) tried in order until the model serves
LADDER = [(0.92, 32768), (0.90, 16384), (0.85, 8192)]
FAMILY_FLAGS = [
    (r"[Qq]wen3", "--reasoning-parser qwen3"),
]

os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def sh(cmd, timeout=120, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stderr[-500:]}")
    return r


def cfg():
    with open(os.path.join(HERE, "models.json")) as f:
        return json.load(f)


def slug(repo):
    return repo.replace("/", "--")


def http(path, payload=None, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode() if payload else None,
    )
    return urllib.request.urlopen(req, timeout=timeout)


def hf_model_bytes(repo):
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        files = json.load(r)
    st = sum(f["size"] for f in files if f["path"].endswith(".safetensors"))
    return st or sum(f.get("size", 0) for f in files)


def gpus_busy():
    r = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    return any(int(x) > 2000 for x in r.stdout.split())


class HwMon(threading.Thread):
    """Samples total board power and max VRAM across both GPUs every 2s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            r = sh("nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits")
            try:
                rows = [tuple(float(v) for v in ln.split(",")) for ln in r.stdout.strip().splitlines()]
                self.samples.append((sum(p for p, _ in rows), max(m for _, m in rows)))
            except Exception:
                pass
            time.sleep(2)

    def summary(self):
        if not self.samples:
            return {}
        pw = [p for p, _ in self.samples]
        return {"mean_w": round(sum(pw) / len(pw)), "max_w": round(max(pw)),
                "max_vram_gb": round(max(m for _, m in self.samples) / 1024, 1)}


def download(repo, image):
    t0 = time.time()
    r = sh(
        f"docker run --rm -v {CACHE}:/root/.cache/huggingface --entrypoint python3 {image} "
        f"-c \"from huggingface_hub import snapshot_download; snapshot_download('{repo}')\"",
        timeout=7200,
    )
    if r.returncode != 0:
        raise RuntimeError(f"download failed: {r.stderr[-400:]}")
    return round(time.time() - t0, 1)


def serve_attempt(repo, image, util, maxlen, flags):
    sh(f"docker rm -f {NAME}", timeout=60)
    r = sh(
        f"docker run -d --name {NAME} --gpus all --shm-size 16g -p {PORT}:8000 "
        f"-v {CACHE}:/root/.cache/huggingface {image} "
        f"--model {repo} --tensor-parallel-size 2 --gpu-memory-utilization {util} "
        f"--max-model-len {maxlen} --host 0.0.0.0 --port 8000 {flags}",
        timeout=120,
    )
    if r.returncode != 0:
        return False, r.stderr[-300:]
    deadline = time.time() + 900  # weights are local; 15 min covers load + compile
    while time.time() < deadline:
        try:
            http("/v1/models", timeout=5)
            return True, ""
        except Exception:
            pass
        alive = sh(f"docker inspect -f '{{{{.State.Running}}}}' {NAME}").stdout.strip()
        if alive != "true":
            logs = sh(f"docker logs --tail 40 {NAME} 2>&1").stdout[-2000:]
            return False, logs
        time.sleep(10)
    return False, "health timeout (15 min); container alive but never served\n" + \
        sh(f"docker logs --tail 40 {NAME} 2>&1").stdout[-2000:]


def completion(model, prompt, max_tokens, stream=False):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7}
    if stream:
        payload["stream"] = True
        t0 = time.time()
        first = None
        with http("/v1/chat/completions", payload, timeout=600) as r:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line and first is None:
                    first = time.time() - t0
        return first
    t0 = time.time()
    resp = json.load(http("/v1/chat/completions", payload, timeout=600))
    u = resp["usage"]
    return u["prompt_tokens"], u["completion_tokens"], time.time() - t0


def battery(model):
    import concurrent.futures
    import random
    import string

    def salt():
        return "".join(random.choices(string.ascii_lowercase, k=10))

    out = {}
    completion(model, f"[{salt()}] Say hello.", 16)  # warmup + compile

    _, ct, dt = completion(model, f"[{salt()}] Write a detailed 1000-word essay on the history of computing.", 1200)
    out["single_stream_toks"] = round(ct / dt, 1)

    out["ttft_s"] = round(completion(model, f"[{salt()}] Say hello.", 24, stream=True), 2)

    big = f"[{salt()}] " + ("The quick brown fox jumps over the lazy dog. " * 1500)
    pt, _, dt = completion(model, big + "\nReply OK.", 1)
    out["prefill"] = {"prompt_tok": pt, "toks": round(pt / dt)}

    out["sweep"] = []
    for n in (8, 32):
        prompts = [f"[{salt()} #{i}] Explain topic {i % 8} in technical detail." for i in range(2 * n)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda p: completion(model, p, 256), prompts))
        wall = time.time() - t0
        total = sum(c for _, c, _ in res)
        out["sweep"].append({"concurrency": n, "agg_toks": round(total / wall)})

    try:  # tool-call smoke: structured call comes back parsed?
        resp = json.load(http("/v1/chat/completions", {
            "model": model, "max_tokens": 256,
            "messages": [{"role": "user", "content": "Weather in Paris? Use the tool."}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        }, timeout=120))
        tcs = resp["choices"][0]["message"].get("tool_calls") or []
        out["tool_call"] = "ok" if tcs and tcs[0]["function"]["name"] == "get_weather" else "no_structured_call"
    except Exception as e:
        out["tool_call"] = f"error: {repr(e)[:80]}"
    return out


def discard(repo):
    path = os.path.join(CACHE, "hub", f"models--{slug(repo)}")
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def run_model(repo, keep=False, image=None, flags=None, skip_check=False):
    c = cfg()
    image = image or c.get("image_default", "vllm/vllm-openai:latest")
    if flags is None:
        flags = next((f for pat, f in FAMILY_FLAGS if re.search(pat, repo)), "")
        flags = c.get("overrides", {}).get(repo, {}).get("flags", flags)
    rec = {"repo": repo, "ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "image": image, "flags": flags, "status": "started"}

    try:
        if gpus_busy():
            raise RuntimeError("GPUs busy (production stack running?) — run 'llm stop' first")
        size = hf_model_bytes(repo)
        rec["disk_gb"] = round(size / 1e9, 1)
        free = shutil.disk_usage(CACHE).free
        if not skip_check and size * 1.3 > free:
            raise RuntimeError(f"insufficient disk: need {size*1.3/1e9:.0f}GB, have {free/1e9:.0f}GB")
        print(f"[{repo}] downloading {rec['disk_gb']}GB...", flush=True)
        rec["download_s"] = download(repo, image)

        served = False
        for util, maxlen in LADDER:
            print(f"[{repo}] serve attempt util={util} len={maxlen}", flush=True)
            ok, err = serve_attempt(repo, image, util, maxlen, flags)
            if ok:
                rec["serve_config"] = {"gpu_mem_util": util, "max_model_len": maxlen}
                served = True
                break
            rec.setdefault("serve_errors", []).append({"util": util, "len": maxlen, "tail": err[-800:]})
        if not served:
            rec["status"] = "serve_failed"
            return rec
        try:
            rec["vllm_version"] = json.load(http("/version", timeout=10)).get("version")
        except Exception:
            pass

        model_id = json.load(http("/v1/models"))["data"][0]["id"]
        mon = HwMon()
        mon.start()
        print(f"[{repo}] running battery...", flush=True)
        rec["battery"] = battery(model_id)
        mon.stop_flag = True
        rec["hw"] = mon.summary()
        agg32 = next((s["agg_toks"] for s in rec["battery"]["sweep"] if s["concurrency"] == 32), None)
        if agg32 and rec["hw"].get("mean_w"):
            rec["tok_per_joule"] = round(agg32 / rec["hw"]["mean_w"], 2)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = rec.get("status", "error")
        rec["error"] = repr(e)[:500]
    finally:
        sh(f"docker rm -f {NAME}", timeout=60)
        if not keep:
            discard(repo)
        with open(os.path.join(RESULTS, slug(repo) + ".json"), "w") as f:
            json.dump(rec, f, indent=2)
        report()
        git_commit(f"results: {repo} [{rec['status']}]")
    return rec


def report():
    rows = []
    for fn in sorted(os.listdir(RESULTS)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(RESULTS, fn)))
        b = d.get("battery", {})
        agg32 = next((s["agg_toks"] for s in b.get("sweep", []) if s["concurrency"] == 32), "")
        note = "OK" if d["status"] == "ok" else d["status"]
        rows.append((agg32 or 0, f"| {d['repo']} | {d.get('disk_gb','')} | "
                     f"{b.get('single_stream_toks','—')} | {b.get('prefill',{}).get('toks','—')} | "
                     f"{agg32 or '—'} | {d.get('hw',{}).get('max_vram_gb','—')} | "
                     f"{d.get('hw',{}).get('mean_w','—')} | {d.get('tok_per_joule','—')} | "
                     f"{b.get('tool_call','—')} | {note} |"))
    rows.sort(key=lambda r: -(r[0] or 0))
    table = ("| model | GB | 1-stream tok/s | prefill tok/s | agg@32 | VRAM GB | mean W | tok/J | tools | status |\n"
             "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(r for _, r in rows))
    rd = os.path.join(HERE, "README.md")
    txt = open(rd).read()
    txt = re.sub(r"(<!--RESULTS:BEGIN-->).*(<!--RESULTS:END-->)",
                 r"\1\n" + table + r"\n\2", txt, flags=re.S)
    open(rd, "w").write(txt)


def git_commit(msg):
    sh(f"cd {HERE} && git add -A && git commit -q -m {json.dumps(msg)}", timeout=60)
    r = sh(f"cd {HERE} && git push -q", timeout=120)
    if r.returncode != 0:
        print("WARN: git push failed (no auth yet?) — committed locally", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("repo")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--image")
    p.add_argument("--flags")
    p.add_argument("--skip-download-check", action="store_true")
    sub.add_parser("queue")
    sub.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "run":
        rec = run_model(a.repo, keep=a.keep, image=a.image, flags=a.flags,
                        skip_check=a.skip_download_check)
        print(json.dumps({k: v for k, v in rec.items() if k != "serve_errors"}, indent=2))
    elif a.cmd == "queue":
        c = cfg()
        for entry in c["queue"]:
            if entry.get("done"):
                continue
            rec = run_model(entry["repo"], keep=entry.get("keep", False),
                            flags=entry.get("flags"))
            entry["done"] = True
            entry["status"] = rec["status"]
            with open(os.path.join(HERE, "models.json"), "w") as f:
                json.dump(c, f, indent=2)
    elif a.cmd == "report":
        report()
        git_commit("regenerate report")


if __name__ == "__main__":
    main()
