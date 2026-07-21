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
ASSETS = os.path.join(HERE, "assets")
PORT = 8199
NAME = "llmbench-vllm"          # container name (reused across engines)
BASE = f"http://127.0.0.1:{PORT}"
# (gpu_memory_utilization, max_model_len) tried in order until the model serves
LADDER = [(0.92, 32768), (0.85, 8192)]
SGLANG_LADDER = [(0.85, 32768), (0.85, 8192)]  # (mem_fraction_static, context_length)
POWER_CAP_W = 340  # per-GPU cap near the 4090 efficiency knee; recorded in every result
# Per-engine serving images. vLLM image also downloads GGUF (has huggingface_hub).
LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
SGLANG_IMAGE = "lmsysorg/sglang:latest"
SLOW_TOKS = 15   # single-stream tok/s below this ⇒ reduced battery (CPU-offloaded configs)
CACHE_BUDGET_GB = 900  # LRU weight cache: keep recent models for free re-runs
FAMILY_FLAGS = [
    (r"[Qq]wen3", "--reasoning-parser qwen3 --enable-auto-tool-choice "
                  "--tool-parser-plugin /plugins/qwen3_coder_fixed.py "
                  "--tool-call-parser qwen3_coder_fixed"),
    (r"[Gg]emma-4", "--enable-auto-tool-choice --reasoning-parser gemma4 "
                    "--tool-call-parser gemma4"),
    # IBM Granite 4.x uses its own tool-call format — the granite parser lets the
    # smoke test register a structured call (else it logs not_configured).
    (r"[Gg]ranite-4", "--enable-auto-tool-choice --tool-call-parser granite"),
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


def download(repo, image, allow_patterns=None):
    """Snapshot a repo into the shared HF cache. allow_patterns limits the fetch to
    matching files (used to grab a single GGUF quant out of a multi-quant repo)."""
    t0 = time.time()
    ap = ""
    if allow_patterns:
        lst = "[" + ",".join("'" + p + "'" for p in allow_patterns) + "]"
        ap = f", allow_patterns={lst}"
    r = sh(
        f"docker run --rm -v {CACHE}:/root/.cache/huggingface --entrypoint python3 {image} "
        f"-c \"from huggingface_hub import snapshot_download; snapshot_download('{repo}'{ap})\"",
        timeout=3600,  # bound HF download stalls so one hang can't wedge the queue
    )
    if r.returncode != 0:
        raise RuntimeError(f"download failed: {r.stderr[-400:]}")
    return round(time.time() - t0, 1)


def hub_dir(repo):
    return os.path.join(CACHE, "hub", "models--" + repo.replace("/", "--"))


def find_gguf(repo, pattern):
    """Locate a downloaded GGUF in the HF cache. Skips MTP/draft/mmproj side-files
    (e.g. unsloth's `MTP/mtp-*.gguf` multi-token-prediction drafts, which aren't
    servable standalone), prefers the main checkpoint at the snapshot root, and
    returns the first shard of a multi-shard quant. Returns None if absent."""
    import glob
    hits = glob.glob(os.path.join(hub_dir(repo), "snapshots", "*", "**", pattern), recursive=True)

    def is_aux(h):
        low, b = h.lower(), os.path.basename(h).lower()
        return ("/mtp/" in low or "/draft/" in low or b.startswith("mtp-")
                or "mmproj" in b or "draft" in b)

    main = [h for h in hits if not is_aux(h)] or hits
    # prefer shallowest path (root over subfolders), then first shard, then name order
    main.sort(key=lambda h: (h.count(os.sep), "00001-of-" not in h, h))
    return main[0] if main else None


def serve_attempt(repo, image, util, maxlen, flags, env=None):
    sh(f"docker rm -f {NAME}", timeout=60)
    env_args = " ".join(f"-e {k}={v}" for k, v in (env or {}).items())
    # Strip the container's CUDA compat libs so the host driver is used —
    # without this, engine workers die instantly on this host (driver 580.xx).
    r = sh(
        f"docker run -d --name {NAME} --gpus all --shm-size 16g -p {PORT}:8000 "
        f"{env_args} -v {CACHE}:/root/.cache/huggingface "
        f"-v {os.path.join(HERE, 'plugins')}:/plugins:ro --entrypoint bash {image} "
        f"-c 'rm -f /etc/ld.so.conf.d/cuda*.conf; ldconfig; "
        f"exec python3 -m vllm.entrypoints.openai.api_server "
        f"--model {repo} --tensor-parallel-size 2 --gpu-memory-utilization {util} "
        f"--max-model-len {maxlen} --max-num-seqs 32 --max-num-batched-tokens 8192 "
        f"--host 0.0.0.0 --port 8000 {flags}'",
        timeout=120,
    )
    if r.returncode != 0:
        return False, r.stderr[-300:]
    deadline = time.time() + 600  # weights are local; 10 min covers load + compile
    while time.time() < deadline:
        try:
            http("/v1/models", timeout=5)
            return True, ""
        except Exception:
            pass
        alive = sh(f"docker inspect -f '{{{{.State.Running}}}}' {NAME}").stdout.strip()
        if alive != "true":
            logs = sh(f"docker logs --tail 200 {NAME} 2>&1").stdout[-6000:]
            return False, logs
        time.sleep(10)
    return False, "health timeout (15 min); container alive but never served\n" + \
        sh(f"docker logs --tail 200 {NAME} 2>&1").stdout[-2000:]


def _poll_health(path, timeout_s, tail=6000):
    """Poll an endpoint until it answers 200, the container dies, or we time out.
    Shared by the sglang and llama.cpp serve paths. Returns (ok, err_or_logtail)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            http(path, timeout=5)
            return True, ""
        except Exception:
            pass
        alive = sh(f"docker inspect -f '{{{{.State.Running}}}}' {NAME}").stdout.strip()
        if alive != "true":
            return False, sh(f"docker logs --tail 200 {NAME} 2>&1").stdout[-tail:]
        time.sleep(10)
    return False, "health timeout; container alive but never served\n" + \
        sh(f"docker logs --tail 200 {NAME} 2>&1").stdout[-2000:]


def serve_sglang(repo, image, mem_frac, ctxlen, flags, env=None):
    """Serve a safetensors checkpoint with SGLang (TP=2) — same weights vLLM eats,
    so it yields a like-for-like engine comparison on the OpenAI API."""
    sh(f"docker rm -f {NAME}", timeout=60)
    env_args = " ".join(f"-e {k}={v}" for k, v in (env or {}).items())
    r = sh(
        f"docker run -d --name {NAME} --gpus all --shm-size 16g -p {PORT}:8000 "
        f"{env_args} -v {CACHE}:/root/.cache/huggingface --entrypoint bash {image} "
        f"-c 'rm -f /etc/ld.so.conf.d/cuda*.conf; ldconfig; "
        f"exec python3 -m sglang.launch_server "
        f"--model-path {repo} --tp 2 --mem-fraction-static {mem_frac} "
        f"--context-length {ctxlen} --host 0.0.0.0 --port 8000 {flags}'",
        timeout=120,
    )
    if r.returncode != 0:
        return False, r.stderr[-300:]
    return _poll_health("/v1/models", 600)


def serve_llamacpp(gguf_path, ctxlen, ngl, n_cpu_moe=None, kv=None):
    """Serve a local GGUF with llama.cpp (both GPUs via --tensor-split 1,1).
    n_cpu_moe offloads MoE expert layers to system RAM so models larger than 48 GB
    VRAM still run (slower). Mirrors the working launch in ~/llm/llamacpp/llamatest.sh."""
    sh(f"docker rm -f {NAME}", timeout=60)
    cont_path = gguf_path.replace(CACHE, "/root/.cache/huggingface")
    extra = f"-ngl {ngl} --tensor-split 1,1 --jinja --threads 24"
    if n_cpu_moe:
        extra += f" --n-cpu-moe {n_cpu_moe}"
    if kv == "q8_0":
        extra += " --cache-type-k q8_0"
    r = sh(
        f"docker run -d --name {NAME} --gpus all --shm-size 16g -p {PORT}:8000 "
        f"-v {CACHE}:/root/.cache/huggingface {LLAMA_IMAGE} "
        f"-m {cont_path} --ctx-size {ctxlen} {extra} --host 0.0.0.0 --port 8000",
        timeout=120,
    )
    if r.returncode != 0:
        return False, r.stderr[-300:]
    # CPU offload makes load slow; 20 min covers big-MoE partial offload
    return _poll_health("/health", 1200)


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


def battery(model, maxlen=32768):
    import concurrent.futures
    import random
    import string

    def salt():
        return "".join(random.choices(string.ascii_lowercase, k=10))

    out = {}
    completion(model, f"[{salt()}] Say hello.", 16)  # warmup + compile

    _, ct, dt = completion(model, f"[{salt()}] Write a detailed 1000-word essay on the history of computing.", 1200)
    out["single_stream_toks"] = round(ct / dt, 1)

    # CPU-offloaded (RAM-assisted) configs decode at a few tok/s — a full concurrency-32
    # sweep with a 16K prefill would run for hours, so scale the heavy probes down and
    # flag the mode. VRAM-resident configs (>SLOW_TOKS) get the full comparable battery.
    slow = out["single_stream_toks"] < SLOW_TOKS
    out["battery_mode"] = "reduced" if slow else "full"

    out["ttft_s"] = round(completion(model, f"[{salt()}] Say hello.", 24, stream=True), 2)

    # size the prefill probe to the served context (must fit inside max_model_len)
    cap = 400 if slow else 1500
    reps = min(cap, max(100, (maxlen - 1200) // 10))
    big = f"[{salt()}] " + ("The quick brown fox jumps over the lazy dog. " * reps)
    pt, _, dt = completion(model, big + "\nReply OK.", 1)
    out["prefill"] = {"prompt_tok": pt, "toks": round(pt / dt)}

    out["sweep"] = []
    conc, gen = ((8,), 128) if slow else ((32,), 256)
    for n in conc:
        prompts = [f"[{salt()} #{i}] Explain topic {i % 8} in technical detail." for i in range(2 * n)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda p: completion(model, p, gen), prompts))
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
    except urllib.error.HTTPError as e:
        out["tool_call"] = ("not_configured" if e.code == 400
                            else f"error: HTTP {e.code}")
    except Exception as e:
        out["tool_call"] = f"error: {repr(e)[:80]}"
    return out


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def lru_prune(exclude_repo=None):
    """Keep the weight cache under CACHE_BUDGET_GB, evicting oldest-used models.
    Root-owned files must be deleted via container."""
    hub = os.path.join(CACHE, "hub")
    if not os.path.isdir(hub):
        return
    entries = []
    for d in os.listdir(hub):
        if not d.startswith("models--"):
            continue
        p = os.path.join(hub, d)
        entries.append((os.path.getmtime(p), d, dir_size(p)))
    total = sum(s for _, _, s in entries)
    image = cfg().get("image_default", "vllm/vllm-openai:latest")
    for mtime, d, size in sorted(entries):
        if total <= CACHE_BUDGET_GB * 1e9:
            break
        if exclude_repo and d == f"models--{slug(exclude_repo)}":
            continue
        print(f"LRU evict {d} ({size/1e9:.0f}GB)", flush=True)
        sh(f"docker run --rm -v {CACHE}:/c --entrypoint rm {image} -rf /c/hub/{d}", timeout=600)
        total -= size


def set_power_cap():
    r = sh(f"sudo -n nvidia-smi -pl {POWER_CAP_W}", timeout=30)
    if r.returncode != 0:
        print(f"WARN: could not set power cap: {r.stderr[-120:]}", flush=True)


def power_limit_now():
    r = sh("nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits")
    try:
        return [round(float(x)) for x in r.stdout.split()]
    except Exception:
        return []


def result_slug(repo, engine, quant):
    """Result filename. Legacy vLLM/native rows keep their original repo-only name so
    the 36 existing files stay valid; every other engine/quant combo is disambiguated."""
    base = repo.replace("/", "--")
    if engine == "vllm" and quant in (None, "native"):
        return base
    q = re.sub(r"[^A-Za-z0-9._-]", "-", str(quant or "native"))
    return f"{base}__{engine}__{q}"


def local_gguf_bytes(repo, pattern):
    import glob
    total = 0
    for h in glob.glob(os.path.join(hub_dir(repo), "snapshots", "*", "**", pattern), recursive=True):
        try:
            total += os.path.getsize(h)  # follows the symlink into blobs/
        except OSError:
            pass
    return total


def run_model(repo, keep=False, image=None, flags=None, skip_check=False,
              engine="vllm", quant=None, gguf_repo=None, gguf_pattern=None,
              n_cpu_moe=None, ngl=999, ctx=None, kv=None, base=None):
    """Full cycle for one (repo, engine, quant): download → serve → battery → record.
    engine ∈ {vllm, sglang, llamacpp}; the battery is engine-agnostic (all speak the
    OpenAI API). llama.cpp reads a GGUF quant (gguf_repo/gguf_pattern) and can offload
    MoE experts to RAM via n_cpu_moe. `base` is the canonical model name used to group
    the cross-engine comparison (defaults to repo)."""
    c = cfg()
    dl_image = image or c.get("image_default", "vllm/vllm-openai:latest")
    serve_image = {"vllm": dl_image, "sglang": SGLANG_IMAGE, "llamacpp": LLAMA_IMAGE}[engine]
    ov = c.get("overrides", {}).get(repo, {})
    if flags is None:
        if engine == "vllm":
            fam = next((f for pat, f in FAMILY_FLAGS if re.search(pat, repo)), "")
            flags = f"{fam} {ov.get('flags', '')}".strip()
        else:
            flags = ov.get("flags", "")
    env = ov.get("env", {})
    quant = quant or "native"
    rec = {"repo": repo, "base": base or repo, "engine": engine, "quant": quant,
           "ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "image": serve_image, "flags": flags, "status": "started"}

    dl_repo = gguf_repo or repo
    try:
        if gpus_busy():
            raise RuntimeError("GPUs busy (production stack running?) — run 'llm stop' first")

        # --- download (whole safetensors repo, or a single GGUF quant) ---
        if engine == "llamacpp":
            rec["gguf_repo"], rec["gguf_pattern"] = dl_repo, gguf_pattern
            print(f"[{repo}] {engine}/{quant} fetching {gguf_pattern} from {dl_repo}...", flush=True)
            rec["download_s"] = download(dl_repo, dl_image, allow_patterns=[gguf_pattern])
            gpath = find_gguf(dl_repo, gguf_pattern)
            if not gpath:
                raise RuntimeError(f"gguf not found after download: {gguf_pattern} in {dl_repo}")
            rec["disk_gb"] = round(local_gguf_bytes(dl_repo, gguf_pattern) / 1e9, 1)
        else:
            size = hf_model_bytes(repo)
            rec["disk_gb"] = round(size / 1e9, 1)
            free = shutil.disk_usage(CACHE).free
            if not skip_check and size * 1.3 > free:
                raise RuntimeError(f"insufficient disk: need {size*1.3/1e9:.0f}GB, have {free/1e9:.0f}GB")
            print(f"[{repo}] {engine}/{quant} downloading {rec['disk_gb']}GB...", flush=True)
            rec["download_s"] = download(repo, dl_image)

        # --- serve (engine-specific; all land on PORT and speak the OpenAI API) ---
        served, served_len = False, 8192
        if engine == "vllm":
            for util, maxlen in LADDER:
                print(f"[{repo}] vllm serve util={util} len={maxlen}", flush=True)
                ok, err = serve_attempt(repo, serve_image, util, maxlen, flags, env)
                if ok:
                    rec["serve_config"] = {"gpu_mem_util": util, "max_model_len": maxlen}
                    served, served_len = True, maxlen
                    break
                rec.setdefault("serve_errors", []).append({"util": util, "len": maxlen, "tail": err[-4000:]})
        elif engine == "sglang":
            for mf, cl in SGLANG_LADDER:
                print(f"[{repo}] sglang serve mem_frac={mf} ctx={cl}", flush=True)
                ok, err = serve_sglang(repo, serve_image, mf, cl, flags, env)
                if ok:
                    rec["serve_config"] = {"mem_fraction_static": mf, "context_length": cl}
                    served, served_len = True, cl
                    break
                rec.setdefault("serve_errors", []).append({"mem_frac": mf, "ctx": cl, "tail": err[-4000:]})
        elif engine == "llamacpp":
            served_len = ctx or 8192
            print(f"[{repo}] llamacpp serve ctx={served_len} ngl={ngl} n_cpu_moe={n_cpu_moe}", flush=True)
            ok, err = serve_llamacpp(gpath, served_len, ngl, n_cpu_moe, kv)
            if ok:
                rec["serve_config"] = {"ctx_size": served_len, "ngl": ngl,
                                       "n_cpu_moe": n_cpu_moe, "kv": kv or "f16"}
                served = True
            else:
                rec.setdefault("serve_errors", []).append({"ctx": served_len, "tail": err[-4000:]})
        if not served:
            rec["status"] = "serve_failed"
            return rec

        try:  # best-effort engine version
            if engine == "vllm":
                rec["engine_version"] = json.load(http("/version", timeout=10)).get("version")
            elif engine == "sglang":
                rec["engine_version"] = json.load(http("/get_server_info", timeout=10)).get("version")
        except Exception:
            pass

        try:
            model_id = json.load(http("/v1/models"))["data"][0]["id"]
        except Exception:
            model_id = repo  # llama.cpp accepts any model field
        mon = HwMon()
        mon.start()
        print(f"[{repo}] running battery...", flush=True)
        rec["battery"] = battery(model_id, served_len)
        mon.stop_flag = True
        rec["hw"] = mon.summary()
        rec["hw"]["power_limit_w"] = power_limit_now()
        agg = rec["battery"]["sweep"][-1] if rec["battery"].get("sweep") else None
        if agg and rec["hw"].get("mean_w"):
            rec["tok_per_joule"] = round(agg["agg_toks"] / rec["hw"]["mean_w"], 2)
        rec["status"] = "ok"
    except Exception as e:
        if rec.get("status") in (None, "started"):
            rec["status"] = "error"
        body = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                body = " body=" + e.read()[:300].decode(errors="replace")
            except Exception:
                pass
        rec["error"] = (repr(e)[:400] + body)[:800]
    finally:
        sh(f"docker rm -f {NAME}", timeout=60)
        if not keep:
            lru_prune(exclude_repo=dl_repo)
        with open(os.path.join(RESULTS, result_slug(repo, engine, quant) + ".json"), "w") as f:
            json.dump(rec, f, indent=2)
        report()
        git_commit(f"results: {repo} [{engine}/{quant}] [{rec['status']}]")
    return rec


def fail_reason(d):
    """One-line cause for a non-ok result. Prefers a curated 'fail_reason' field;
    otherwise classifies the serve-error / error tail into a short human cause."""
    if d.get("fail_reason"):
        return d["fail_reason"]
    if d.get("error"):
        return d["error"].splitlines()[0][:140]
    tail = d["serve_errors"][-1].get("tail", "") if d.get("serve_errors") else ""
    sigs = [
        (r"does not recognize this architecture", "unsupported architecture (no vLLM/Transformers impl)"),
        (r"not compatible with vLLM", "architecture not compatible with vLLM"),
        (r"Invalid type of HuggingFace processor", "incomplete checkpoint: missing multimodal processor files"),
        (r"could not locate think", "reasoning-parser token mismatch"),
        (r"Quantization method .*does not match", "quantization config mismatch"),
        (r"WorkerProc failed to start|Engine core initialization failed", "engine-core init crash during weight load"),
        (r"trust_remote_code", "needs trust_remote_code"),
        (r"out of memory|OutOfMemory", "out of GPU memory"),
    ]
    for pat, msg in sigs:
        if re.search(pat, tail):
            return msg
    for l in reversed(tail.splitlines()):
        if re.search(r"Error|error|raise ", l):
            return l.strip()[-140:]
    return "serve failed (see result json)"


def _tool_class(tc):
    """Bucket a tool-call smoke-test outcome into a chart color class."""
    if tc == "ok":
        return "ok"
    if isinstance(tc, str) and tc.startswith("error"):
        return "error"
    return "neutral"


def render_chart(points):
    """Hand-write an SVG efficiency scatter (stdlib only, no matplotlib).
    x = agg@32 throughput, y = tok/J, dot radius ∝ weights on disk, color by
    tool-call support. Self-contained light panel so it reads on GitHub's light
    AND dark README themes. Written to assets/efficiency.svg."""
    if not points:
        return
    os.makedirs(ASSETS, exist_ok=True)
    W, H = 940, 520
    PAD_L, PAD_R, PAD_T, PAD_B = 74, 212, 66, 70
    px0, py0 = PAD_L, H - PAD_B                       # plot origin (bottom-left)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    xmax = max(p["agg"] for p in points) * 1.06
    ymax = max(p["tokj"] for p in points) * 1.10
    COLOR = {"ok": "#2ca02c", "neutral": "#8a8f98", "error": "#e0952b"}
    ink, grid, panel = "#1b1f24", "#e6e8eb", "#fbfbfc"

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def sx(v):
        return px0 + pw * v / xmax

    def sy(v):
        return py0 - ph * v / ymax

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="{panel}" stroke="#d7dae0"/>')
    s.append(f'<text x="{PAD_L}" y="30" font-size="19" font-weight="700" fill="{ink}">'
             f'Inference efficiency on 2×RTX 4090 — throughput vs energy</text>')
    s.append(f'<text x="{PAD_L}" y="49" font-size="12.5" fill="#5b616b">'
             f'each dot = one served model · dot size scales with weights on disk · upper-right is better</text>')

    # gridlines + ticks
    for i in range(6):
        gx = px0 + pw * i / 5
        s.append(f'<line x1="{gx:.1f}" y1="{PAD_T}" x2="{gx:.1f}" y2="{py0}" stroke="{grid}"/>')
        s.append(f'<text x="{gx:.1f}" y="{py0+18}" font-size="11" fill="#5b616b" '
                 f'text-anchor="middle">{round(xmax*i/5):,}</text>')
        gy = py0 - ph * i / 5
        s.append(f'<line x1="{px0}" y1="{gy:.1f}" x2="{px0+pw}" y2="{gy:.1f}" stroke="{grid}"/>')
        s.append(f'<text x="{px0-8}" y="{gy+4:.1f}" font-size="11" fill="#5b616b" '
                 f'text-anchor="end">{round(ymax*i/5)}</text>')
    s.append(f'<line x1="{px0}" y1="{py0}" x2="{px0+pw}" y2="{py0}" stroke="#b7bcc4"/>')
    s.append(f'<line x1="{px0}" y1="{PAD_T}" x2="{px0}" y2="{py0}" stroke="#b7bcc4"/>')
    s.append(f'<text x="{px0+pw/2}" y="{H-24}" font-size="12.5" fill="{ink}" '
             f'text-anchor="middle">aggregate throughput under concurrent decode (tok/s)</text>')
    s.append(f'<text x="22" y="{PAD_T+ph/2}" font-size="12.5" fill="{ink}" text-anchor="middle" '
             f'transform="rotate(-90 22 {PAD_T+ph/2:.1f})">efficiency — tokens per joule</text>')

    # marker shape encodes engine; fill color encodes tool-call support
    def marker(cx, cy, r, engine, color):
        common = f'fill="{color}" fill-opacity="0.72" stroke="#ffffff" stroke-width="1"'
        if engine == "llamacpp":
            return (f'<polygon points="{cx:.1f},{cy-r:.1f} {cx-r:.1f},{cy+r*0.72:.1f} '
                    f'{cx+r:.1f},{cy+r*0.72:.1f}" {common}/>')
        if engine == "sglang":
            return f'<rect x="{cx-r:.1f}" y="{cy-r:.1f}" width="{2*r:.1f}" height="{2*r:.1f}" rx="1.5" {common}/>'
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {common}/>'

    # largest drawn first so small markers stay visible on top
    for p in sorted(points, key=lambda p: -p["gb"]):
        r = 4.5 + (p["gb"] ** 0.5) * 1.7
        s.append(marker(sx(p["agg"]), sy(p["tokj"]), r, p.get("engine", "vllm"), COLOR[p["cls"]]))

    # label the extremes only (keeps the dense cluster legible): the top-4 by
    # tok/J plus the single least-efficient model — best vs worst, well separated
    by_tokj = sorted(points, key=lambda p: -p["tokj"])
    labeled = by_tokj[:4] + [by_tokj[-1]]
    for p in labeled:
        x, y = sx(p["agg"]), sy(p["tokj"])
        right = x < px0 + pw * 0.7
        tx, anchor = (x + 10, "start") if right else (x - 10, "end")
        s.append(f'<text x="{tx:.1f}" y="{y+3.5:.1f}" font-size="10.5" fill="{ink}" '
                 f'text-anchor="{anchor}">{esc(p["name"])}</text>')

    # legend
    lx, ly = px0 + pw + 24, PAD_T + 6
    legend = [("ok", "tool-calls parsed"), ("neutral", "no / unconfigured"),
              ("error", "tool test errored")]
    s.append(f'<text x="{lx}" y="{ly}" font-size="11.5" font-weight="700" fill="{ink}">'
             f'tool-call support</text>')
    for i, (cls, lab) in enumerate(legend):
        cy = ly + 20 + i * 20
        s.append(f'<circle cx="{lx+6}" cy="{cy-4:.1f}" r="6" fill="{COLOR[cls]}" '
                 f'fill-opacity="0.72" stroke="#ffffff"/>')
        s.append(f'<text x="{lx+20}" y="{cy}" font-size="11" fill="#3b414a">{lab}</text>')
    sz_y = ly + 20 + len(legend) * 20 + 14
    s.append(f'<text x="{lx}" y="{sz_y}" font-size="11.5" font-weight="700" fill="{ink}">'
             f'dot size = GB on disk</text>')
    for i, gb in enumerate((2, 12, 32)):
        cy = sz_y + 22 + i * 22
        r = 4.5 + (gb ** 0.5) * 1.7
        s.append(f'<circle cx="{lx+12}" cy="{cy-4:.1f}" r="{r:.1f}" fill="#8a8f98" '
                 f'fill-opacity="0.5" stroke="#ffffff"/>')
        s.append(f'<text x="{lx+30}" y="{cy}" font-size="11" fill="#3b414a">{gb} GB</text>')

    eg_y = sz_y + 22 + len((2, 12, 32)) * 22 + 8
    s.append(f'<text x="{lx}" y="{eg_y}" font-size="11.5" font-weight="700" fill="{ink}">'
             f'engine (shape)</text>')
    for i, (eng, lab) in enumerate((("vllm", "vLLM"), ("llamacpp", "llama.cpp"), ("sglang", "SGLang"))):
        cy = eg_y + 20 + i * 20
        s.append(marker(lx + 6, cy - 4, 6, eng, "#8a8f98"))
        s.append(f'<text x="{lx+20}" y="{cy}" font-size="11" fill="#3b414a">{lab}</text>')

    s.append('</svg>\n')
    with open(os.path.join(ASSETS, "efficiency.svg"), "w") as f:
        f.write("\n".join(s))


def _agg(b):
    """(aggregate tok/s, concurrency) from the last sweep rung, or (None, None)."""
    sw = b.get("sweep", [])
    return (sw[-1]["agg_toks"], sw[-1]["concurrency"]) if sw else (None, None)


def report():
    oks, fails = [], []
    for fn in sorted(os.listdir(RESULTS)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(RESULTS, fn)))
        d["_engine"], d["_quant"] = d.get("engine", "vllm"), d.get("quant", "native")
        (oks if d["status"] == "ok" else fails).append(d)

    # --- main leaderboard (all engines/quants, sorted by aggregate throughput) ---
    ok_rows, points = [], []
    for d in oks:
        b = d.get("battery", {})
        agg, _ = _agg(b)
        star = "\\*" if b.get("battery_mode") == "reduced" else ""
        ok_rows.append((agg or 0,
            f"| {d['repo']} | {d['_engine']} | {d['_quant']} | {d.get('disk_gb','')} | "
            f"{b.get('single_stream_toks','—')} | {b.get('prefill',{}).get('toks','—')} | "
            f"{(str(agg)+star) if agg else '—'} | {d.get('hw',{}).get('max_vram_gb','—')} | "
            f"{d.get('hw',{}).get('mean_w','—')} | {d.get('tok_per_joule','—')} | "
            f"{b.get('tool_call','—')} |"))
        if agg and d.get("tok_per_joule") and d.get("disk_gb"):
            points.append({"name": d["repo"].split("/")[-1], "agg": agg,
                           "tokj": d["tok_per_joule"], "gb": d["disk_gb"],
                           "cls": _tool_class(b.get("tool_call")), "engine": d["_engine"]})
    ok_rows.sort(key=lambda r: -(r[0] or 0))
    table = ("| model | engine | quant | GB | 1-stream tok/s | prefill tok/s | agg tok/s | "
             "VRAM GB | mean W | tok/J | tools |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
             + "\n".join(r for _, r in ok_rows))
    if any(d.get("battery", {}).get("battery_mode") == "reduced" for d in oks):
        table += ("\n\n<sub>\\* reduced battery (concurrency 8, shorter generation) — a "
                  "CPU-offloaded config too slow for the full concurrency-32 sweep.</sub>")
    if fails:
        fr = sorted(f"| {d['repo']} | {d['_engine']} | {d['_quant']} | {d.get('disk_gb','—')} | "
                    f"{d['status']} | {fail_reason(d)} |" for d in fails)
        table += ("\n\n**Did not serve on this rig** — no throughput data; recorded with cause:\n\n"
                  "| model | engine | quant | GB | status | identified cause |\n"
                  "|---|---|---|---|---|---|\n" + "\n".join(fr))

    # --- cross-engine comparison: base models served by >1 engine ---
    groups = {}
    for d in oks:
        groups.setdefault(d.get("base") or d["repo"], []).append(d)
    xe_blocks = []
    for repo in sorted(groups):
        rows = groups[repo]
        if len({d["_engine"] for d in rows}) < 2:
            continue
        rows.sort(key=lambda d: (d["_engine"], -(d.get("battery", {}).get("single_stream_toks") or 0)))
        body = []
        for d in rows:
            b = d.get("battery", {})
            agg, _ = _agg(b)
            body.append(f"| {d['_engine']} | {d['_quant']} | `{d['repo']}` | {d.get('disk_gb','—')} | "
                        f"{b.get('single_stream_toks','—')} | {agg or '—'} | "
                        f"{d.get('hw',{}).get('max_vram_gb','—')} | {d.get('tok_per_joule','—')} | "
                        f"{b.get('tool_call','—')} |")
        xe_blocks.append(f"**{repo}**\n\n| engine | quant | source | GB | 1-stream tok/s | agg tok/s | "
                         f"VRAM GB | tok/J | tools |\n|---|---|---|---|---|---|---|---|---|\n"
                         + "\n".join(body))
    xengine = ("\n\n".join(xe_blocks) if xe_blocks else
               "_No base model has been served by more than one engine yet._")

    render_chart(points)
    served, failed = len(ok_rows), len(fails)
    engines = sorted({d["_engine"] for d in oks + fails})
    summary = (f"**{served + failed} configs tested · {served} served · {failed} did-not-serve · "
               f"engines: {', '.join(engines)} · "
               f"updated {time.strftime('%Y-%m-%d', time.gmtime())}**")
    chart = "![Efficiency: aggregate throughput vs tokens-per-joule, by engine](assets/efficiency.svg)"

    rd = os.path.join(HERE, "README.md")
    txt = open(rd).read()

    def inject(name, content):
        return re.sub(rf"(<!--{name}:BEGIN-->).*(<!--{name}:END-->)",
                      lambda m: m.group(1) + "\n" + content + "\n" + m.group(2), txt, flags=re.S)

    txt = inject("SUMMARY", summary)
    txt = inject("CHART", chart)
    txt = inject("RESULTS", table)
    txt = inject("XENGINE", xengine)
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
    p.add_argument("--engine", default="vllm", choices=["vllm", "sglang", "llamacpp"])
    p.add_argument("--quant")
    p.add_argument("--gguf-repo")
    p.add_argument("--gguf-pattern")
    p.add_argument("--n-cpu-moe", type=int)
    p.add_argument("--ngl", type=int, default=999)
    p.add_argument("--ctx", type=int)
    p.add_argument("--kv")
    p.add_argument("--base")
    p.add_argument("--skip-download-check", action="store_true")
    sub.add_parser("queue")
    sub.add_parser("report")
    a = ap.parse_args()

    # fields a queue entry (or CLI run) may carry through to run_model()
    ekeys = ("engine", "quant", "gguf_repo", "gguf_pattern", "n_cpu_moe", "ngl", "ctx", "kv", "base")

    if a.cmd == "run":
        set_power_cap()
        ekw = {k: getattr(a, k) for k in ekeys}
        rec = run_model(a.repo, keep=a.keep, image=a.image, flags=a.flags,
                        skip_check=a.skip_download_check, **ekw)
        print(json.dumps({k: v for k, v in rec.items() if k != "serve_errors"}, indent=2))
    elif a.cmd == "queue":
        set_power_cap()
        c = cfg()
        img = c.get("image_default", "vllm/vllm-openai:latest")

        def prefetch(entry):
            # prefetch weights for the next entry: a single GGUF for llamacpp, else the repo
            repo = entry.get("gguf_repo") or entry["repo"]
            try:
                print(f"[prefetch] {repo}", flush=True)
                download(repo, img, allow_patterns=[entry["gguf_pattern"]]
                         if entry.get("engine") == "llamacpp" else None)
            except Exception as e:
                print(f"[prefetch] {repo} failed: {repr(e)[:80]}", flush=True)

        q = c["queue"]
        for i, entry in enumerate(q):
            if entry.get("done"):
                continue
            nxt = next((e for e in q[i + 1:] if not e.get("done")), None)
            th = threading.Thread(target=prefetch, args=(nxt,), daemon=True) if nxt else None
            if th:
                th.start()
            ekw = {k: entry[k] for k in ekeys if k in entry}
            rec = run_model(entry["repo"], keep=entry.get("keep", False),
                            flags=entry.get("flags"), **ekw)
            entry["done"] = True
            entry["status"] = rec["status"]
            with open(os.path.join(HERE, "models.json"), "w") as f:
                json.dump(c, f, indent=2)
            if th:
                th.join(timeout=1800)
    elif a.cmd == "report":
        report()
        git_commit("regenerate report")


if __name__ == "__main__":
    main()
