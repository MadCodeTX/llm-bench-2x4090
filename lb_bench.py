#!/usr/bin/env python3
"""Experiment: 2x single-GPU vLLM replicas (data-parallel, nginx round-robin) vs.
the standard 1x tensor-parallel-2 serving mode used by bench.py.

Rationale: this rig has no NVLink, so tensor-parallel-2 pays a per-token PCIe
all-reduce on every forward pass. For models small enough to fit entirely on one
24GB card, that communication tax buys nothing — an independent TP=1 replica per
GPU should have equal or lower per-token latency, and two of them running
concurrently should roughly double aggregate capacity versus one TP=2 engine.

Measures, for each repo already benchmarked by bench.py (TP=2 baseline in results/):
  - combined single-stream throughput: two concurrent long-generation requests,
    one pinned to each replica directly (not through the proxy) — the real
    achievable "two users, one per GPU" peak.
  - aggregate throughput at concurrency 32 and 64 through an nginx round-robin
    proxy in front of both replicas.
Writes results-lb/<slug>.json and prints a comparison against results/<slug>.json.

Usage: lb_bench.py <hf-repo> [<hf-repo> ...]
"""
import concurrent.futures
import json
import os
import random
import string
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench

RESULTS_LB = os.path.join(bench.HERE, "results-lb")
os.makedirs(RESULTS_LB, exist_ok=True)

R0, R1 = 8201, 8202       # per-replica ports (host-mapped)
LB_PORT = bench.PORT      # nginx proxy listens where bench.http() already points
N0, N1 = "llmbench-lb0", "llmbench-lb1"
NGINX = "llmbench-lb-nginx"
NGINX_CONF = "/tmp/llmbench-nginx-lb.conf"


def salt():
    return "".join(random.choices(string.ascii_lowercase, k=10))


def replica_http(port, path, payload=None, timeout=300):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}" + path,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode() if payload else None,
    )
    return urllib.request.urlopen(req, timeout=timeout)


def replica_completion(port, model, prompt, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7}
    t0 = time.time()
    resp = json.load(replica_http(port, "/v1/chat/completions", payload, timeout=600))
    u = resp["usage"]
    return u["prompt_tokens"], u["completion_tokens"], time.time() - t0


def serve_replica(repo, image, util, maxlen, flags, env, gpu_id, port, name):
    bench.sh(f"docker rm -f {name}", timeout=60)
    env_args = " ".join(f"-e {k}={v}" for k, v in (env or {}).items())
    r = bench.sh(
        f"docker run -d --name {name} --gpus device={gpu_id} --shm-size 16g -p {port}:8000 "
        f"{env_args} -v {bench.CACHE}:/root/.cache/huggingface "
        f"-v {os.path.join(bench.HERE, 'plugins')}:/plugins:ro --entrypoint bash {image} "
        f"-c 'rm -f /etc/ld.so.conf.d/cuda*.conf; ldconfig; "
        f"exec python3 -m vllm.entrypoints.openai.api_server "
        f"--model {repo} --tensor-parallel-size 1 --gpu-memory-utilization {util} "
        f"--max-model-len {maxlen} --max-num-seqs 32 --max-num-batched-tokens 8192 "
        f"--host 0.0.0.0 --port 8000 {flags}'",
        timeout=120,
    )
    if r.returncode != 0:
        return False, r.stderr[-300:]
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            replica_http(port, "/v1/models", timeout=5)
            return True, ""
        except Exception:
            pass
        alive = bench.sh(f"docker inspect -f '{{{{.State.Running}}}}' {name}").stdout.strip()
        if alive != "true":
            logs = bench.sh(f"docker logs --tail 200 {name} 2>&1").stdout[-4000:]
            return False, logs
        time.sleep(10)
    return False, "health timeout (10 min)\n" + bench.sh(f"docker logs --tail 200 {name} 2>&1").stdout[-2000:]


def start_nginx():
    bench.sh(f"docker rm -f {NGINX}", timeout=60)
    conf = f"""events {{}}
http {{
  upstream vllm_lb {{
    server 127.0.0.1:{R0};
    server 127.0.0.1:{R1};
  }}
  server {{
    listen {LB_PORT};
    proxy_buffering off;
    proxy_read_timeout 600s;
    location / {{
      proxy_pass http://vllm_lb;
      proxy_http_version 1.1;
    }}
  }}
}}
"""
    with open(NGINX_CONF, "w") as f:
        f.write(conf)
    r = bench.sh(
        f"docker run -d --name {NGINX} --network host "
        f"-v {NGINX_CONF}:/etc/nginx/nginx.conf:ro nginx:alpine",
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"nginx failed to start: {r.stderr[-300:]}")
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            bench.http("/v1/models", timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("nginx proxy never came up")


def teardown():
    for n in (N0, N1, NGINX):
        bench.sh(f"docker rm -f {n}", timeout=60)


def lb_battery(model):
    out = {}
    # warm both engines directly (cudagraph capture / compile)
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda p: replica_completion(p, model, f"[{salt()}] Say hello.", 16), (R0, R1)))

    # combined single-stream peak: one concurrent long-generation request per GPU,
    # addressed directly (not through the LB) to guarantee an even 1-per-replica split
    def one_stream(port):
        return replica_completion(
            port, model,
            f"[{salt()}] Write a detailed 1000-word essay on the history of computing.", 1200)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        res = list(ex.map(one_stream, (R0, R1)))
    wall = time.time() - t0
    out["single_stream_toks_per_replica"] = [round(c / d, 1) for _, c, d in res]
    out["single_stream_toks_combined"] = round(sum(c for _, c, _ in res) / wall, 1)

    # aggregate throughput through the round-robin proxy
    out["sweep"] = []
    for n in (32, 64):
        prompts = [f"[{salt()} #{i}] Explain topic {i % 8} in technical detail." for i in range(2 * n)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda p: bench.completion(model, p, 256), prompts))
        wall = time.time() - t0
        total = sum(c for _, c, _ in res)
        out["sweep"].append({"concurrency": n, "agg_toks": round(total / wall)})
    return out


def run_lb(repo):
    c = bench.cfg()
    image = c.get("image_default")
    ov = c.get("overrides", {}).get(repo, {})
    flags = ov.get("flags", "")
    env = ov.get("env", {})
    base_rec = None
    base_path = os.path.join(bench.RESULTS, bench.slug(repo) + ".json")
    if os.path.exists(base_path):
        base_rec = json.load(open(base_path))
    util, maxlen = 0.92, 32768
    if base_rec and base_rec.get("serve_config"):
        util = base_rec["serve_config"].get("gpu_mem_util", util)
        maxlen = base_rec["serve_config"].get("max_model_len", maxlen)

    rec = {"repo": repo, "mode": "2x-tp1-replica-lb",
           "ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "image": image, "flags": flags, "status": "started"}
    try:
        if bench.gpus_busy():
            raise RuntimeError("GPUs busy — stop production stack / other containers first")
        print(f"[{repo}] serving replica 0 (GPU0, TP=1)...", flush=True)
        ok0, err0 = serve_replica(repo, image, util, maxlen, flags, env, 0, R0, N0)
        if not ok0:
            raise RuntimeError(f"replica0 serve failed: {err0[-500:]}")
        print(f"[{repo}] serving replica 1 (GPU1, TP=1)...", flush=True)
        ok1, err1 = serve_replica(repo, image, util, maxlen, flags, env, 1, R1, N1)
        if not ok1:
            raise RuntimeError(f"replica1 serve failed: {err1[-500:]}")
        rec["serve_config"] = {"gpu_mem_util": util, "max_model_len": maxlen,
                                "tensor_parallel_size": 1, "replicas": 2}

        print(f"[{repo}] starting nginx round-robin proxy...", flush=True)
        start_nginx()

        model_id = json.load(bench.http("/v1/models"))["data"][0]["id"]
        mon = bench.HwMon()
        mon.start()
        print(f"[{repo}] running LB battery...", flush=True)
        rec["battery"] = lb_battery(model_id)
        mon.stop_flag = True
        rec["hw"] = mon.summary()
        rec["hw"]["power_limit_w"] = bench.power_limit_now()
        agg32 = next((s["agg_toks"] for s in rec["battery"]["sweep"] if s["concurrency"] == 32), None)
        agg64 = next((s["agg_toks"] for s in rec["battery"]["sweep"] if s["concurrency"] == 64), None)
        if agg32 and rec["hw"].get("mean_w"):
            rec["tok_per_joule_agg32"] = round(agg32 / rec["hw"]["mean_w"], 2)
        if agg64 and rec["hw"].get("mean_w"):
            rec["tok_per_joule_agg64"] = round(agg64 / rec["hw"]["mean_w"], 2)
        rec["status"] = "ok"
    except Exception as e:
        if rec.get("status") in (None, "started"):
            rec["status"] = "error"
        rec["error"] = repr(e)[:800]
    finally:
        teardown()
        with open(os.path.join(RESULTS_LB, bench.slug(repo) + ".json"), "w") as f:
            json.dump(rec, f, indent=2)
    return rec


def compare(repo):
    base_path = os.path.join(bench.RESULTS, bench.slug(repo) + ".json")
    lb_path = os.path.join(RESULTS_LB, bench.slug(repo) + ".json")
    if not (os.path.exists(base_path) and os.path.exists(lb_path)):
        return {"repo": repo, "note": "missing baseline or LB result"}
    base = json.load(open(base_path))
    lb = json.load(open(lb_path))
    bb, lbb = base.get("battery", {}), lb.get("battery", {})
    base_agg32 = next((s["agg_toks"] for s in bb.get("sweep", []) if s["concurrency"] == 32), None)
    lb_agg32 = next((s["agg_toks"] for s in lbb.get("sweep", []) if s["concurrency"] == 32), None)
    lb_agg64 = next((s["agg_toks"] for s in lbb.get("sweep", []) if s["concurrency"] == 64), None)
    return {
        "repo": repo,
        "single_stream_tp2": bb.get("single_stream_toks"),
        "single_stream_lb_combined": lbb.get("single_stream_toks_combined"),
        "single_stream_lb_per_replica": lbb.get("single_stream_toks_per_replica"),
        "agg32_tp2": base_agg32,
        "agg32_lb": lb_agg32,
        "agg64_lb": lb_agg64,
        "mean_w_tp2": base.get("hw", {}).get("mean_w"),
        "mean_w_lb": lb.get("hw", {}).get("mean_w"),
    }


if __name__ == "__main__":
    repos = sys.argv[1:]
    if not repos:
        print("usage: lb_bench.py <hf-repo> [...]")
        sys.exit(1)
    bench.set_power_cap()
    comparisons = []
    for repo in repos:
        print(f"\n=== {repo} ===", flush=True)
        rec = run_lb(repo)
        print(json.dumps({k: v for k, v in rec.items() if k != "serve_errors"}, indent=2))
        comparisons.append(compare(repo))
    print("\n=== COMPARISON (TP=2 baseline vs 2x TP=1 replica + LB) ===")
    print(json.dumps(comparisons, indent=2))
