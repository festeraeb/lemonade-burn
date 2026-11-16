#!/usr/bin/env python3
"""limits_variant.py SEAT BIN CTX PORT LABEL [prompt sizes...]   (run on c4; the seat's own cards must be free)
SEAT A = Tiel on AF+D8 (split 19,21), B = Qwen3.6 on 3C+3B+86 (split 6,17,17); chunk 256.  Starts BIN (a file in the wip/build bin dir), waits for /health,
runs decode x3 then each prompt size, polls nvidia-smi for peak VRAM per card, appends JSON lines to ~/lbtest/limits.jsonl, kills the server PID (never pkill)."""
import json, os, subprocess, sys, threading, time, signal, urllib.request
seat, binp, ctx, port, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sizes = [int(x) for x in sys.argv[6:]] or [2000, 3800, 6000]
BENCH = "/mnt/t440/data/builds/lb-qwen35-limits-wt/scripts/qwen/limits_bench.py"
SEATS = {
 "A": dict(cards=["GPU-fd9b03e5-aed9-cbca-a57b-efc90d61addd", "GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab"], split="19,21", gguf="Tiel-Coder-35B-A3B-UD-Q5_K_XL.gguf"),
 "B": dict(cards=["GPU-560cc467-2cb5-dc80-479a-e3f891fb1929", "GPU-c012cc70-3f06-1718-84e2-073aa275f0af", "GPU-0c4b4f5d-5f54-46db-be25-f6ffdf75ac79"], split="6,17,17", gguf="Qwen3.6-35B-A3B-Q5_K_M.gguf"),
}
S = SEATS[seat]
D = os.environ.get("LB_D", "/mnt/t440/raid0/engines/lemonade-burn/20261116-qwen35-limits-wip")
env = dict(os.environ, LD_LIBRARY_PATH=D + "/lib", CUDA_PATH=D + "/cuda-path", CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDARC_CUDA_VERSION="12080",
           CUDA_VISIBLE_DEVICES=",".join(S["cards"]), QWEN_SPLIT=S["split"], LB_QW_PREFILL_CHUNK=os.environ.get("CHUNK", "256"), LB_QW_TRACE="1")
for kv in os.environ.get("EXTRA", "").split():
    k, v = kv.split("=", 1); env[k] = v
out = open(os.path.expanduser("~/lbtest/limits.jsonl"), "a")
def emit(**kw):
    kw.update(label=label, seat=seat, ctx=ctx, bin=os.path.basename(binp)); out.write(json.dumps(kw) + "\n"); out.flush(); print(json.dumps(kw), flush=True)
def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = {}
    for l in r.strip().splitlines():
        u, m, t = [x.strip() for x in l.split(",")]; d[u] = (int(m), int(t))
    return d
peak = {c: [0, 0] for c in S["cards"]}
stop = False
def poll():
    while not stop:
        try:
            d = smi()
            for c in S["cards"]:
                if c in d: peak[c] = [max(peak[c][0], d[c][0]), max(peak[c][1], d[c][1])]
                if c in d and d[c][1] >= 85: emit(warn="temp>=85 on " + c[:12], t=d[c][1]); os.kill(pid, signal.SIGTERM)
        except Exception: pass
        time.sleep(0.5)
free = smi()
busy = [c for c in S["cards"] if free.get(c, (0, 0))[0] > 300]
if busy: sys.exit(f"cards not free: {busy}")
log = open(os.path.expanduser(f"~/lbtest/limits-{label}.log"), "w")
p = subprocess.Popen([D + "/bin/" + binp, "--model", "/mnt/t440/raid0/models/" + S["gguf"], "--gpu", "--ctx", str(ctx), "--host", "127.0.0.1", "--port", str(port), "--model-id", "test"], env=env, stdout=log, stderr=log, cwd=os.path.expanduser("~/lbtest"))
pid = p.pid
t0 = time.time()
threading.Thread(target=poll, daemon=True).start()
ok = False
while time.time() - t0 < 900 and p.poll() is None:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(); ok = True; break
    except Exception: time.sleep(2)
if not ok:
    emit(event="start_failed", rc=p.poll()); stop = True; p.kill(); sys.exit(1)
emit(event="ready", load_s=round(time.time() - t0, 1), pid=pid, vram_mib=[smi()[c][0] for c in S["cards"]])
def run(args):
    r = subprocess.run(["python3", BENCH, str(port), "test"] + args, capture_output=True, text=True, timeout=1800)
    return [json.loads(l) for l in r.stdout.strip().splitlines() if l.startswith("{")], r.stderr[-300:]
try:
    res, err = run(["decode", "256"]); emit(event="decode", runs=res, err=err, vram_mib=[smi()[c][0] for c in S["cards"]])
    if os.environ.get("THINK"):
        r = subprocess.run(["python3", BENCH.replace("limits_bench.py", "think_check.py"), str(port), "test"], capture_output=True, text=True, timeout=3000)
        for l in r.stdout.strip().splitlines():
            emit(event="think", **json.loads(l))
        if r.stderr: emit(event="think_err", err=r.stderr[-300:])
    for i, n in enumerate([x for x in sizes if x > 0]):
        try:
            res, err = run(["prompt", str(n), str(i + 1)]); emit(event="prompt", want=n, runs=res, err=err, vram_mib=[smi()[c][0] for c in S["cards"]], peak_mib=[peak[c][0] for c in S["cards"]])
        except Exception as e:
            emit(event="prompt_failed", want=n, err=str(e)[:200]); break
        if p.poll() is not None: emit(event="server_died", rc=p.poll()); break
finally:
    stop = True
    emit(event="done", peak_mib=[peak[c][0] for c in S["cards"]], peak_temp=[peak[c][1] for c in S["cards"]])
    if p.poll() is None:
        p.terminate()
        try: p.wait(30)
        except Exception: p.kill()
