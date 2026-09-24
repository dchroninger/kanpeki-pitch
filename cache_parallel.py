"""Run JVS caching sharded across W workers x T threads. Prints throughput JSON.

Usage: cache_parallel.py JOBS.json OUT.pkl WORKERS THREADS [DEVICE]"""
import json, pickle, subprocess, sys, tempfile, time
from pathlib import Path

jobs = json.load(open(sys.argv[1])); out = sys.argv[2]
W, T = int(sys.argv[3]), int(sys.argv[4]); dev = sys.argv[5] if len(sys.argv) > 5 else "cpu"
tmp = Path(tempfile.mkdtemp(dir="."))
t0 = time.time()
procs = []
for w in range(W):
    shard = jobs[w::W]
    (tmp / f"j{w}.json").write_text(json.dumps(shard))
    procs.append(subprocess.Popen([sys.executable, "cache_worker.py", str(tmp / f"j{w}.json"), str(tmp / f"o{w}.pkl"), str(T), dev],
                                  stdout=subprocess.PIPE, stderr=open(tmp / f"e{w}.txt", "w"), text=True))
stats = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
merged = [u for w in range(W) for u in pickle.load(open(tmp / f"o{w}.pkl", "rb"))]
pickle.dump(merged, open(out, "wb"))
wall = time.time() - t0
audio = sum(s["audio_s"] for s in stats)
print(json.dumps(dict(workers=W, threads=T, device=dev, recordings=len(merged), wall_s=round(wall, 1),
                      rec_per_s=round(len(merged) / wall, 2), audio_x_realtime=round(audio / wall, 1))))
for f in tmp.iterdir(): f.unlink()
tmp.rmdir()
