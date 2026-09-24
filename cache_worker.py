"""One caching worker: processes a shard of (speaker, sentence) JVS jobs -> pickle.

Usage: cache_worker.py JOBS.json OUT.pkl THREADS DEVICE"""
import json, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
jobs_path, out_path, threads, device = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
import torch
torch.set_num_threads(threads)
from pathlib import Path
import numpy as np, soundfile as sf
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import accent, align, pitch
align.DEVICE = device

JVS = Path(__file__).parent / "vendor" / "jvs" / "jvs_ver1"
trans = dict(l.strip().split(":", 1) for l in open(JVS / "jvs001/parallel100/transcripts_utf8.txt", encoding="utf-8") if ":" in l)
out, t0 = [], time.time()
for spk, uid in json.load(open(jobs_path)):
    wp = JVS / spk / "parallel100" / "wav24kHz16bit" / f"{uid}.wav"
    if not wp.exists():
        continue
    phrases, _ = accent.analyze(trans[uid])
    wav, sr = sf.read(wp, dtype="float32")
    pad = np.zeros(4800, np.float32); a16 = np.concatenate([pad, align.to16k(wav, sr), pad])
    try:
        sp = align.force_align(align.logprobs(a16), [m.hira for p in phrases for m in p.moras])
    except Exception as e:
        print("SKIP", spk, uid, e, file=sys.stderr); continue
    if sp is None:
        continue
    tr = pitch.track(a16, 16000, conf_thresh=0.0)
    out.append(dict(uid=f"{spk}/{uid}", spk=spk, sent=uid, dur=len(wav) / sr,
                    gold=[(p.reading, p.acc, [m.devoiced for m in p.moras]) for p in phrases],
                    spans=[(s.start, s.end) for s in sp], t=tr.t, hz=tr.hz, conf=tr.conf))
pickle.dump(out, open(out_path, "wb"))
print(json.dumps(dict(n=len(out), secs=time.time() - t0, audio_s=sum(u["dur"] for u in out))))
