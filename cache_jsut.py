"""Cache pitch track + mora spans + gold accents for JSUT utterances (for fast grader tuning)."""
import pickle, random, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, yaml
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import align, pitch
from eval_jsut import parse_label, J

labels = yaml.safe_load(open(J / "jsut-label/e2e_symbol/katakana.yaml"))
ids = sorted(labels); random.Random(42).shuffle(ids)
out = []
for uid in ids[:int(sys.argv[1])]:
    gold = parse_label(labels[uid])
    wav, sr = sf.read(J / "jsut_ver1.1/basic5000/wav" / f"{uid}.wav", dtype="float32")
    pad = np.zeros(4800, np.float32); a16 = np.concatenate([pad, align.to16k(wav, sr), pad])
    try:
        sp = align.force_align(align.logprobs(a16), [m.hira for p in gold for m in p.moras])
    except Exception as e:
        print("SKIP", uid, e, file=sys.stderr); continue
    tr = pitch.track(a16, 16000, conf_thresh=0.0)
    if sp is None:
        continue
    out.append(dict(uid=uid, gold=[(p.reading, p.acc, [m.devoiced for m in p.moras]) for p in gold],
                    spans=[(s.start, s.end) for s in sp], t=tr.t, hz=tr.hz, conf=tr.conf))
    print(uid, file=sys.stderr, flush=True)
    if len(out) % 250 == 0:
        pickle.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "jsut_cache.pkl", "wb"))
pickle.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "jsut_cache.pkl", "wb"))
print(len(out))
