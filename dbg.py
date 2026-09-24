import warnings, re; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, yaml
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import align
from eval_jsut import parse_label, J
labels = yaml.safe_load(open(J/"jsut-label/e2e_symbol/katakana.yaml"))
for uid in ["BASIC5000_0001", "BASIC5000_0004"]:
    gold = parse_label(labels[uid])
    wav, sr = sf.read(J/"jsut_ver1.1/basic5000/wav"/f"{uid}.wav", dtype="float32")
    pad = np.zeros(4800, np.float32); a16 = np.concatenate([pad, align.to16k(wav, sr), pad])
    sp = align.force_align(align.logprobs(a16), [m.hira for p in gold for m in p.moras])
    lab = [l.split() for l in open(J/"jsut-label/labels/basic5000"/f"{uid}.lab")]
    ph = [(int(a)/1e7, re.search(r"-(.+?)\+", c).group(1)) for a, b, c in lab]
    print(uid)
    print("  ours :", " ".join(f"{m.kana}@{s.start-0.3:.2f}" for m, s in zip([m for p in gold for m in p.moras], sp))[:400])
    print("  label:", " ".join(f"{p}@{t:.2f}" for t, p in ph)[:400])
