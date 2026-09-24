"""Cache JVS parallel100 (100 speakers x 100 same sentences) with OpenJTalk-predicted accents as labels.

Labels are noisy (~17% of phrases wrong), but wrong in the same way for every speaker, so
evaluation can drop phrases that most speakers 'fail' (label error) - see eval_jvs.py."""
import pickle, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import accent, align, pitch

JVS = Path(__file__).parent / "vendor" / "jvs" / "jvs_ver1"
speakers = sorted(p.name for p in JVS.glob("jvs*") if p.is_dir())[: int(sys.argv[1])]
per_spk = int(sys.argv[2]) if len(sys.argv) > 2 else 100
out = []
for spk in speakers:
    d = JVS / spk / "parallel100"
    trans = dict(l.strip().split(":", 1) for l in open(d / "transcripts_utf8.txt", encoding="utf-8") if ":" in l)
    for uid in sorted(trans)[:per_spk]:
        wp = d / "wav24kHz16bit" / f"{uid}.wav"
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
        out.append(dict(uid=f"{spk}/{uid}", spk=spk, sent=uid,
                        gold=[(p.reading, p.acc, [m.devoiced for m in p.moras]) for p in phrases],
                        spans=[(s.start, s.end) for s in sp], t=tr.t, hz=tr.hz, conf=tr.conf))
    print(spk, len(out), file=sys.stderr, flush=True)
    pickle.dump(out, open("jvs_cache.pkl", "wb"))
