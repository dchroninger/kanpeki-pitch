import warnings; warnings.filterwarnings("ignore")
import numpy as np
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import accent, align, pitch, tts
phrases,_ = accent.analyze("箸が好きです")
for vvm,sid in (("0",2),("4",11)):
    tts.load(vvm)
    q=tts.synth().create_audio_query_from_kana(tts.to_kana(phrases,[2,2]),sid); tts.enforce_pattern(q,phrases,[2,2])
    print("query", [(m.text, round(m.pitch,2)) for ap in q.accent_phrases for m in ap.moras])
    wav, sr = tts.speak(phrases, sid, [2,2], speed=0.85)
    pad=np.zeros(4800,np.float32); a16=np.concatenate([pad,align.to16k(wav,sr),pad])
    tr=pitch.track(a16,16000,conf_thresh=0.5)
    sp=align.force_align(align.logprobs(a16),[m.hira for p in phrases for m in p.moras])
    for m,s in zip([m.hira for p in phrases for m in p.moras], sp):
        f=(tr.t>=s.start)&(tr.t<s.end)
        print(f" {m} {s.start:.2f}-{s.end:.2f}", " ".join(f"{h:.0f}" if c>=.5 else "-" for h,c in zip(tr.hz[f],tr.conf[f])))
