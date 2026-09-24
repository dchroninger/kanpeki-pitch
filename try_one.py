import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
from jpa import accent, grade, tts
from transformers.utils import logging as hl; hl.set_verbosity_error()

text = sys.argv[1] if len(sys.argv) > 1 else "箸が好きです"
phrases, _ = accent.analyze(text)
print("expected:", [(p.surface, p.reading, p.acc, p.pattern) for p in phrases])
for vvm, sid in [("0", 2), ("4", 11)]:
    tts.load(vvm)
    base = [p.acc for p in phrases]
    wrong = base.copy(); wrong[0] = (base[0] + 1) % (len(phrases[0].moras) + 1)
    for label, accs in [("correct", base), ("wrong", wrong)]:
        wav, sr = tts.speak(phrases, sid, accs, speed=0.85)
        sf.write(f"out_{sid}_{label}.wav", wav, sr)
        r = grade.grade(wav, sr, phrases)
        print(f"\n[voice {sid} {label} {tts.to_kana(phrases, accs)}] heard: {r.heard_kana}")
        for pr in r.phrases:
            print(f"  {pr.phrase.surface:8s} exp {pr.phrase.pattern:6s} heard {pr.heard_pattern:6s} ok={pr.expected_ok} conf={pr.confidence:.2f} st={np.round(pr.mora_st,1).tolist()}")
