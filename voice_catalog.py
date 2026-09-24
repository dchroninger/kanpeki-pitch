"""Build voice_catalog.json: one neutral style per VOICEVOX character, with measured median F0 -> male/female."""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from jpa import accent, tts, pitch, align

cat = json.load(open("voice_catalog.json"))  # [vvm, character, style, id]
PREF = ["ノーマル", "ふつう", "人間ver.", "つぼみ"]
chars = {}
for vvm, c, s, i in cat:
    chars.setdefault(c, []).append((vvm, s, i))
phrases, _ = accent.analyze("今日はいい天気ですね。明日は雨が降るそうです。")
out = []
for c, styles in chars.items():
    vvm, s, i = sorted(styles, key=lambda x: PREF.index(x[1]) if x[1] in PREF else 99)[0]
    tts.load(vvm)
    wav, sr = tts.speak(phrases, i, enforce="raw")
    tr = pitch.track(align.to16k(wav, sr), 16000, conf_thresh=0.6)
    f0 = float(np.median(tr.hz[tr.conf >= 0.6]))
    out.append(dict(id=i, vvm=vvm, name=c, style=s, f0=round(f0), gender="M" if f0 < 165 else "F"))
    print(f"{c:14s} {s:8s} {f0:5.0f} Hz  {out[-1]['gender']}", flush=True)
json.dump(sorted(out, key=lambda v: v["f0"]), open("voice_catalog_neutral.json", "w"), ensure_ascii=False, indent=1)
