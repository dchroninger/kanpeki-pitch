"""Install any .aivmx passed on the command line, then catalogue every AivisSpeech style with measured F0.
Writes voice_catalog_aivis.json. Needs the engine running (vendor/aivis/macOS-arm64/run --port 10101)."""
import json, sys, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np
from jpa import tts, pitch, align

for path in sys.argv[1:]:
    import subprocess
    subprocess.run(["curl", "-s", "-X", "POST", "-F", f"file=@{path}", f"{tts.AIVIS}/aivm_models/install"], check=True)
speakers = json.load(urllib.request.urlopen(f"{tts.AIVIS}/speakers"))
groups = tts.clauses("今日はいい天気ですね")
out = []
for sp in speakers:
    for st in sp["styles"]:
        wav, sr = tts.aivis_speak_groups(groups, st["id"])
        tr = pitch.track(align.to16k(wav, sr), 16000, conf_thresh=0.6)
        f0 = float(np.median(tr.hz[tr.conf >= 0.6])) if (tr.conf >= 0.6).any() else 0.0
        out.append(dict(id=st["id"], engine="aivis", name=sp["name"], style=st["name"], f0=round(f0),
                        gender="M" if f0 < 165 else "F"))
        print(f"{sp['name']:8s} {st['name']:10s} {f0:5.0f} Hz {out[-1]['gender']}", flush=True)
json.dump(out, open("voice_catalog_aivis.json", "w"), ensure_ascii=False, indent=1)
