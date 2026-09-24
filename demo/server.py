"""Throwaway demo: speak a JSUT sentence, get pitch-accent feedback.  Run: uv run python demo/server.py"""
import io, os, random, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import numpy as np, pickle, soundfile as sf, torch, yaml
import pyopenjtalk as pj
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from transformers.utils import logging as hl; hl.set_verbosity_error()

from jpa import accent, align, pitch, tts
from jpa.grade import equivalent
from learn import posterior
import learn_v3, seq_model as sm
from eval_jsut import parse_label, J

TREE = os.environ.get("TREE", "model_v3_full_p7.pkl")
SEQ = os.environ.get("SEQ", "seq_full_aug.pt")
HI, LO = 0.9, 0.1
align.DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------- sentences: JSUT with hand labels, OpenJTalk reading must match the label's morae
labels = yaml.safe_load(open(J / "jsut-label/e2e_symbol/katakana.yaml"))
texts = dict(l.strip().split(":", 1) for l in open(J / "jsut_ver1.1/basic5000/transcript_utf8.txt"))
hira = lambda s: "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
KANJI = re.compile(r"[一-鿿々〆ヶ]")


def furigana(text):
    """[(surface, ruby-or-None)] with okurigana stripped from the ruby."""
    out = []
    for n in pj.run_frontend(text):
        s, r = n["string"], hira(n["read"])
        if not KANJI.search(s) or not r or r == "、":
            out.append((s, None)); continue
        pre = 0
        while pre < min(len(s), len(r)) and s[pre] == r[pre] and not KANJI.match(s[pre]): pre += 1
        suf = 0
        while suf < min(len(s), len(r)) - pre and s[-1 - suf] == r[-1 - suf] and not KANJI.match(s[-1 - suf]): suf += 1
        if pre: out.append((s[:pre], None))
        out.append((s[pre:len(s) - suf], r[pre:len(r) - suf]))
        if suf: out.append((s[len(s) - suf:], None))
    return out


SENTS = []
for uid in sorted(labels):
    gold = parse_label(labels[uid])
    n = sum(len(p.moras) for p in gold)
    if not 12 <= n <= 30:
        continue
    pred, _ = accent.analyze(texts[uid])
    if [p.reading for p in pred] != [p.reading for p in gold]:
        continue
    SENTS.append(uid)
print(f"{len(SENTS)} demo sentences", flush=True)

# ---------- models
tree = pickle.load(open(TREE, "rb"))
net = None
if Path(SEQ).exists():
    dummy = dict(gold=[("アメガ", 1, [False] * 3)], spans=[(0.0, .1), (.1, .2), (.2, .3)],
                 t=np.linspace(0, .3, 20), hz=np.full(20, 200.0), conf=np.ones(20))
    net = sm.Net(next(sm.utt_phrases(dummy))[1].shape[1]); net.load_state_dict(torch.load(SEQ)); net.eval()
print(f"tree={TREE} seq={SEQ if net else 'none'} device={align.DEVICE}", flush=True)
import json
CATALOG = json.load(open(ROOT / "voice_catalog_neutral.json"))          # one neutral style per character
BY_ID = {v["id"]: v for v in CATALOG}
KEEP = ROOT / "demo" / "voices_keep.json"
DEFAULT_KEEP = [2, 3, 11, 13, 16, 14]
kept = lambda: json.load(open(KEEP)) if KEEP.exists() else DEFAULT_KEEP
# 夏目漱石『吾輩は猫である』冒頭 (1905, public domain, Aozora Bunko)
SAMPLE_TEXT = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
PRIVATE = ROOT / "demo" / "private"   # gitignored: personal texts (e.g. OCR of a book you own)


def texts_available():
    out = [dict(key="soseki", title="吾輩は猫である (夏目漱石)", text=SAMPLE_TEXT)]
    for f in sorted(PRIVATE.glob("*.txt")):
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        # "#tts A=B" lines: pronunciation fixes applied only when speaking (displayed text stays as printed)
        subs = [l[5:].split("=", 1) for l in lines[1:] if l.startswith("#tts ")]
        body = "".join(l for l in lines[1:] if not l.startswith("#tts ")).strip()
        out.append(dict(key=f.stem, title=lines[0], text=body, subs=subs))
    return out


def sentences(text):
    return [accent.analyze(x)[0] for x in re.split(r"[。！？!?\n]", text) if x.strip()]


_sample_cache = {}

app = FastAPI()


def sentence_payload(uid):
    gold = parse_label(labels[uid])
    return dict(id=uid, text=texts[uid], furigana=furigana(texts[uid]),
                phrases=[dict(moras=[hira(m.kana) for m in p.moras], acc=p.acc, pattern=p.pattern) for p in gold])


@app.get("/")
def index():
    return FileResponse(ROOT / "demo" / "index.html")


@app.get("/api/sentence")
def sentence():
    return sentence_payload(random.choice(SENTS))


@app.get("/api/voices")
def voices():
    return [dict(id=i, name=BY_ID[i]["name"], gender=BY_ID[i]["gender"]) for i in kept() if i in BY_ID]


@app.get("/voices")
def voices_page():
    return FileResponse(ROOT / "demo" / "voices.html")


@app.get("/api/catalog")
def catalog():
    k = set(kept())
    return [dict(v, keep=v["id"] in k) for v in CATALOG]


@app.post("/api/voices/keep")
async def set_keep(request: Request):
    ids = [int(i) for i in await request.json()]
    json.dump(ids, open(KEEP, "w"))
    return dict(saved=len(ids))


@app.get("/api/texts")
def texts():
    return [dict(key=t["key"], title=t["title"], text=t["text"]) for t in texts_available()]


@app.post("/api/parts")
async def parts(request: Request):
    """Split a text (by key, or custom) into speakable sentences, pronunciation overrides applied."""
    d = await request.json()
    if d.get("key"):
        t = next(t for t in texts_available() if t["key"] == d["key"])
        body = t["text"]
        for a, b in t.get("subs", []):
            body = body.replace(a, b)
    else:
        body = d.get("text", "")[:5000]
    return [x.strip() + "。" for x in re.split(r"[。！？!?\n]", body) if x.strip()]


_part_cache = {}


@app.get("/api/part")
def part(voice: int, s: str):
    """One sentence of audio: small, so playback can start after ~1 s and stream the rest."""
    k = (voice, s)
    if k not in _part_cache:
        v = BY_ID[voice]; tts.load(v["vvm"])
        wav, sr = tts.speak(accent.analyze(s)[0], voice, speed=0.95, enforce="natural")
        wav = np.concatenate([wav, np.zeros(int(0.3 * sr), np.float32)])
        buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV")
        _part_cache[k] = buf.getvalue()
    return Response(_part_cache[k], media_type="audio/wav")


@app.get("/api/sample")
def sample(voice: int, key: str = "soseki", text: str = ""):
    if text:
        body, ck = text[:600], ("custom", voice, text[:600])
    else:
        t = next(t for t in texts_available() if t["key"] == key)
        body = t["text"]
        for a, b in t.get("subs", []):
            body = body.replace(a, b)
        ck = (key, voice, body)
    if ck not in _sample_cache:
        v = BY_ID[voice]; tts.load(v["vvm"])
        parts, sr = [], 24000
        for sentence in sentences(body):  # one sentence at a time, with a breath between
            wav, sr = tts.speak(sentence, voice, speed=0.95, enforce="natural")
            parts += [wav, np.zeros(int(0.35 * sr), np.float32)]
        buf = io.BytesIO(); sf.write(buf, np.concatenate(parts), sr, format="WAV")
        _sample_cache[ck] = buf.getvalue()
    return Response(_sample_cache[ck], media_type="audio/wav")


@app.get("/api/tts")
def speak(id: str, voice: int = 2, mode: str = "natural"):
    tts.load(BY_ID[voice]["vvm"])
    wav, sr = tts.speak(parse_label(labels[id]), voice, speed=0.9, enforce={"natural": "natural", "enforce": True, "raw": "raw"}[mode])
    buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.post("/api/grade")
async def grade(request: Request, id: str, save: int = 1):
    body = await request.body()
    import time
    if save:  # user recordings are kept (first real learner data); scripted tests pass save=0
        rec_dir = ROOT / "demo" / "recordings"; rec_dir.mkdir(exist_ok=True)
        (rec_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{id}.wav").write_bytes(body)
    audio, sr = sf.read(io.BytesIO(body), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(1)
    gold = parse_label(labels[id])
    pad = np.zeros(4800, np.float32)
    a16 = np.concatenate([pad, align.to16k(audio, sr), pad])
    lp = align.logprobs(a16)
    heard_kana = align.transcribe(lp)
    spans = align.force_align(lp, [m.hira for p in gold for m in p.moras])
    if spans is None:
        return JSONResponse(dict(error="Couldn't line the audio up with the sentence. Try again, a little slower?", heard=heard_kana))
    tr = pitch.track(a16, 16000, conf_thresh=0.0)
    u = dict(gold=[(p.reading, p.acc, [False] * len(p.moras)) for p in gold],
             spans=[(s.start, s.end) for s in spans], t=tr.t, hz=tr.hz, conf=tr.conf)
    tree_posts = [posterior(tree.predict_proba(F)[:, 1], info["n"]) for info, F in learn_v3.phrase_rows(u)]
    if net is not None:
        with torch.no_grad():
            seq_posts = [dict(enumerate(sm.log_post(net, X, inph).exp().numpy())) for _, X, inph in sm.utt_phrases(u)]
        posts = [{k: (a[k] + b[k]) / 2 for k in a} for a, b in zip(tree_posts, seq_posts)]
    else:
        posts = tree_posts
    out, i = [], 0
    for pi, (p, post) in enumerate(zip(gold, posts)):
        n, final = len(p.moras), pi == len(gold) - 1
        pe = sum(v for k, v in post.items() if equivalent(k, p.acc, n, final))
        verdict = "ok" if pe >= HI else "wrong" if pe <= LO else "unsure"
        best = max(post, key=post.get)
        if verdict != "wrong":
            best = p.acc if verdict == "ok" else best
        heard = accent.pattern(n, best)
        exp = p.pattern
        bad = [j for j in range(n) if verdict == "wrong" and heard[j] != exp[j]]
        out.append(dict(verdict=verdict, confidence=round(float(pe), 3), heard_acc=int(best), heard_pattern=heard, wrong_morae=bad))
        i += n
    judged = [o for o in out if o["verdict"] != "unsure"]
    score = round(100 * sum(o["verdict"] == "ok" for o in judged) / len(judged)) if judged else None
    return dict(score=score, phrases=out, heard_kana=heard_kana, n_unsure=sum(o["verdict"] == "unsure" for o in out))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
