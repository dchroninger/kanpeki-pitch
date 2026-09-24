"""List JVS phrases most speakers 'failed', with evidence to tell label errors from grader blind spots.

For each: sentence, phrase, OpenJTalk label, what speakers majority-produced (per grader),
% of speakers flagged, and Kanjium dictionary accents for each word in the phrase."""
import pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyopenjtalk as pj
from jpa import accent
from jpa.grade import equivalent
from learn import phrase_rows, posterior

model = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "nucleus_model.pkl", "rb"))
C = pickle.load(open(sys.argv[2] if len(sys.argv) > 2 else "jvs_cache.pkl", "rb"))
JVS = Path("vendor/jvs/jvs_ver1")
text = dict(l.strip().split(":", 1) for l in open(JVS / "jvs001/parallel100/transcripts_utf8.txt") if ":" in l)

kj = defaultdict(list)
for l in open("vendor/kanjium/accents.txt"):
    w, r, a = l.rstrip("\n").split("\t")
    kj[w].append((r, a))


def phrase_words(sentence):
    """Words per accent phrase (same grouping as accent.analyze)."""
    out = []
    for n in pj.run_frontend(sentence):
        if n["pos"] == "記号" or not n["pron"] or n["pron"] in ("、", "。"):
            continue
        if n["chain_flag"] == 1 and out:
            out[-1].append((n["string"], n["pos"], n["acc"]))
        else:
            out.append([(n["string"], n["pos"], n["acc"])])
    return out


flag, heard, meta = defaultdict(list), defaultdict(Counter), {}
for u in C:
    for pi, (info, F) in enumerate(phrase_rows(u)):
        n, gold, final = info["n"], info["acc"], info["final"]
        post = posterior(model.predict_proba(F)[:, 1], n)
        pe = sum(v for k, v in post.items() if equivalent(k, gold, n, final))
        key = (u["sent"], pi)
        flag[key].append(pe <= 0.2)
        best = max(post, key=post.get)
        heard[key][accent.pattern(n, best)] += 1
        meta[key] = (u["gold"][pi][0], gold, n)

dropped = sorted((k for k in flag if np.mean(flag[k]) > 0.5), key=lambda k: -np.mean(flag[k]))
print(f"{len(dropped)} dropped of {len(flag)} phrases\n")
for k in dropped:
    sent, pi = k
    reading, gold, n = meta[k]
    words = phrase_words(text[sent])
    ws = words[pi] if pi < len(words) else []
    dic = "; ".join(f"{w}:{','.join(a for r, a in kj[w][:3])}" for w, pos, _ in ws if kj.get(w)) or "-"
    top = ", ".join(f"{p}×{c}" for p, c in heard[k].most_common(2))
    print(f"{sent[-3:]} #{pi} {reading:14s} label {accent.pattern(n, gold):10s} speakers→ {top:28s} "
          f"flagged {100*np.mean(flag[k]):3.0f}%  dict: {dic}")
