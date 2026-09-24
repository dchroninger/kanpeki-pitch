"""Eval on real native speech (JSUT basic5000, female Tokyo speaker, hand-labelled accents).

1. Grader vs labelled accent: every utterance is 'correct' by definition, so any `ok=False`
   is a false alarm, and heard_acc vs label measures raw detection accuracy.
2. OpenJTalk accent prediction vs label: how trustworthy the answer key is for free speech."""
import json, random, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import accent, grade

J = Path(__file__).parent / "vendor" / "jsut"
TEXT = J / "jsut_ver1.1" / "basic5000" / "transcript_utf8.txt"


def parse_label(s: str) -> list[accent.Phrase]:
    s = s.strip().strip("^$?")
    phrases = []
    for chunk in re.split(r"[#_]", s):
        chunk = chunk.replace("[", "")
        if not chunk:
            continue
        acc = 0
        if "]" in chunk:
            acc = len(accent.split_moras(chunk[:chunk.index("]")]))
        moras = accent.split_moras(chunk.replace("]", ""))
        phrases.append(accent.Phrase(chunk.replace("]", ""), moras, acc))
    return phrases


def main(n=150):
    import yaml
    labels = yaml.safe_load(open(J / "jsut-label" / "e2e_symbol" / "katakana.yaml"))
    texts = dict(l.strip().split(":", 1) for l in open(TEXT))
    ids = sorted(labels)
    random.Random(0).shuffle(ids)
    rows, ojt = [], []
    for uid in ids[:n]:
        gold = parse_label(labels[uid])
        # --- OpenJTalk prediction vs gold (only when phrase segmentation agrees)
        pred, _ = accent.analyze(texts[uid])
        if [p.reading for p in pred] == [p.reading for p in gold]:
            for p, g in zip(pred, gold):
                ojt.append(dict(uid=uid, r=g.reading, pred=p.acc, gold=g.acc,
                                same=p.acc == g.acc or {p.acc, g.acc} == {0, len(g.moras)}))
        # --- grader on native audio, expected = gold
        wav, sr = sf.read(J / "jsut_ver1.1" / "basic5000" / "wav" / f"{uid}.wav", dtype="float32")
        r = grade.grade(wav, sr, gold)
        for pr in r.phrases:
            n_ = len(pr.phrase.moras)
            rows.append(dict(uid=uid, r=pr.phrase.reading, gold=pr.phrase.acc, heard=pr.heard_acc,
                             match=pr.heard_acc is not None and (pr.heard_acc == pr.phrase.acc or {pr.heard_acc, pr.phrase.acc} <= {0, n_}),
                             ok=pr.expected_ok, conf=round(pr.confidence, 2), n=n_,
                             st=[None if x != x else round(x, 1) for x in pr.mora_st]))
        print(uid, file=sys.stderr, flush=True)
    json.dump(dict(rows=rows, ojt=ojt), open("eval_jsut.json", "w"), ensure_ascii=False, indent=0)
    report(rows, ojt)


def report(rows, ojt):
    N = len(rows)
    def pct(a, b): return f"{100 * a / b:5.1f}%" if b else "n/a"
    print(f"native phrases: {N}")
    print(f"  passed       {pct(sum(r['ok'] is True for r in rows), N)}")
    print(f"  FALSE ALARM  {pct(sum(r['ok'] is False for r in rows), N)}")
    print(f"  abstain      {pct(sum(r['ok'] is None for r in rows), N)}")
    print(f"  raw best-fit == label: {pct(sum(r['match'] for r in rows), N)}")
    for lo, hi in [(1, 3), (4, 6), (7, 99)]:
        rs = [r for r in rows if lo <= r["n"] <= hi]
        print(f"  {lo}-{hi} morae n={len(rs):4d}  false-alarm {pct(sum(r['ok'] is False for r in rs), len(rs))}  abstain {pct(sum(r['ok'] is None for r in rs), len(rs))}")
    print(f"\nOpenJTalk accent prediction == native label: {pct(sum(o['same'] for o in ojt), len(ojt))} of {len(ojt)} phrases")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        d = json.load(open("eval_jsut.json")); report(d["rows"], d["ojt"])
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
