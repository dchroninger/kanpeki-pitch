"""Cross-speaker test: JSUT-trained nucleus model on 100 JVS speakers (49 M / 51 F).

Labels are OpenJTalk predictions. A phrase that most speakers 'fail' is a label error, not a
grader error, so phrases flagged for > LABEL_ERR of speakers are excluded."""
import pickle, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from jpa.grade import equivalent
from learn import phrase_rows, posterior

LABEL_ERR = 0.5
HI, LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8, float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
model = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "nucleus_model.pkl", "rb"))
C = pickle.load(open(sys.argv[4] if len(sys.argv) > 4 else "jvs_cache.pkl", "rb"))
gender = {l.split()[0]: l.split()[1] for l in open(Path("vendor/jvs/jvs_ver1/gender_f0range.txt")) if l.startswith("jvs")}

rng = random.Random(0)
rows = []
for u in C:
    for pi, (info, F) in enumerate(phrase_rows(u)):
        n, gold, final = info["n"], info["acc"], info["final"]
        post = posterior(model.predict_proba(F)[:, 1], n)
        verdict = lambda exp: (lambda pe: True if pe >= HI else False if pe <= LO else None)(
            sum(v for k, v in post.items() if equivalent(k, exp, n, final)))
        alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
        rows.append(dict(key=(u["sent"], pi), spk=u["spk"], g=gender[u["spk"]], n=n,
                         c=verdict(gold), w=verdict(rng.choice(alts)) if alts else "na"))

by_key = defaultdict(list)
for r in rows:
    by_key[r["key"]].append(r["c"] is False)
bad_labels = {k for k, v in by_key.items() if np.mean(v) > LABEL_ERR}
kept = [r for r in rows if r["key"] not in bad_labels]
print(f"{len(C)} utterances, {len({r['spk'] for r in rows})} speakers, {len(by_key)} distinct phrases; "
      f"{len(bad_labels)} dropped as likely label errors (>{int(LABEL_ERR*100)}% of speakers flagged)")


def report(rs, name):
    c = [r["c"] for r in rs]; w = [r["w"] for r in rs if r["w"] != "na"]
    pct = lambda a, b: f"{100*a/b:5.1f}%" if b else "  n/a"
    print(f"{name:10s} n={len(rs):5d}  FA {pct(sum(x is False for x in c), len(c))}  pass {pct(sum(x is True for x in c), len(c))}"
          f"  | caught {pct(sum(x is False for x in w), len(w))}  MISS {pct(sum(x is True for x in w), len(w))}"
          f"  | abstain {pct(sum(x is None for x in c + w), len(c) + len(w))}")


print(f"thresholds hi={HI} lo={LO}")
report(kept, "all")
report([r for r in kept if r["g"] == "M"], "male")
report([r for r in kept if r["g"] == "F"], "female")
spk_fa = defaultdict(list)
for r in kept:
    spk_fa[r["spk"]].append(r["c"] is False)
worst = sorted(((np.mean(v), s) for s, v in spk_fa.items()), reverse=True)[:5]
print("worst speakers (FA):", ", ".join(f"{s}({gender[s]}) {100*f:.0f}%" for f, s in worst))
