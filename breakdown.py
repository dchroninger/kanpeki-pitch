"""Where do false alarms / abstains come from? Break down by phrase properties. Usage: breakdown.py MODEL v1|v2"""
import pickle, random, sys
from collections import defaultdict
import numpy as np
from jpa import accent
from jpa.grade import equivalent
from learn import posterior
import train_combined as tc, learn_v2
model = pickle.load(open(sys.argv[1], "rb"))
if sys.argv[2] == "v2":
    tc.phrase_rows = learn_v2.phrase_rows
sents = sorted({u["sent"] for u in tc.V}); random.Random(3).shuffle(sents); TEST_SENT = set(sents[:10])
in_jtrain = {id(u) for u in tc.J_train}
data = [u for u in tc.J if id(u) not in in_jtrain] + [u for u in tc.V if u["spk"] in tc.test_spk and u["sent"] in TEST_SENT]
FE = tc.Feats(data)
T = defaultdict(lambda: [0, 0, 0])  # n, FA, abstain (strict)
for (u, pi, info), p in zip(FE.items, FE.probs(model)):
    if (u.get("sent"), pi) in tc.bad and "spk" in u:
        continue
    n, gold, final = info["n"], info["acc"], info["final"]
    moras = accent.split_moras(u["gold"][pi][0]); dev = u["gold"][pi][2]
    post = posterior(p, n)
    pe = sum(v for k, v in post.items() if equivalent(k, gold, n, final))
    ok = True if pe >= 0.9 else False if pe <= 0.1 else None
    tags = ["ALL", f"acc={'0 heiban' if gold == 0 else '1 atama' if gold == 1 else 'n odaka' if gold == n else '2+ naka'}",
            f"len={'1-2' if n <= 2 else '3-4' if n <= 4 else '5-7' if n <= 7 else '8+'}",
            "utt-initial" if pi == 0 else "utt-final" if final else "utt-medial"]
    if gold == 1:
        tags += ["atama & 1st devoiced" if dev[0] else "atama & 1st voiced"]
        if n > 1 and moras[1].kana in ("ー", "ン", "ッ"): tags.append("atama & 2nd special (ー/ン/ッ)")
        tags.append("atama & utt-initial" if pi == 0 else "atama & not initial")
    if any(dev): tags.append("has devoiced mora")
    for g in tags:
        T[g][0] += 1; T[g][1] += ok is False; T[g][2] += ok is None
print(f"{'group':34s} {'n':>6s} {'FA%':>6s} {'abst%':>6s}")
for g, (n, fa, ab) in sorted(T.items(), key=lambda x: x[0]):
    print(f"{g:34s} {n:6d} {100*fa/n:6.1f} {100*ab/n:6.1f}")
