"""Retrain on JSUT-4000 + JVS train speakers (all sentences cached so far, minus fixed test sentences)
and evaluate on a FIXED test set: 30 unseen speakers x 10 unseen sentences (+ JSUT held-out 1000).

Appends one row per threshold setting to RESEARCH.md.  Usage: batch_eval.py LABEL [batch pickles...]"""
import pickle, random, sys
from collections import defaultdict
import numpy as np
from jpa.grade import equivalent
from learn import posterior
from train_combined import J, V, FV, Feats, bad, test_spk, J_train, dataset, fit, gender, old
import learn

label, batch_files = sys.argv[1], sys.argv[2:]
sents = sorted({u["sent"] for u in V}); random.Random(3).shuffle(sents)
TEST_SENT = set(sents[:10])  # fixed forever, from the original 30

new = [u for f in batch_files for u in pickle.load(open(f, "rb"))]
FN = Feats(new) if new else None
# label-error filter for new sentences: >50% of TRAIN speakers flagged by model A
bad_new = set()
if FN:
    fl = defaultdict(list)
    for (u, pi, info), p in zip(FN.items, FN.probs(old)):
        if u["spk"] in test_spk:
            continue
        post = posterior(p, info["n"])
        fl[(u["sent"], pi)].append(sum(v for k, v in post.items() if equivalent(k, info["acc"], info["n"], info["final"])) <= 0.2)
    bad_new = {k for k, v in fl.items() if np.mean(v) > 0.5}

FJ = Feats(J)
in_jtrain = {id(u) for u in J_train}
X, y = dataset(FJ, lambda u: id(u) in in_jtrain)
Xs, ys = [X], [y]
X2, y2 = dataset(FV, lambda u: u["spk"] not in test_spk and u["sent"] not in TEST_SENT, bad); Xs.append(X2); ys.append(y2)
if FN:
    X3, y3 = dataset(FN, lambda u: u["spk"] not in test_spk, bad_new); Xs.append(X3); ys.append(y3)
model = fit(np.vstack(Xs), np.concatenate(ys))
n_train_sent = len({u["sent"] for u in V if u["sent"] not in TEST_SENT} | {u["sent"] for u in new})


def run(fe, keep, skip, hi, lo, probs):
    r = random.Random(0)
    T = defaultdict(lambda: defaultdict(int))
    for (u, pi, info), p in zip(fe.items, probs):
        if not keep(u) or (u.get("sent"), pi) in skip:
            continue
        n, gold, final = info["n"], info["acc"], info["final"]
        post = posterior(p, n)
        groups = ["all", gender.get(u.get("spk"), "all")] + (["atama"] if gold == 1 else [])
        alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
        for exp, kind in [(gold, "c")] + ([(r.choice(alts), "w")] if alts else []):
            pe = sum(v for k, v in post.items() if equivalent(k, exp, n, final))
            ok = True if pe >= hi else False if pe <= lo else None
            for g in set(groups):
                T[g]["n" + kind] += 1; T[g][(kind, ok)] += 1
    pc = lambda g, a, b: 100 * T[g][a] / max(1, T[g][b])
    ab = lambda g: 100 * (T[g][("c", None)] + T[g][("w", None)]) / max(1, T[g]["nc"] + T[g]["nw"])
    return {g: dict(FA=pc(g, ("c", False), "nc"), MISS=pc(g, ("w", True), "nw"),
                    caught=pc(g, ("w", False), "nw"), abst=ab(g)) for g in T}


pv, pj = FV.probs(model), FJ.probs(model)
te = lambda u: u["spk"] in test_spk and u["sent"] in TEST_SENT
rows = []
for hi, lo, name in [(0.8, 0.2, "balanced"), (0.9, 0.1, "strict")]:
    v = run(FV, te, bad, hi, lo, pv)
    j = run(FJ, lambda u: id(u) not in in_jtrain, set(), hi, lo, pj)["all"]
    rows.append(f"| {label} | {n_train_sent} | {name} | {v['M']['FA']:.1f} / {v['F']['FA']:.1f} | "
                f"{v['all']['MISS']:.1f} | {v['all']['caught']:.1f} | {v['all']['abst']:.1f} | "
                f"{v['atama']['FA']:.1f} / {v['atama']['abst']:.1f} | {j['FA']:.1f} / {j['MISS']:.1f} / {j['abst']:.1f} |")
with open("RESEARCH.md", "a") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(rows))
pickle.dump(model, open(f"model_{label.replace(' ', '_')}.pkl", "wb"))
