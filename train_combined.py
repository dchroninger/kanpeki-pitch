"""Retrain nucleus model on JSUT (hand labels) + JVS (OpenJTalk labels minus consensus label errors).

Held out for testing: last 1000 JSUT utterances, and 30 JVS speakers (15 M / 15 F) never seen in training.
Compares: A = JSUT-300 (current), B = JSUT-4000, C = JSUT-4000 + JVS-70 speakers."""
import pickle, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from jpa.grade import equivalent
from learn import phrase_rows, posterior

J = pickle.load(open("jsut_cache_full.pkl", "rb"))
V = pickle.load(open("jvs_cache.pkl", "rb"))
old = pickle.load(open("nucleus_model.pkl", "rb"))


class Feats:
    """All phrases of a set of utterances, features extracted once, predicted in one batch."""
    def __init__(self, utts):
        self.items, blocks = [], []
        for u in utts:
            for pi, (info, F) in enumerate(phrase_rows(u)):
                self.items.append((u, pi, info)); blocks.append(F)
        self.X = np.vstack(blocks)
        self.off = np.cumsum([0] + [len(b) for b in blocks])

    def probs(self, model):
        p = model.predict_proba(self.X)[:, 1]
        return [p[self.off[i]:self.off[i + 1]] for i in range(len(self.items))]
gender = {l.split()[0]: l.split()[1] for l in open(Path("vendor/jvs/jvs_ver1/gender_f0range.txt")) if l.startswith("jvs")}

# --- JVS label errors by consensus (old model): phrase failed by >50% of speakers
FV = Feats(V)
fl = defaultdict(list)
for (u, pi, info), p in zip(FV.items, FV.probs(old)):
    post = posterior(p, info["n"])
    pe = sum(v for k, v in post.items() if equivalent(k, info["acc"], info["n"], info["final"]))
    fl[(u["sent"], pi)].append(pe <= 0.2)
bad = {k for k, v in fl.items() if np.mean(v) > 0.5}
# manual review (2026-09-24): label is correct, grader was wrong -> keep for training and testing
BLIND_SPOTS = {("VOICEACTRESS100_028", 12), ("VOICEACTRESS100_028", 8), ("VOICEACTRESS100_002", 2),
               ("VOICEACTRESS100_011", 5), ("VOICEACTRESS100_009", 3)}
bad -= BLIND_SPOTS

# --- splits
spk = sorted({u["spk"] for u in V})
rng = random.Random(7)
m = [s for s in spk if gender[s] == "M"]; f = [s for s in spk if gender[s] == "F"]
rng.shuffle(m); rng.shuffle(f)
test_spk = set(m[:15] + f[:15])
V_train = [u for u in V if u["spk"] not in test_spk]
V_test = [u for u in V if u["spk"] in test_spk]
J_train, J_test = J[:4000], J[4000:]


def dataset(fe: Feats, keep_utt=lambda u: True, skip=frozenset()):
    X, y = [], []
    for i, (u, pi, info) in enumerate(fe.items):
        if not keep_utt(u) or (u.get("sent"), pi) in skip:
            continue
        X.append(fe.X[fe.off[i]:fe.off[i + 1]]); y += [int(j + 1 == info["acc"]) for j in range(info["n"])]
    return np.vstack(X), np.array(y)


def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0).fit(X, y)


def evaluate(model, fe: Feats, hi, lo, keep_utt=lambda u: True, skip=frozenset(), by_gender=False, seed=0, probs=None):
    r = random.Random(seed)
    T = defaultdict(lambda: defaultdict(int))
    probs = probs if probs is not None else fe.probs(model)
    for (u, pi, info), p in zip(fe.items, probs):
        if not keep_utt(u) or (u.get("sent"), pi) in skip:
            continue
        g = gender.get(u.get("spk"), "all") if by_gender else "all"
        if True:
            n, gold, final = info["n"], info["acc"], info["final"]
            post = posterior(p, n)
            alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
            for exp, kind in [(gold, "c")] + ([(r.choice(alts), "w")] if alts else []):
                pe = sum(v for k, v in post.items() if equivalent(k, exp, n, final))
                ok = True if pe >= hi else False if pe <= lo else None
                for key in {g, "all"}:
                    t = T[key]; t["n" + kind] += 1
                    t[(kind, ok)] += 1
    out = {}
    for key, t in T.items():
        out[key] = dict(FA=t[("c", False)] / t["nc"], passed=t[("c", True)] / t["nc"],
                        caught=t[("w", False)] / t["nw"], MISS=t[("w", True)] / t["nw"],
                        abstain=(t[("c", None)] + t[("w", None)]) / (t["nc"] + t["nw"]))
    return out


def fmt(d):
    return "  ".join(f"{k} {100*v:5.1f}%" for k, v in d.items())


if __name__ == "__main__":
    import time
    t0 = time.time()
    print(f"JVS: {len(bad)} label-error phrases excluded; test speakers {len(test_spk)} (15M/15F); "
          f"JSUT train {len(J_train)} / test {len(J_test)}", flush=True)
    FJ = Feats(J)
    in_jtrain = {id(u) for u in J_train}
    jtr, jte = (lambda u: id(u) in in_jtrain), (lambda u: id(u) not in in_jtrain)
    vtr, vte = (lambda u: u["spk"] not in test_spk), (lambda u: u["spk"] in test_spk)
    print(f"features ready {time.time()-t0:.0f}s", flush=True)
    models = {"A: JSUT-300 (current)": old}
    X, y = dataset(FJ, jtr); models["B: JSUT-4000"] = fit(X, y)
    X2, y2 = dataset(FV, vtr, bad)
    models["C: JSUT-4000 + JVS-70spk"] = fit(np.vstack([X, X2]), np.concatenate([y, y2]))
    print(f"trained {time.time()-t0:.0f}s", flush=True)
    P = {name: (FJ.probs(m), FV.probs(m)) for name, m in models.items()}
    for hi, lo in [(0.8, 0.2), (0.9, 0.1)]:
        print(f"\n=== thresholds hi={hi} lo={lo}")
        for name, mdl in models.items():
            pj, pv = P[name]
            print(f"{name}")
            print(f"   JSUT held-out     {fmt(evaluate(mdl, FJ, hi, lo, jte, probs=pj)['all'])}")
            ev = evaluate(mdl, FV, hi, lo, vte, bad, by_gender=True, probs=pv)
            for g in ("M", "F"):
                print(f"   JVS unseen {g}      {fmt(ev[g])}")
    pickle.dump(models["C: JSUT-4000 + JVS-70spk"], open("nucleus_model_v2.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
