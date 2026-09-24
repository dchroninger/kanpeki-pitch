"""Learned nucleus detector: per-mora 'is this the accent nucleus?' classifier on pitch features.

Phrase posterior over accent types: P(k) ∝ p_k Π_{j≠k}(1-p_j), P(heiban) ∝ Π_j(1-p_j).
Verdict: pass if P(expected ~ equivalents) >= HI, flag if <= LO, else abstain."""
import pickle, random, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from jpa import accent
from jpa.grade import equivalent
from tune import TRAIN, TEST

SPECIAL = {"ン", "ー", "ッ"}
NB = 2  # context morae each side


def contour(u):
    t, hz, cf = u["t"], u["hz"], u["conf"]
    v = (cf >= 0.5) & (hz > 0)
    st = np.full_like(hz, np.nan)
    st[v] = 12 * np.log2(hz[v] / np.median(hz[v]))
    return t, st


def mora_feats(t, st, s, e):
    """start / mid / end pitch, slope, voiced fraction, duration."""
    m = (t >= s) & (t < e)
    y = st[m]; ok = ~np.isnan(y)
    if ok.sum() == 0:
        return [np.nan] * 5 + [0.0, e - s]
    yy, tt = y[ok], t[m][ok]
    third = max(1, len(yy) // 3)
    slope = np.polyfit(tt, yy, 1)[0] if len(yy) >= 3 else 0.0
    return [yy[:third].mean(), np.median(yy), yy[-third:].mean(), slope, yy.max(), ok.mean(), e - s]


NF = 7


def phrase_rows(u):
    """Yields (phrase info, feature matrix [n, F]) for each phrase in utterance."""
    t, st = contour(u)
    spans = u["spans"]
    allf = [mora_feats(t, st, s, e) for s, e in spans]
    i = 0
    for pi, (reading, acc, dev) in enumerate(u["gold"]):
        moras = accent.split_moras(reading)
        n = len(moras)
        pf = np.array(allf[i:i + n], float)
        # normalise pitch features to phrase median (removes phrase-level reset / declination)
        ref = np.nanmedian(pf[:, 1]) if not np.all(np.isnan(pf[:, 1])) else 0.0
        pf[:, :3] -= ref; pf[:, 4] -= ref
        rows = []
        for j in range(n):
            f = [j, n, j / n, n - j, float(moras[j].kana in SPECIAL), float(dev[j]),
                 float(j + 1 < n and moras[j + 1].kana in SPECIAL), float(pi == len(u["gold"]) - 1)]
            for d in range(-NB, NB + 1):
                jj = j + d
                f += list(pf[jj]) if 0 <= jj < n else [np.nan] * NF
            # deltas between this mora and the next ones (the fall)
            for d in (1, 2):
                f += [pf[j + d, 1] - pf[j, 1], pf[j + d, 0] - pf[j, 2]] if j + d < n else [np.nan, np.nan]
            rows.append(f)
        yield dict(n=n, acc=min(acc, n), final=pi == len(u["gold"]) - 1), np.array(rows, float)
        i += n


def dataset(data):
    X, y = [], []
    for u in data:
        for info, F in phrase_rows(u):
            X.append(F); y += [int(j + 1 == info["acc"]) for j in range(info["n"])]
    return np.vstack(X), np.array(y)


def posterior(p, n):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    logq = np.log1p(-p).sum()
    post = {0: logq}
    for k in range(1, n + 1):
        post[k] = logq - np.log1p(-p[k - 1]) + np.log(p[k - 1])
    m = max(post.values())
    z = {k: np.exp(v - m) for k, v in post.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def evaluate(model, data, hi=0.8, lo=0.2, seed=0):
    rng = random.Random(seed)
    T = dict(pass_=0, fa=0, ac=0, caught=0, miss=0, aw=0, nc=0, nw=0, top1=0)
    for u in data:
        for info, F in phrase_rows(u):
            n, gold, final = info["n"], info["acc"], info["final"]
            post = posterior(model.predict_proba(F)[:, 1], n)
            best = max(post, key=post.get)
            T["top1"] += equivalent(best, gold, n, final)
            alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
            for exp, kind in [(gold, "c")] + ([(rng.choice(alts), "w")] if alts else []):
                pe = sum(v for k, v in post.items() if equivalent(k, exp, n, final))
                ok = True if pe >= hi else False if pe <= lo else None
                T["nc" if kind == "c" else "nw"] += 1
                if kind == "c":
                    T["pass_" if ok is True else "fa" if ok is False else "ac"] += 1
                else:
                    T["caught" if ok is False else "miss" if ok is True else "aw"] += 1
    return T


def fmt(T):
    return (f"top-1 {100*T['top1']/T['nc']:5.1f}% | FA {100*T['fa']/T['nc']:5.1f}%  pass {100*T['pass_']/T['nc']:5.1f}%"
            f" | caught {100*T['caught']/T['nw']:5.1f}%  MISS {100*T['miss']/T['nw']:5.1f}%"
            f" | abstain {100*(T['ac']+T['aw'])/(T['nc']+T['nw']):5.1f}%")


if __name__ == "__main__":
    X, y = dataset(TRAIN)
    print("train morae", X.shape, "nucleus rate", y.mean().round(3))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0).fit(X, y)
    for hi, lo in [(0.8, 0.2), (0.9, 0.1), (0.7, 0.3)]:
        print(f"TEST hi={hi} lo={lo}: {fmt(evaluate(model, TEST, hi, lo))}")
    pickle.dump(model, open("nucleus_model.pkl", "wb"))
