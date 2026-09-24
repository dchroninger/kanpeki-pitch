"""Tune the per-mora pitch measurement + scoring on cached native speech.

Train/test split by utterance. For every phrase we grade twice:
  - against its true (labelled) accent  -> pass / FALSE ALARM / abstain
  - against a random non-equivalent accent -> caught / MISSED / abstain
"""
import pickle, random, sys
import numpy as np
from jpa import accent
from jpa.grade import equivalent

C = pickle.load(open("jsut_cache.pkl", "rb"))
split = len(C) // 2
TRAIN, TEST = C[:split], C[split:]


def mora_values(u, a=0.25, b=0.75, lag=0.0, conf=0.5):
    """Per-mora pitch (st re: utterance median) from window [a,b] of each span, shifted by lag s."""
    t, hz, cf = u["t"], u["hz"], u["conf"]
    v = (cf >= conf) & (hz > 0)
    st = np.full_like(hz, np.nan)
    st[v] = 12 * np.log2(hz[v] / np.median(hz[v]))
    out = []
    for s, e in u["spans"]:
        lo, hi = s + a * (e - s) + lag, s + b * (e - s) + lag
        m = (t >= lo) & (t < hi) & ~np.isnan(st)
        out.append(float(np.median(st[m])) if m.any() else np.nan)
    return out


def score_level(y, n, decl=-0.4, min_step=1.0):
    """Original: 2-level template + fixed declination. Returns {k: residual}."""
    y = np.asarray(y, float); ok = ~np.isnan(y)
    if ok.sum() < 2:
        return None
    yd = y - decl * np.arange(n)
    res = {}
    for k in range(n + 1):
        h = np.array([c == "H" for c in accent.pattern(n, k)], float)[ok]
        if h.min() == h.max():
            res[k] = float(np.var(yd[ok]) * ok.sum()) + 1.0; continue
        b = max(np.polyfit(h, yd[ok], 1)[0], min_step)
        a = np.mean(yd[ok] - b * h)
        res[k] = float(np.sum((a + b * h - yd[ok]) ** 2))
    return res


def phrases_of(u):
    i = 0
    for pi, (reading, acc, dev) in enumerate(u["gold"]):
        n = len(dev)
        yield pi == len(u["gold"]) - 1, n, acc, i
        i += n


def evaluate(data, measure, scorer, sigma2=2.0, seed=0, verbose=False):
    rng = random.Random(seed)
    tally = dict(pass_=0, fa=0, abst_c=0, caught=0, miss=0, abst_w=0, nc=0, nw=0)
    for u in data:
        y_all = measure(u)
        for final, n, gold, i in phrases_of(u):
            y = y_all[i:i + n]
            sc = scorer(y, n)
            alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
            for exp, kind in [(gold, "c")] + ([(rng.choice(alts), "w")] if alts else []):
                tally["nc" if kind == "c" else "nw"] += 1
                if sc is None:
                    tally["abst_c" if kind == "c" else "abst_w"] += 1; continue
                best = min(sc, key=sc.get)
                same = [k for k in sc if equivalent(k, best, n, final)]
                if exp in same:
                    runner = min([r for k, r in sc.items() if k not in same], default=sc[best] + 20)
                    ok = True if 1 - np.exp(-(runner - sc[best]) / sigma2) >= 0.5 else None
                elif exp not in sc:
                    ok = None
                else:
                    ok = False if 1 - np.exp(-(sc[exp] - sc[best]) / sigma2) >= 0.5 else None
                if kind == "c":
                    tally["pass_" if ok is True else "fa" if ok is False else "abst_c"] += 1
                else:
                    tally["caught" if ok is False else "miss" if ok is True else "abst_w"] += 1
    T = tally
    return dict(false_alarm=T["fa"] / T["nc"], passed=T["pass_"] / T["nc"],
                caught=T["caught"] / T["nw"], missed=T["miss"] / T["nw"],
                abstain=(T["abst_c"] + T["abst_w"]) / (T["nc"] + T["nw"]))


def fmt(r):
    return f"FA {100*r['false_alarm']:5.1f}%  pass {100*r['passed']:5.1f}%  | caught {100*r['caught']:5.1f}%  MISS {100*r['missed']:5.1f}%  | abstain {100*r['abstain']:5.1f}%"


if __name__ == "__main__":
    print(f"{len(C)} utterances cached; train {len(TRAIN)} / test {len(TEST)}")
    print("baseline (mid-mora, level fit):", fmt(evaluate(TRAIN, mora_values, score_level)))
    best = None
    for a, b in [(0.0, 0.5), (0.0, 0.4), (0.1, 0.6), (0.25, 0.75)]:
        for lag in [-0.06, -0.03, 0.0, 0.03]:
            r = evaluate(TRAIN, lambda u: mora_values(u, a, b, lag), score_level)
            key = r["false_alarm"] + r["missed"] * 2 - 0.3 * (r["passed"] + r["caught"])
            print(f"  win=({a},{b}) lag={lag:+.2f}: {fmt(r)}")
            if best is None or key < best[0]:
                best = (key, a, b, lag)
    print("best:", best)
