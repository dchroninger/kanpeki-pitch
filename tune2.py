import numpy as np
from tune import TRAIN, TEST, mora_values, evaluate, fmt
from jpa import accent

SPECIAL = {"ン", "ー", "ッ"}


def specials(u):
    out = []
    for reading, acc, dev in u["gold"]:
        out += [m.kana in SPECIAL for m in accent.split_moras(reading)]
    return out


def make_scorer(decl=-0.4, min_step=1.0, skip_special=True, skip_post=False):
    def score(y, n, spec):
        y = np.asarray(y, float)
        res = {}
        for k in range(n + 1):
            h = np.array([c == "H" for c in accent.pattern(n, k)], float)
            use = ~np.isnan(y)
            # the drop is realised after the whole syllable: special mora after nucleus is "don't care"
            if skip_special and 1 <= k < n and spec[k]:
                use[k] = False
            # peak delay: the post-nucleus mora is in transit -> don't care
            if skip_post and 1 <= k < n:
                use[k] = False
            if use.sum() < 2:
                continue
            yd = (y - decl * np.arange(n))[use]; hh = h[use]
            if hh.min() == hh.max():
                res[k] = float(np.var(yd) * len(yd)) + 1.0; continue
            b = max(np.polyfit(hh, yd, 1)[0], min_step)
            a = np.mean(yd - b * hh)
            # residual per used point, rescaled to full length so templates with don't-cares aren't favoured
            res[k] = float(np.sum((a + b * hh - yd) ** 2)) * n / len(yd)
        return res or None
    return score


def run(data, a, b, lag, **kw):
    sc = make_scorer(**kw)
    cache = {}
    def measure(u):
        cache["spec"] = specials(u); return mora_values(u, a, b, lag)
    # evaluate() calls scorer(y, n) per phrase in order; thread specials through a closure
    def scorer(y, n):
        i = scorer.i; scorer.i += n
        return sc(y, n, cache["spec"][i:i + n])
    def measure_reset(u):
        scorer.i = 0; return measure(u)
    return evaluate(data, measure_reset, scorer)


if __name__ == "__main__":
    for a, b, lag in [(0.25, 0.75, 0.0), (0.25, 0.75, 0.03), (0.5, 1.0, 0.0), (0.4, 1.0, 0.0), (0.6, 1.0, 0.0), (0.5, 1.0, 0.03)]:
        for kw in [dict(skip_special=False), dict(skip_special=True), dict(skip_special=True, skip_post=True)]:
            print(f"win=({a},{b}) lag={lag:+.2f} {kw}: {fmt(run(TRAIN, a, b, lag, **kw))}")
