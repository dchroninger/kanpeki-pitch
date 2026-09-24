"""Feature set v3: v2 + the raw contour shape — 5 interpolated pitch samples per mora for morae j-1..j+2
(phrase-relative), plus voicing fraction at each sample. Lets the model see *where* in a mora the fall is."""
import numpy as np
from learn import contour
import learn_v2

K = 5  # samples per mora


def phrase_rows(u):
    t, st = contour(u)
    sti = learn_v2._interp(t, st)
    spans = u["spans"]
    i = 0
    for info, F in learn_v2.phrase_rows(u):
        n = info["n"]
        ph = spans[i:i + n]
        samp = []
        for s, e in ph:
            q = s + (np.arange(K) + 0.5) / K * (e - s)
            y = np.interp(q, t, sti) if len(t) else np.full(K, np.nan)
            vo = np.interp(q, t, (~np.isnan(st)).astype(float)) if len(t) else np.zeros(K)
            samp.append((y, vo))
        ref = np.nanmedian(np.concatenate([y for y, _ in samp])) if samp else 0.0
        extra = []
        for j in range(n):
            f = []
            for d in (-1, 0, 1, 2):
                jj = j + d
                if 0 <= jj < n:
                    f += list(samp[jj][0] - ref) + list(samp[jj][1])
                else:
                    f += [np.nan] * (2 * K)
            extra.append(f)
        yield info, np.hstack([F, np.array(extra, float)])
        i += n
