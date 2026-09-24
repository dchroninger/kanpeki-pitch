"""Feature set v2: v1 + contour-shape features aimed at the atamadaka blind spot.

Adds per mora j:
  - peak position: where the F0 max falls within [start_j, end_{j+1}] (0..1), catches peak delay
  - steepest fall (st/s) and its position within [start_j, end_{j+1}]
  - utterance-relative mid pitch (v1 is phrase-relative only)
  - interpolated pitch for unvoiced/devoiced morae + 'was interpolated' flag
Adds per phrase: utterance-initial flag, pitch step from previous phrase's last voiced frame."""
import numpy as np
from jpa import accent
from learn import contour, mora_feats, NF, NB, SPECIAL


def _interp(t, st):
    ok = ~np.isnan(st)
    if ok.sum() < 2:
        return st.copy()
    return np.interp(t, t[ok], st[ok])


def _smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode="same") if len(y) >= k else y


def shape_feats(t, st, sti, s0, e1):
    """Peak position, steepest-fall slope and position over window [s0, e1)."""
    m = (t >= s0) & (t < e1)
    if m.sum() < 4:
        return [np.nan, np.nan, np.nan]
    tt, yy = t[m], _smooth(sti[m])
    span = max(e1 - s0, 1e-3)
    peak = (tt[np.argmax(yy)] - s0) / span
    d = np.diff(yy) / np.maximum(np.diff(tt), 1e-3)
    i = int(np.argmin(d))
    return [peak, d[i], (tt[i] - s0) / span]


def phrase_rows(u):
    t, st = contour(u)
    sti = _interp(t, st)
    spans = u["spans"]
    allf = [mora_feats(t, st, s, e) for s, e in spans]
    # utterance-relative mid pitch, with interpolation for unvoiced morae
    umid, uint = [], []
    for s, e in spans:
        m = (t >= s) & (t < e)
        v = st[m][~np.isnan(st[m])]
        if len(v):
            umid.append(float(np.median(v))); uint.append(0.0)
        else:
            mm = (t >= s) & (t < e)
            umid.append(float(np.median(sti[mm])) if mm.any() else np.nan); uint.append(1.0)
    i = 0
    prev_last = np.nan
    for pi, (reading, acc, dev) in enumerate(u["gold"]):
        moras = accent.split_moras(reading)
        n = len(moras)
        pf = np.array(allf[i:i + n], float)
        ref = np.nanmedian(pf[:, 1]) if not np.all(np.isnan(pf[:, 1])) else 0.0
        pf[:, :3] -= ref; pf[:, 4] -= ref
        first_v = next((x for x in umid[i:i + n] if x == x), np.nan)
        step_in = first_v - prev_last
        rows = []
        for j in range(n):
            f = [j, n, j / n, n - j, float(moras[j].kana in SPECIAL), float(dev[j]),
                 float(j + 1 < n and moras[j + 1].kana in SPECIAL), float(pi == len(u["gold"]) - 1)]
            for d in range(-NB, NB + 1):
                jj = j + d
                f += list(pf[jj]) if 0 <= jj < n else [np.nan] * NF
            for d in (1, 2):
                f += [pf[j + d, 1] - pf[j, 1], pf[j + d, 0] - pf[j, 2]] if j + d < n else [np.nan, np.nan]
            # --- v2
            s0 = spans[i + j][0]
            e1 = spans[i + j + 1][1] if j + 1 < n else spans[i + j][1]
            f += shape_feats(t, st, sti, s0, e1)
            f += [umid[i + j], uint[i + j],
                  umid[i + j + 1] - umid[i + j] if j + 1 < n else np.nan,
                  float(pi == 0), step_in, float(j == 0)]
            rows.append(f)
        yield dict(n=n, acc=min(acc, n), final=pi == len(u["gold"]) - 1), np.array(rows, float)
        last_v = [x for x in umid[i:i + n] if x == x]
        prev_last = last_v[-1] if last_v else prev_last
        i += n
