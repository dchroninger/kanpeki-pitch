"""Per-mora pitch -> best-fitting accent type per phrase -> verdict vs expected."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import accent, align, pitch

MIN_STEP_ST = 1.0   # H/L separation (semitones) below which we can't tell
FLAT_ST = 1.5       # best-fit H/L step below this -> abstain, tell user to exaggerate
PAD_S = 0.3         # CTC model drops the first mora without leading silence
CONF = 0.5          # SwiftF0 voicing confidence
DECL_ST = -0.4      # natural downdrift per mora (semitones)
SIGMA2 = 2.0        # residual variance (st^2) used to turn residual gaps into confidence


def equivalent(k1: int, k2: int, n: int, utterance_final: bool) -> bool:
    """Accent types we cannot (or should not) tell apart in this position."""
    p1, p2 = accent.pattern(n, k1), accent.pattern(n, k2)
    if p1 == p2 or {k1, k2} <= {0, n}:
        return True
    # utterance-final lowering: the last mora drops regardless, so don't judge on it alone
    return utterance_final and p1[:-1] == p2[:-1]


@dataclass
class PhraseResult:
    phrase: accent.Phrase
    mora_st: list[float]          # NaN = no reliable pitch (devoiced etc.)
    heard_acc: int | None
    expected_ok: bool | None      # None = not confident
    confidence: float             # 0..1
    ambiguous_with: list[int]     # acc types indistinguishable here (e.g. odaka vs heiban)
    too_flat: bool = False        # pitch movement too small to judge -> coach "exaggerate"

    @property
    def heard_pattern(self) -> str:
        return accent.pattern(len(self.phrase.moras), self.heard_acc) if self.heard_acc is not None else "?"


@dataclass
class Result:
    heard_kana: str
    phrases: list[PhraseResult]
    spans: list[align.MoraSpan] | None


def mora_pitches(tr: pitch.Track, spans: list[align.MoraSpan]) -> list[float]:
    """Pitch of each mora = median of the middle half of its voiced frames.

    F0 glides between mora centres, so both edges are contaminated by the neighbours."""
    out = []
    for sp in spans:
        m = (tr.t >= sp.start) & (tr.t < sp.end) & ~np.isnan(tr.st)
        v = tr.st[m]
        if len(v) == 0:
            out.append(float("nan"))
            continue
        q = len(v) // 4
        out.append(float(np.median(v[q:len(v) - q])))
    return out


def fit_acc(st: list[float], n: int) -> list[tuple[float, int]]:
    """Residual of every accent type 0..n under y = a + B*H + DECL*i (a, B free; B >= MIN_STEP).

    Lower = better. Declination is fixed so short phrases can't be fit by any template."""
    y = np.array(st, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return []
    idx = np.arange(n, dtype=float)
    yd = y - DECL_ST * idx
    scores = []
    for k in range(0, n + 1):
        h = np.array([1.0 if c == "H" else 0.0 for c in accent.pattern(n, k)])
        hk = h[ok]
        if hk.min() == hk.max():   # template has no H/L contrast on the voiced morae
            scores.append((float(np.var(yd[ok]) * ok.sum()) + 1.0, k, 0.0))
            continue
        X = np.stack([np.ones(ok.sum()), hk], 1)
        (a, b), *_ = np.linalg.lstsq(X, yd[ok], rcond=None)
        b = max(b, MIN_STEP_ST)
        a = float(np.mean(yd[ok] - b * hk))
        scores.append((float(np.sum((a + b * hk - yd[ok]) ** 2)), k, float(b)))
    return sorted(scores)


def grade(audio: np.ndarray, sr: int, phrases: list[accent.Phrase]) -> Result:
    pad = np.zeros(int(PAD_S * 16000), np.float32)
    a16 = np.concatenate([pad, align.to16k(audio, sr), pad])
    lp = align.logprobs(a16)
    heard = align.transcribe(lp)
    morae = [m.hira for p in phrases for m in p.moras]
    spans = align.force_align(lp, morae)
    tr = pitch.track(a16, 16000, conf_thresh=CONF)
    results: list[PhraseResult] = []
    if spans is None:
        return Result(heard, [PhraseResult(p, [], None, None, 0.0, []) for p in phrases], None)
    all_st = mora_pitches(tr, spans)
    i = 0
    for pi, p in enumerate(phrases):
        n = len(p.moras)
        st = all_st[i:i + n]
        i += n
        scores = fit_acc(st, n)
        if not scores:
            results.append(PhraseResult(p, st, None, None, 0.0, []))
            continue
        final = pi == len(phrases) - 1
        best_r, best_k, step = scores[0]
        same = [k for _, k, _ in scores if equivalent(k, best_k, n, final)]
        runner = next((r for r, k, _ in scores if k not in same), best_r + 20)
        conf = float(1 - np.exp(-(runner - best_r) / SIGMA2))
        flat = step < FLAT_ST
        if flat:
            ok = None
        elif p.acc in same:
            ok = True if conf >= 0.5 else None
        else:
            exp_r = next(r for r, k, _ in scores if k == p.acc)
            ok = False if (1 - np.exp(-(exp_r - best_r) / SIGMA2)) >= 0.5 else None
        results.append(PhraseResult(p, st, best_k, ok, conf, sorted(set(same) - {best_k}), flat))
    return Result(heard, results, spans)
