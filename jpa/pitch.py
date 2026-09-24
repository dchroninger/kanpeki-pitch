"""F0 tracking (SwiftF0) -> semitones relative to speaker median."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from swift_f0 import SwiftF0

_det = None


@dataclass
class Track:
    t: np.ndarray       # seconds
    hz: np.ndarray
    conf: np.ndarray
    st: np.ndarray      # semitones re: median voiced F0, NaN when unvoiced

    def voiced(self, thresh: float = 0.9) -> np.ndarray:
        return self.conf >= thresh


def track(audio: np.ndarray, sr: int, conf_thresh: float = 0.9) -> Track:
    global _det
    if _det is None:
        _det = SwiftF0()
    r = _det.detect(audio.astype(np.float32), sr, fmin=60.0, fmax=500.0)
    hz = np.asarray(r.pitch_hz, dtype=np.float64)
    conf = np.asarray(r.confidence, dtype=np.float64)
    v = (conf >= conf_thresh) & (hz > 0)
    st = np.full_like(hz, np.nan)
    if v.any():
        ref = np.median(hz[v])
        st[v] = 12 * np.log2(hz[v] / ref)
    return Track(np.asarray(r.timestamps, dtype=np.float64), hz, conf, st)
