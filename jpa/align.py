"""Kana CTC model: free transcription (what was said) + forced alignment to morae."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "hiragana-asr"))

from src.asr.kana_vocab import KanaVocab  # noqa: E402
from src.asr.model import load_checkpoint  # noqa: E402

CKPT = ROOT / "models" / "best-medium-ep5-inference.pt"
FRAME_S = 0.02  # wav2vec2 stride

_model = None
_fe = None
vocab = KanaVocab()


def _load():
    global _model, _fe
    if _model is None:
        from transformers import Wav2Vec2FeatureExtractor
        _model = load_checkpoint(str(CKPT)).eval()
        _fe = Wav2Vec2FeatureExtractor.from_pretrained("reazon-research/japanese-wav2vec2-large")
    return _model, _fe


def to16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(audio).float(), sr, 16000).numpy()


def logprobs(audio16: np.ndarray) -> np.ndarray:
    model, fe = _load()
    x = fe(audio16, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        lg = model(x.input_values)["kana_logits"][0]
    return lg.log_softmax(-1).numpy()


def transcribe(lp: np.ndarray) -> str:
    return vocab.decode(lp.argmax(-1).tolist())


@dataclass
class MoraSpan:
    start: float
    end: float
    score: float  # mean log-prob of its tokens (alignment confidence)


def force_align(lp: np.ndarray, mora_hira: list[str]) -> list[MoraSpan] | None:
    """CTC Viterbi alignment. Returns one span per mora (end = next mora's start)."""
    toks, owner = [], []
    for mi, m in enumerate(mora_hira):
        for ch in m:
            if ch in vocab.stoi:
                toks.append(vocab.stoi[ch])
                owner.append(mi)
    if not toks:
        return None
    T, S = lp.shape[0], 2 * len(toks) + 1
    ext = [0] * S
    for i, t in enumerate(toks):
        ext[2 * i + 1] = t
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = lp[0, 0]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        for s in range(S):
            best, arg = prev[s], 0
            if s >= 1 and prev[s - 1] > best:
                best, arg = prev[s - 1], 1
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] and prev[s - 2] > best:
                best, arg = prev[s - 2], 2
            dp[t, s] = best + lp[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    path = np.zeros(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    # first frame of each token
    tok_first: dict[int, int] = {}
    tok_lp: dict[int, list[float]] = {}
    for t, s in enumerate(path):
        if s % 2 == 1:
            ti = s // 2
            tok_first.setdefault(ti, t)
            tok_lp.setdefault(ti, []).append(lp[t, ext[s]])
    if len(tok_first) != len(toks):
        return None
    n = len(mora_hira)
    starts = [None] * n
    scores = [[] for _ in range(n)]
    for ti, mi in enumerate(owner):
        starts[mi] = tok_first[ti] if starts[mi] is None else starts[mi]
        scores[mi] += tok_lp[ti]
    last = max(t for t, s in enumerate(path) if s % 2 == 1)
    spans = []
    for mi in range(n):
        st = starts[mi]
        en = starts[mi + 1] if mi + 1 < n else last + 3
        spans.append(MoraSpan(st * FRAME_S, en * FRAME_S, float(np.mean(scores[mi]))))
    return spans
