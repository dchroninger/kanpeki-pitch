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
    ext_a = np.array(ext)
    emit = lp[:, ext_a]                                    # (T, S)
    # skip transition s-2 -> s allowed onto a token that differs from the token two back
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext_a[2:] != 0) & (ext_a[2:] != ext_a[:-2])
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = emit[0, 0]
    if S > 1:
        dp[0, 1] = emit[0, 1]
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(can_skip, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stack = np.stack([prev, c1, c2])
        arg = stack.argmax(0)
        dp[t] = stack[arg, np.arange(S)] + emit[t]
        bp[t] = arg
    s = S - 1 if S < 2 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    path = np.zeros(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
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
    # morae with no in-vocab token (rare kana) borrow the previous mora's start
    for mi in range(n):
        if starts[mi] is None:
            starts[mi] = starts[mi - 1] if mi else 0
    last = max(t for t, s in enumerate(path) if s % 2 == 1)
    spans = []
    for mi in range(n):
        st = starts[mi]
        en = starts[mi + 1] if mi + 1 < n else last + 3
        spans.append(MoraSpan(st * FRAME_S, en * FRAME_S, float(np.mean(scores[mi]))))
    return spans
