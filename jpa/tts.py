"""VOICEVOX TTS with explicit accent control (AquesTalk-style kana: ' marks nucleus, / separates phrases)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import accent

VV = Path(__file__).resolve().parents[1] / "vendor" / "voicevox" / "vv"
_syn = None
_loaded: set[str] = set()


def synth():
    global _syn
    if _syn is None:
        from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer
        ort = Onnxruntime.load_once(filename=str(VV / "onnxruntime/lib/libvoicevox_onnxruntime.1.17.3.dylib"))
        _syn = Synthesizer(ort, OpenJtalk(str(VV / "dict/open_jtalk_dic_utf_8-1.11")))
    return _syn


def load(vvm: str):
    from voicevox_core.blocking import VoiceModelFile
    if vvm in _loaded:
        return
    with VoiceModelFile.open(str(VV / "models/vvms" / f"{vvm}.vvm")) as m:
        synth().load_voice_model(m)
    _loaded.add(vvm)


def voices(vvm: str) -> list[tuple[str, str, int]]:
    from voicevox_core.blocking import VoiceModelFile
    with VoiceModelFile.open(str(VV / "models/vvms" / f"{vvm}.vvm")) as m:
        return [(c.name, s.name, s.id) for c in m.metas for s in c.styles if s.type == "talk"]


def to_kana(phrases: list[accent.Phrase], accs: list[int] | None = None) -> str:
    parts = []
    for i, p in enumerate(phrases):
        acc = p.acc if accs is None else accs[i]
        s = ""
        for mi, m in enumerate(p.moras):
            s += ("_" if m.devoiced else "") + m.kana.replace("ヲ", "オ")
            # heiban is written as a mark on the last mora: within one phrase it is LH..H,
            # identical to odaka (they only differ on the following phrase)
            if mi + 1 == (acc or len(p.moras)):
                s += "'"
        parts.append(s)
    return "/".join(parts)


ST = np.log(2) / 12  # one semitone in VOICEVOX's log-F0 units


def enforce_pattern(q, phrases: list[accent.Phrase], accs: list[int], range_st: float = 4.0,
                    decl_st: float = -0.3):
    """Overwrite VOICEVOX's predicted mora pitches so the H/L pattern is exactly the dictionary's.

    Keeps the voice's own average pitch; unvoiced/devoiced morae stay 0."""
    voiced = [m.pitch for ap in q.accent_phrases for m in ap.moras if m.pitch > 0]
    base = float(np.mean(voiced)) if voiced else 5.0
    for ap, p, acc in zip(q.accent_phrases, phrases, accs):
        pat = accent.pattern(len(p.moras), acc)
        for i, (m, c) in enumerate(zip(ap.moras, pat)):
            if m.pitch > 0:
                m.pitch = base + ST * ((range_st / 2 if c == "H" else -range_st / 2) + decl_st * i)
    return q


def speak(phrases: list[accent.Phrase], style_id: int, accs: list[int] | None = None,
          speed: float = 1.0, pitch_shift: float = 0.0, enforce: bool = True,
          range_st: float = 4.0) -> tuple[np.ndarray, int]:
    import io
    import soundfile as sf
    accs = accs if accs is not None else [p.acc for p in phrases]
    q = synth().create_audio_query_from_kana(to_kana(phrases, accs), style_id)
    if enforce:
        enforce_pattern(q, phrases, accs, range_st)
    q.speed_scale = speed
    q.pitch_scale = pitch_shift
    wav, sr = sf.read(io.BytesIO(synth().synthesis(q, style_id)), dtype="float32")
    return wav, sr
