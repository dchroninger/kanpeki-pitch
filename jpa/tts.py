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


_VOWEL_ROWS = {"ア": "アカサタナハマヤラワガザダバパャァ", "イ": "イキシチニヒミリギジヂビピィ",
               "ウ": "ウクスツヌフムユルグズヅブプュゥヴ", "エ": "エケセテネヘメレゲゼデベペェ",
               "オ": "オコソトノホモヨロヲゴゾドボポョォ"}
_VOWEL = {ch: v for v, row in _VOWEL_ROWS.items() for ch in row}


def _vv_mora(kana: str, prev: str) -> str:
    """VOICEVOX kana rejects ー and ヲ: spell the long vowel out, ヲ as オ."""
    if kana == "ー":
        return _VOWEL.get(prev[-1:], "ウ") if prev else "ウ"
    return kana.replace("ヲ", "オ")


def to_kana(phrases: list[accent.Phrase], accs: list[int] | None = None) -> str:
    parts = []
    for i, p in enumerate(phrases):
        acc = p.acc if accs is None else accs[i]
        s = ""
        for mi, m in enumerate(p.moras):
            prev = _vv_mora(p.moras[mi - 1].kana, "") if mi else ""
            s += ("_" if m.devoiced else "") + _vv_mora(m.kana, prev)
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


def _violations(q, pat, rise=1.0, max_sag=1.5):
    """Does contour q (log-F0 per mora, 0 = unvoiced) contradict H/L pattern pat? Compares voiced neighbours."""
    v = [i for i, x in enumerate(q) if x > 0]
    for a, b in zip(v, v[1:]):
        d = (q[b] - q[a]) / ST
        if pat[a] == "L" and pat[b] == "H" and d < rise: return True
        if pat[a] == "H" and pat[b] == "L" and d > -rise: return True
        if pat[a] == pat[b] and d < -max_sag: return True   # a fake drop inside a level stretch
    return False


def natural_fix(q, phrases: list[accent.Phrase], accs: list[int], range_st: float = 4.0):
    """Keep VOICEVOX's own contour; per phrase, blend toward the dictionary pattern only as far as needed.

    Returns the blend weight used per phrase (0 = untouched, 1 = fully enforced)."""
    used = []
    for ap, p, acc in zip(q.accent_phrases, phrases, accs):
        pat = accent.pattern(len(p.moras), acc)
        nat = [m.pitch for m in ap.moras]
        voiced = [x for x in nat if x > 0]
        if not voiced:
            used.append(0.0); continue
        base = float(np.mean(voiced))
        tgt = [base + ST * ((range_st / 2 if c == "H" else -range_st / 2) - 0.3 * i) if x > 0 else 0.0
               for i, (x, c) in enumerate(zip(nat, pat))]
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            new = [x + w * (t - x) if x > 0 else 0.0 for x, t in zip(nat, tgt)]
            if not _violations(new, pat) or w == 1.0:
                break
        for m, x in zip(ap.moras, new):
            m.pitch = x
        used.append(w)
    return used


def speak(phrases: list[accent.Phrase], style_id: int, accs: list[int] | None = None,
          speed: float = 1.0, pitch_shift: float = 0.0, enforce: bool | str = True,
          range_st: float = 4.0) -> tuple[np.ndarray, int]:
    """enforce: True/"enforce" = flat dictionary levels, "natural" = minimal fix of VOICEVOX's own contour,
    False/"raw" = VOICEVOX untouched."""
    import io
    import soundfile as sf
    accs = accs if accs is not None else [p.acc for p in phrases]
    q = synth().create_audio_query_from_kana(to_kana(phrases, accs), style_id)
    if enforce == "natural":
        speak.last_blend = natural_fix(q, phrases, accs, range_st)
    elif enforce and enforce != "raw":
        enforce_pattern(q, phrases, accs, range_st)
    q.speed_scale = speed
    q.pitch_scale = pitch_shift
    wav, sr = sf.read(io.BytesIO(synth().synthesis(q, style_id)), dtype="float32")
    return wav, sr


def speak_groups(groups: list[list[accent.Phrase]], style_id: int, speed: float = 0.9,
                 pause_scale: float = 1.4, enforce="natural", range_st: float = 4.0):
    """Speak clause groups with a real pause between them (、 in VOICEVOX kana), not one run-on breath."""
    import io
    import soundfile as sf
    groups = [g for g in groups if g]
    flat = [p for g in groups for p in g]
    accs = [p.acc for p in flat]
    kana = "、".join(to_kana(g) for g in groups)
    q = synth().create_audio_query_from_kana(kana, style_id)
    if enforce == "natural":
        natural_fix(q, flat, accs, range_st)
    elif enforce and enforce != "raw":
        enforce_pattern(q, flat, accs, range_st)
    q.speed_scale = speed
    q.pause_length_scale = pause_scale
    q.post_phoneme_length = 0.1
    wav, sr = sf.read(io.BytesIO(synth().synthesis(q, style_id)), dtype="float32")
    return wav, sr


def clauses(sentence: str) -> list[list[accent.Phrase]]:
    """Split a sentence at 、/，/・-free clause marks and analyse each clause separately."""
    import re
    return [accent.analyze(c)[0] for c in re.split(r"[、，,]", sentence) if c.strip()]
