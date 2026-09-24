"""Text -> accent phrases with expected per-mora H/L (Tokyo standard, via OpenJTalk)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pyopenjtalk as pj

SMALL = set("ャュョァィゥェォヮ")
DEVOICE_MARK = "’"


@dataclass
class Mora:
    kana: str          # katakana, 1-2 chars
    devoiced: bool = False

    @property
    def hira(self) -> str:
        out = []
        for ch in self.kana:
            if ch == "ヴ":
                out.append("ぶ")
            elif "ァ" <= ch <= "ヶ":
                out.append(chr(ord(ch) - 0x60))
            else:
                out.append(ch)
        return "".join(out)


@dataclass
class Phrase:
    surface: str
    moras: list[Mora]
    acc: int                      # 0 = heiban, k = drop after mora k
    node_idx: int = 0             # index of head NJD node (for resynthesis)
    alt_accs: list[int] = field(default_factory=list)

    @property
    def pattern(self) -> str:
        return pattern(len(self.moras), self.acc)

    @property
    def reading(self) -> str:
        return "".join(m.kana for m in self.moras)


def pattern(n: int, acc: int) -> str:
    if n == 0:
        return ""
    if acc == 1:
        return "H" + "L" * (n - 1)
    if acc == 0:
        return "L" + "H" * (n - 1)
    return "L" + "H" * (acc - 1) + "L" * (n - acc)


def split_moras(pron: str) -> list[Mora]:
    moras: list[Mora] = []
    for ch in pron:
        if ch == DEVOICE_MARK:
            if moras:
                moras[-1].devoiced = True
        elif ch in SMALL and moras:
            moras[-1].kana += ch
        else:
            moras.append(Mora(ch))
    return moras


def analyze(text: str, njd: list | None = None) -> tuple[list[Phrase], list]:
    """Returns accent phrases plus the raw NJD features (for resynthesis)."""
    njd = njd if njd is not None else pj.run_frontend(text)
    phrases: list[Phrase] = []
    for i, n in enumerate(njd):
        if n["pos"] == "記号" or not n["pron"] or n["pron"] in ("、", "。", "？", "！"):
            continue
        moras = split_moras(n["pron"])
        if not moras:
            continue
        if n["chain_flag"] == 1 and phrases:
            phrases[-1].moras += moras
            phrases[-1].surface += n["string"]
        else:
            phrases.append(Phrase(n["string"], moras, n["acc"], node_idx=i))
    for p in phrases:
        p.acc = min(p.acc, len(p.moras))
    return phrases, njd


def with_accents(njd: list, overrides: dict[int, int]) -> list:
    """Copy of NJD with phrase-head accents overridden {node_idx: acc}."""
    out = copy.deepcopy(njd)
    for idx, acc in overrides.items():
        out[idx]["acc"] = acc
    return out


def synthesize(njd: list, speed: float = 1.0, half_tone: float = 0.0):
    labels = pj.make_label(njd)
    wav, sr = pj.synthesize(labels, speed=speed, half_tone=half_tone)
    return (wav / 32768.0).astype("float32"), sr
