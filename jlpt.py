"""JLPT level per word (Waller's tanos.co.uk lists, CC BY, via elzup/jlpt-word-list) and per sentence.
Sentence level = hardest content word (5 = N5 easiest ... 1 = N1, 0 = beyond / not in any list)."""
import csv, re
from pathlib import Path
import pyopenjtalk as pj

SRC = Path(__file__).parent / "vendor" / "jlpt-word-list" / "src"
hira = lambda s: "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
LEVEL: dict[str, int] = {}
for n in (1, 2, 3, 4, 5):                      # later (easier) levels overwrite harder ones
    for row in csv.DictReader(open(SRC / f"n{n}.csv", encoding="utf-8")):
        for form in re.split(r"[;；、,]", row["expression"]) + re.split(r"[;；、,]", row["reading"]):
            form = form.strip().strip("～〜")
            if form:
                LEVEL[form] = max(LEVEL.get(form, 0), n)
SKIP_POS = {"記号", "助詞", "助動詞", "フィラー", "感動詞"}
SKIP_G1 = {"固有名詞", "数", "非自立", "接尾"}


def word_levels(text):
    out = []
    for n in pj.run_frontend(text):
        if n["pos"] in SKIP_POS or n.get("pos_group1") in SKIP_G1 or not n["pron"]:
            continue
        base = n["orig"] if n["orig"] not in ("", "*") else n["string"]
        lv = LEVEL.get(base) or LEVEL.get(hira(n["read"])) or LEVEL.get(n["string"]) or 0
        out.append((base, lv))
    return out


def sentence_level(text):
    lv = [l for _, l in word_levels(text)]
    return min(lv) if lv else 5
