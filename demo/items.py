"""Practice items for word mode and minimal-pair mode.

Words: JLPT vocabulary (Waller lists) whose Kanjium accent is known and whose OpenJTalk reading matches the
list's reading (so furigana, aligner morae and accent all agree). Nouns can take a particle and a follow-up
so the particle isn't utterance-final (final lowering would hide heiban vs odaka).

Minimal pairs: JLPT nouns sharing a reading but with audibly different accents (箸/橋/端)."""
import csv, random, re
from collections import defaultdict
from pathlib import Path

import pyopenjtalk as pj

from jpa import accent

ROOT = Path(__file__).resolve().parents[1]
hira = lambda s: "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)

# Kanjium: word \t reading \t accents ("0", "0,2", "1(2)" ...) — every listed accent is accepted
KANJIUM: dict[tuple[str, str], list[int]] = {}
for line in open(ROOT / "vendor/kanjium/accents.txt", encoding="utf-8"):
    w, r, a = line.rstrip("\n").split("\t")
    KANJIUM[(w, r)] = sorted({int(x) for x in re.findall(r"\d+", a)})

# particle + follow-up carriers (N5 grammar), chosen so the particle is never the last mora
CARRIERS = [("が", "あります"), ("を", "見ます")]
# classic teaching sets not all on the JLPT lists (added for minimal pairs; level = hardest JLPT member or N1)
CLASSIC = [("箸", "はし"), ("橋", "はし"), ("端", "はし"), ("雨", "あめ"), ("飴", "あめ"), ("柿", "かき"), ("牡蠣", "かき"),
           ("酒", "さけ"), ("鮭", "さけ"), ("神", "かみ"), ("紙", "かみ"), ("髪", "かみ"), ("花", "はな"), ("鼻", "はな"),
           ("今", "いま"), ("居間", "いま"), ("雲", "くも"), ("蜘蛛", "くも"), ("日", "ひ"), ("火", "ひ")]


def _jlpt_rows():
    src = ROOT / "vendor/jlpt-word-list/src"
    for n in (5, 4, 3, 2, 1):
        for row in csv.DictReader(open(src / f"n{n}.csv", encoding="utf-8")):
            forms = [f.strip() for f in re.split(r"[;；、,]", row["expression"]) if f.strip()]
            reading = re.split(r"[;；、,]", row["reading"])[0].strip()
            if forms and reading and "～" not in forms[0] and "〜" not in forms[0]:
                yield n, forms[0], reading, row["meaning"]


def build_words():
    """{id: word dict}. One entry per word (easiest JLPT level wins)."""
    words = {}
    rows = list(_jlpt_rows())
    listed = {w for _, w, _, _ in rows}
    rows += [(0, w, r, "") for w, r in CLASSIC if w not in listed]   # level 0 = not on the JLPT lists
    for level, w, reading, meaning in rows:
        if w in words:
            continue
        nodes = [x for x in pj.run_frontend(w) if x["pos"] != "記号"]
        if not nodes:
            continue
        pron = "".join(x["pron"] for x in nodes)
        if hira("".join(x["read"] for x in nodes)) != reading:
            continue                      # OpenJTalk would read it differently than the list → skip
        moras = accent.split_moras(pron)
        n = len(moras)
        accs = [a for a in KANJIUM.get((w, reading), []) if 0 <= a <= n]
        if not accs:
            continue
        pos = nodes[0]["pos"]
        noun = pos == "名詞" and nodes[0].get("pos_group1") not in ("接尾", "非自立", "代名詞", "数")
        words[w] = dict(word=w, reading=reading, level=level, accs=accs, n=n, noun=noun and len(nodes) == 1,
                        meaning=meaning.split(";")[0].strip())
    return words


def build_pairs(words):
    """Groups of nouns with the same reading and distinct single accents (all distinguishable with a carrier)."""
    by_reading = defaultdict(list)
    for w in words.values():
        if w["noun"] and len(w["accs"]) == 1:
            by_reading[w["reading"]].append(w)
    groups = []
    for reading, ws in by_reading.items():
        seen, members = set(), []
        for w in sorted(ws, key=lambda w: w["level"], reverse=True):
            if w["accs"][0] not in seen:
                seen.add(w["accs"][0]); members.append(w)
        if len(members) >= 2:
            groups.append(members)
    return groups


def item_text(w, particle_mode, rng=random):
    """-> (display text, carrier or None). particle_mode: word | particle | carrier."""
    if not w["noun"] or particle_mode == "word":
        return w["word"], None
    part, follow = rng.choice(CARRIERS)
    if particle_mode == "particle":
        return w["word"] + part, (part, "")
    return w["word"] + part + follow, (part, follow)


def item_phrases(w, text):
    """Accent phrases for the item text with the word's phrase forced to the dictionary accent.
    Returns (phrases, accepted accents per phrase)."""
    phrases, _ = accent.analyze(text)
    if not phrases:
        return [], []
    alts = [None for _ in phrases]      # follow-up phrases are context only: shown, not graded
    phrases[0].acc = w["accs"][0]
    alts[0] = list(w["accs"])
    return phrases, alts
