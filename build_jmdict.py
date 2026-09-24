"""JMdict_e.gz -> jmdict_index.pkl: {headword (kanji or kana): [(readings, glosses, common)]}.
JMdict © EDRDG, CC BY-SA 4.0."""
import gzip, pickle
import xml.etree.ElementTree as ET

PRI = {"news1", "ichi1", "spec1", "spec2", "gai1"}
idx = {}
root = ET.parse(gzip.open("vendor/jmdict/JMdict_e.gz")).getroot()
for e in root.iter("entry"):
    kebs = [k.findtext("keb") for k in e.findall("k_ele")]
    rebs = [r.findtext("reb") for r in e.findall("r_ele")]
    common = any(p.text in PRI for p in e.iter() if p.tag in ("ke_pri", "re_pri"))
    senses = e.findall("sense")
    if not senses:
        continue
    gl = [g.text for g in senses[0].findall("gloss")][:3]
    if len(gl) < 2 and len(senses) > 1:
        gl += [g.text for g in senses[1].findall("gloss")][:1]
    item = (tuple(rebs), tuple(gl), common, tuple(kebs))
    for h in (kebs or rebs):
        idx.setdefault(h, []).append(item)
    if kebs:  # kana-only lookups (e.g. すやすや, たくさん) should still find kanji entries' readings
        for r in rebs:
            idx.setdefault(r, []).append(item)
pickle.dump(idx, open("jmdict_index.pkl", "wb"))
print(len(idx), "headwords")
