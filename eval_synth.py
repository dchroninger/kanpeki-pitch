"""Batch eval of the grader on VOICEVOX audio with known correct / wrong accents.

For each item x voice x condition: one correct rendition, plus one rendition where a single
phrase gets a different accent. Reports catch rate, false-alarm rate, abstain rate."""
import json, random, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers.utils import logging as hl; hl.set_verbosity_error()
from jpa import accent, grade, tts

ITEMS = [
    # minimal pairs / single words + particle
    "箸が", "橋が", "端が", "雨が", "飴が", "柿が", "牡蠣が", "紙が", "髪が", "神が",
    "今が", "居間が", "花が", "鼻が", "酒が", "鮭が", "日本が", "二本が",
    "先生が", "学校に", "友達と", "電話を", "図書館で", "病院に",
    # verbs / adjectives (devoicing, conjugation)
    "聞きます", "食べた", "行きました", "高いです", "暑かった", "好きです",
    # sentences
    "箸が好きです", "雨が降っています", "日本語を勉強しています", "明日は学校に行きます",
    "昨日友達と映画を見ました", "この本はとても面白いです", "駅までどのくらいかかりますか",
]
VOICES = [("0", 2), ("0", 3), ("1", 14), ("4", 11), ("15", 13), ("2", 16), ("6", 30), ("9", 12)]
rng = random.Random(0)


def add_noise(x, snr_db):
    p = np.mean(x ** 2)
    return (x + rng.gauss(0, 1) * 0 + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, np.sqrt(p / 10 ** (snr_db / 10)), x.shape)).astype(np.float32)


def main(n_per_item=6):
    for v, _ in VOICES:
        tts.load(v)
    rows = []
    for text in ITEMS:
        phrases, _ = accent.analyze(text)
        base = [p.acc for p in phrases]
        for _ in range(n_per_item):
            vvm, sid = rng.choice(VOICES)
            cond = dict(speed=rng.choice([0.8, 1.0, 1.15]), range_st=rng.choice([2.0, 3.0, 4.0, 6.0]),
                        snr=rng.choice([None, None, 20, 10]))
            # wrong: pick a phrase, change its accent to a different, audibly-distinct one
            pi = rng.randrange(len(phrases))
            n = len(phrases[pi].moras)
            final = pi == len(phrases) - 1
            alts = [k for k in range(n + 1) if not grade.equivalent(k, base[pi], n, final)]
            renditions = [("correct", base, None)]
            if alts:
                wrong = base.copy(); wrong[pi] = rng.choice(alts)
                renditions.append(("wrong", wrong, pi))
            for kind, accs, target in renditions:
                try:
                    wav, sr = tts.speak(phrases, sid, accs, speed=cond["speed"], range_st=cond["range_st"])
                except Exception as e:
                    print("SKIP", text, tts.to_kana(phrases, accs), e, file=sys.stderr)
                    continue
                if cond["snr"]:
                    wav = add_noise(wav, cond["snr"])
                r = grade.grade(wav, sr, phrases)
                for j, pr in enumerate(r.phrases):
                    truth_wrong = kind == "wrong" and j == target
                    rows.append(dict(text=text, phrase=pr.phrase.surface, kind="wrong" if truth_wrong else "correct",
                                     exp=pr.phrase.acc, said=accs[j], heard=pr.heard_acc, ok=pr.expected_ok,
                                     conf=round(pr.confidence, 2), flat=pr.too_flat, n=len(pr.phrase.moras), sid=sid,
                                     heard_kana=r.heard_kana, st=[None if x != x else round(x, 1) for x in pr.mora_st], **cond))
            print(text, file=sys.stderr, flush=True)
    json.dump(rows, open("eval_rows.json", "w"), ensure_ascii=False, indent=0)
    report(rows)


def report(rows):
    def pct(a, b): return f"{100 * a / b:5.1f}%" if b else "  n/a"
    for kind in ("correct", "wrong"):
        rs = [r for r in rows if r["kind"] == kind]
        good = sum(r["ok"] is (kind == "correct") for r in rs)
        bad = sum(r["ok"] is (kind != "correct") for r in rs)
        abst = sum(r["ok"] is None for r in rs)
        label = ("passed", "FALSE ALARM") if kind == "correct" else ("caught", "MISSED")
        flat = sum(r["ok"] is None and r.get("flat", False) for r in rs)
        print(f"{kind:8s} n={len(rs):4d}  {label[0]} {pct(good, len(rs))}  {label[1]} {pct(bad, len(rs))}  abstain {pct(abst, len(rs))} (of which too-flat {pct(flat, len(rs))})")
    for key in ("range_st", "snr", "speed"):
        print(f"\nby {key}:")
        for val in sorted({str(r[key]) for r in rows}):
            rs = [r for r in rows if str(r[key]) == val]
            fa = sum(r["kind"] == "correct" and r["ok"] is False for r in rs)
            nc = sum(r["kind"] == "correct" for r in rs)
            ms = sum(r["kind"] == "wrong" and r["ok"] is True for r in rs)
            nw = sum(r["kind"] == "wrong" for r in rs)
            ab = sum(r["ok"] is None for r in rs)
            print(f"  {val:6s} false-alarm {pct(fa, nc)}  missed {pct(ms, nw)}  abstain {pct(ab, len(rs))}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(json.load(open("eval_rows.json")))
    else:
        main()
