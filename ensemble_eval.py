"""Tree (v3) vs sequence vs ensemble on the fixed test sets, with a threshold sweep.
Usage: ensemble_eval.py TREE.pkl SEQ.pt OUT.md"""
import pickle, random, sys
from collections import defaultdict
import numpy as np, torch
import train_combined as tc, learn_v3
from learn import posterior
from jpa.grade import equivalent
import seq_model as sm

tree = pickle.load(open(sys.argv[1], "rb"))
sents = sorted({u["sent"] for u in tc.V}); random.Random(3).shuffle(sents); TEST_SENT = set(sents[:10])
in_jtrain = {id(u) for u in tc.J_train}
tests = {"JSUT": sm.items_of(tc.J, lambda u: id(u) not in in_jtrain),
         "JVS": sm.items_of(tc.V, lambda u: u["spk"] in tc.test_spk and u["sent"] in TEST_SENT, tc.bad)}
net = sm.Net(tests["JSUT"][0][3].shape[1]); net.load_state_dict(torch.load(sys.argv[2])); net.eval()

# --- clean JVS subset: phrase = one noun + particles, OpenJTalk label agrees with Kanjium dictionary
import pyopenjtalk as pj
from collections import defaultdict as dd
from pathlib import Path
kj = dd(set)
for l in open("vendor/kanjium/accents.txt"):
    w, r, a = l.rstrip("\n").split("\t")
    kj[w] |= {int(x) for x in a.split(",") if x.isdigit()}
text = dict(l.strip().split(":", 1) for l in open(Path("vendor/jvs/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt")) if ":" in l)
clean = set()
for sent, t in text.items():
    phr = []
    for nd in pj.run_frontend(t):
        if nd["pos"] == "記号" or not nd["pron"] or nd["pron"] in ("、", "。"):
            continue
        if nd["chain_flag"] == 1 and phr: phr[-1].append(nd)
        else: phr.append([nd])
    for pi, nodes in enumerate(phr):
        head, rest = nodes[0], nodes[1:]
        if head["pos"] == "名詞" and all(x["pos"] == "助詞" for x in rest) and head["acc"] in kj.get(head["string"], set()):
            clean.add((sent, pi))
tests["JVS-dict"] = [it for it in tests["JVS"] if (it[0]["sent"], it[1]) in clean]
print(f"JVS-dict subset: {len(tests['JVS-dict'])} of {len(tests['JVS'])} test phrases", flush=True)

posts = {}
for name, items in tests.items():
    tc.phrase_rows = learn_v3.phrase_rows
    utts = list({id(u): u for u, *_ in items}.values())
    FE = tc.Feats(utts)
    tp = {(id(u), pi): posterior(p, info["n"]) for (u, pi, info), p in zip(FE.items, FE.probs(tree))}
    rows = []
    with torch.no_grad():
        for u, pi, info, X, inph in items:
            sp = dict(enumerate(sm.log_post(net, X, inph).exp().numpy()))
            rows.append((u, info, tp[(id(u), pi)], sp))
    posts[name] = rows


def score(rows, combine, hi, lo):
    r = random.Random(0); T = defaultdict(lambda: defaultdict(int))
    for u, info, tp, sp in rows:
        n, gold, final = info["n"], info["acc"], info["final"]
        post = combine(tp, sp)
        alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
        g = tc.gender.get(u.get("spk"), "all")
        for exp, kind in [(gold, "c")] + ([(r.choice(alts), "w")] if alts else []):
            pe = sum(v for k, v in post.items() if equivalent(k, exp, n, final))
            ok = True if pe >= hi else False if pe <= lo else None
            for key in {"all", g}:
                T[key]["n" + kind] += 1; T[key][(kind, ok)] += 1
    f = lambda k: dict(FA=100 * T[k][("c", False)] / T[k]["nc"], MISS=100 * T[k][("w", True)] / T[k]["nw"],
                       abst=100 * (T[k][("c", None)] + T[k][("w", None)]) / (T[k]["nc"] + T[k]["nw"]))
    return {k: f(k) for k in T}


def geo(a, b, w=0.5):
    ks = a.keys(); z = {k: (max(a[k], 1e-6) ** (1 - w)) * (max(b[k], 1e-6) ** w) for k in ks}; s = sum(z.values())
    return {k: v / s for k, v in z.items()}


combos = {"tree v3": lambda t, s: t, "seq": lambda t, s: s,
          "avg": lambda t, s: {k: (t[k] + s[k]) / 2 for k in t}, "geo-mean": geo}
out = ["| Model | Thresholds | JVS FA M / F | JVS MISS | JVS abstain | JVS-dict FA / MISS / abst | JSUT FA | JSUT MISS | JSUT abstain |", "|---|---|---|---|---|---|---|---|---|"]
for name, comb in combos.items():
    for hi, lo in [(0.8, 0.2), (0.9, 0.1), (0.95, 0.05)]:
        v, j = score(posts["JVS"], comb, hi, lo), score(posts["JSUT"], comb, hi, lo)["all"]
        d = score(posts["JVS-dict"], comb, hi, lo)["all"]
        out.append(f"| {name} | {hi}/{lo} | {v['M']['FA']:.1f} / {v['F']['FA']:.1f} | {v['all']['MISS']:.1f} | {v['all']['abst']:.1f} | "
                   f"{d['FA']:.1f} / {d['MISS']:.1f} / {d['abst']:.1f} | "
                   f"{j['FA']:.1f} | {j['MISS']:.1f} | {j['abst']:.1f} |")
open(sys.argv[3], "a").write("\n".join(out) + "\n")
print("\n".join(out))
