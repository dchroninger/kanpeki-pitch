"""Sequence model (BiGRU) over per-mora pitch-contour samples -> softmax over accent types {0..n} per phrase.

Input per mora: K contour samples (phrase-relative st, interpolated), K voicing flags, utterance-relative
mid pitch, special-mora flag, devoiced flag, phrase-position features. 2 context morae each side (masked).
Loss: -log P(gold equivalence class) (odaka/heiban merged; utterance-final lowering merged).
Usage: seq_model.py LABEL OUT.md [batch pickles...]"""
import pickle, random, sys, time
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(4)
from jpa import accent
from jpa.grade import equivalent
from learn import contour, SPECIAL
import learn_v2

K, CTX = 6, 2


def utt_phrases(u):
    """Per phrase: (info, feature matrix [CTX+n+CTX, D], mask of in-phrase rows)."""
    t, st = contour(u)
    sti = learn_v2._interp(t, st)
    voiced = (~np.isnan(st)).astype(float)
    spans = u["spans"]
    rows, meta = [], []
    for pi, (reading, acc, dev) in enumerate(u["gold"]):
        moras = accent.split_moras(reading)
        for j, m in enumerate(moras):
            meta.append((pi, j, len(moras), m.kana in SPECIAL, dev[j]))
    for idx, (s, e) in enumerate(spans):
        q = s + (np.arange(K) + 0.5) / K * (e - s)
        y = np.interp(q, t, sti) if len(t) else np.zeros(K)
        v = np.interp(q, t, voiced) if len(t) else np.zeros(K)
        pi, j, n, spec, dv = meta[idx]
        rows.append(list(y) + list(v) + [float(spec), float(dv), j / n, float(j == 0), float(j == n - 1), (e - s) * 5])
    R = np.nan_to_num(np.array(rows, float))
    i = 0
    for pi, (reading, acc, dev) in enumerate(u["gold"]):
        n = len(accent.split_moras(reading))
        lo, hi = max(0, i - CTX), min(len(R), i + n + CTX)
        X = R[lo:hi].copy()
        inph = np.zeros(hi - lo, bool); inph[i - lo:i - lo + n] = True
        ref = np.median(X[inph][:, :K])
        X[:, :K] -= ref
        yield dict(n=n, acc=min(acc, n), final=pi == len(u["gold"]) - 1), X.astype(np.float32), inph
        i += n


class Net(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, h), nn.GELU())
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.nuc = nn.Linear(2 * h, 1)     # per mora: "drop after this mora"
        self.flat = nn.Linear(2 * h, 1)    # pooled: heiban

    def forward(self, x):                  # x: [1, L, d]
        hdn, _ = self.gru(self.inp(x))
        return self.nuc(hdn)[0, :, 0], self.flat(hdn.mean(1))[0, 0]


def log_post(net, X, inph):
    nuc, flat = net(torch.from_numpy(X)[None])
    logits = torch.cat([flat[None], nuc[torch.from_numpy(inph)]])   # index k: 0 = heiban, k = drop after mora k
    return torch.log_softmax(logits, 0)


def items_of(utts, keep, skip=frozenset()):
    out = []
    for u in utts:
        if not keep(u) or len(u["spans"]) != sum(len(accent.split_moras(r)) for r, _, _ in u["gold"]):
            continue  # cached before the small-kana splitter fix
        for pi, (info, X, inph) in enumerate(utt_phrases(u)):
            if (u.get("sent"), pi) in skip:
                continue
            out.append((u, pi, info, X, inph))
    return out


def run(net, items, hi, lo):
    import train_combined as tc
    net.eval(); r = random.Random(0)
    T = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for u, pi, info, X, inph in items:
            n, gold, final = info["n"], info["acc"], info["final"]
            post = dict(enumerate(log_post(net, X, inph).exp().numpy()))
            groups = ["all", tc.gender.get(u.get("spk"), "all")] + (["atama"] if gold == 1 else []) + (["naka"] if 1 < gold < n else [])
            alts = [k for k in range(n + 1) if not equivalent(k, gold, n, final)]
            for exp, kind in [(gold, "c")] + ([(r.choice(alts), "w")] if alts else []):
                pe = sum(v for k, v in post.items() if equivalent(k, exp, n, final))
                ok = True if pe >= hi else False if pe <= lo else None
                for g in set(groups):
                    T[g]["n" + kind] += 1; T[g][(kind, ok)] += 1
    pc = lambda g, a, b: 100 * T[g][a] / max(1, T[g][b])
    ab = lambda g: 100 * (T[g][("c", None)] + T[g][("w", None)]) / max(1, T[g]["nc"] + T[g]["nw"])
    return {g: dict(FA=pc(g, ("c", False), "nc"), MISS=pc(g, ("w", True), "nw"), caught=pc(g, ("w", False), "nw"), abst=ab(g)) for g in T}




if __name__ == "__main__":
    import train_combined as tc
    label, out_md, batch_files = sys.argv[1], sys.argv[2], sys.argv[3:]
    sents = sorted({u["sent"] for u in tc.V}); random.Random(3).shuffle(sents); TEST_SENT = set(sents[:10])
    in_jtrain = {id(u) for u in tc.J_train}
    new = [u for f in batch_files for u in pickle.load(open(f, "rb"))]
    bad_new = set()
    if new:  # label-error filter for new sentences, same rule as batch_eval (v1 + model A, train speakers)
        from learn import posterior
        FN = tc.Feats(new); fl = defaultdict(list)
        for (u, pi, info), p in zip(FN.items, FN.probs(tc.old)):
            if u["spk"] in tc.test_spk: continue
            post = posterior(p, info["n"])
            fl[(u["sent"], pi)].append(sum(v for k, v in post.items() if equivalent(k, info["acc"], info["n"], info["final"])) <= 0.2)
        bad_new = {k for k, v in fl.items() if np.mean(v) > 0.5}

    t0 = time.time()
    train = (items_of(tc.J, lambda u: id(u) in in_jtrain)
             + items_of(tc.V, lambda u: u["spk"] not in tc.test_spk and u["sent"] not in TEST_SENT, tc.bad)
             + items_of(new, lambda u: u["spk"] not in tc.test_spk, bad_new))
    test_j = items_of(tc.J, lambda u: id(u) not in in_jtrain)
    test_v = items_of(tc.V, lambda u: u["spk"] in tc.test_spk and u["sent"] in TEST_SENT, tc.bad)
    print(f"train phrases {len(train)}  test JSUT {len(test_j)}  JVS {len(test_v)}  ({time.time()-t0:.0f}s)", flush=True)

    torch.manual_seed(0)
    net = Net(train[0][3].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = random.Random(0)
    import os
    AUG, BAL = os.environ.get("AUG") == "1", os.environ.get("BAL") == "1"
    jsut_items = [it for it in train if "spk" not in it[0]]
    jvs_items = [it for it in train if "spk" in it[0]]
    print(f"AUG={AUG} BAL={BAL}  JSUT {len(jsut_items)} / JVS {len(jvs_items)} phrases", flush=True)
    for ep in range(6):
        net.train()
        epoch = (jvs_items + rng.sample(jsut_items, min(len(jsut_items), len(jvs_items)))) if BAL else list(train)
        rng.shuffle(epoch); tot = 0.0
        for bi in range(0, len(epoch), 64):
            opt.zero_grad(); loss = 0.0
            for u, pi, info, X, inph in epoch[bi:bi + 64]:
                if AUG:  # simulate other speakers: pitch-range scale + jitter on the contour samples
                    X = X.copy()
                    X[:, :K] = X[:, :K] * rng.uniform(0.55, 1.5) + np.float32(rng.gauss(0, 0.3)) * np.random.randn(*X[:, :K].shape).astype(np.float32)
                lp = log_post(net, X, inph)
                n = info["n"]
                eq = [k for k in range(n + 1) if equivalent(k, info["acc"], n, info["final"])]
                loss = loss - torch.logsumexp(lp[eq], 0)
            (loss / 64).backward(); opt.step(); tot += float(loss.detach())
        print(f"epoch {ep} loss {tot/len(epoch):.3f} ({time.time()-t0:.0f}s)", flush=True)


    rows = []
    for hi, lo, name in [(0.8, 0.2, "balanced"), (0.9, 0.1, "strict")]:
        v, j = run(net, test_v, hi, lo), run(net, test_j, hi, lo)
        rows.append(f"| {label} [seq] | {name} | {v['M']['FA']:.1f} / {v['F']['FA']:.1f} | {v['all']['MISS']:.1f} | {v['all']['caught']:.1f} | "
                    f"{v['all']['abst']:.1f} | {v['atama']['FA']:.1f} / {v['atama']['abst']:.1f} | {j['naka']['FA']:.1f} / {j['naka']['abst']:.1f} | "
                    f"{j['all']['FA']:.1f} / {j['all']['MISS']:.1f} / {j['all']['abst']:.1f} |")
    open(out_md, "a").write("\n".join(rows) + "\n")
    print("\n".join(rows))
    torch.save(net.state_dict(), f"seq_{label.replace(' ', '_')}.pt")
