"""Fair test: JVS held-out speakers AND held-out sentences (train on 20 sentences, test on the other 10)."""
import random, numpy as np
from train_combined import (J, V, FV, Feats, bad, test_spk, J_train, dataset, fit, evaluate, fmt, old)

sents = sorted({u["sent"] for u in V}); random.Random(3).shuffle(sents)
test_sent = set(sents[:10])
FJ = Feats(J)
in_jtrain = {id(u) for u in J_train}
X, y = dataset(FJ, lambda u: id(u) in in_jtrain)
B = fit(X, y)
X2, y2 = dataset(FV, lambda u: u["spk"] not in test_spk and u["sent"] not in test_sent, bad)
C = fit(np.vstack([X, X2]), np.concatenate([y, y2]))
te = lambda u: u["spk"] in test_spk and u["sent"] in test_sent
for hi, lo in [(0.8, 0.2), (0.9, 0.1)]:
    print(f"\n=== hi={hi} lo={lo}  (unseen speakers + unseen sentences)")
    for name, m in [("A: JSUT-300", old), ("B: JSUT-4000", B), ("C: JSUT-4000 + JVS 70spk/20sent", C)]:
        ev = evaluate(m, FV, hi, lo, te, bad, by_gender=True)
        print(f"{name}\n   M  {fmt(ev['M'])}\n   F  {fmt(ev['F'])}")
