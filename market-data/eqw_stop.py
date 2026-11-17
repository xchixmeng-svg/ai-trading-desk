"""eqw_stop (2026-11-18): the equal-risk quarterly/monthly re-pick of library_wf.py, run on all 263 streams saved by
dynamic_wf2.py, with Chi's stop inside each period (fall S % from the period high -> flat until the next rebuild).
Each stream scaled to 25 % yearly volatility from its previous 12 months; keep 12-month Sharpe >= 0.5 and profit,
best variant per family, best 4 families by Sharpe, equal weight; whole mix x 1 / 2 / 3."""
import json
import numpy as np, pandas as pd
M = np.load("dynamic_streams.npy"); meta = json.load(open("dynamic_streams.json")); NM, FAM = meta["names"], meta["fam"]
d = pd.read_parquet("daily.parquet"); NC = set(open("noncrypto.txt").read().split()); d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
days = np.sort(d.day.unique()); assert len(days) == len(M)
def pick(a):
    t0 = int((pd.Timestamp(a, unit="ms", tz="UTC") - pd.DateOffset(months=12)).timestamp() * 1000); tr = (days >= t0) & (days < a)
    best = {}
    for k, nm in enumerate(NM):
        v = M[tr, k]; sd = v.std()
        if sd <= 0 or (v != 0).sum() < 20 or v.sum() <= 0: continue
        sh = v.mean() / sd * np.sqrt(365)
        if sh < .5: continue
        f = FAM[nm]
        if f not in best or sh > best[f][0]: best[f] = (sh, k, min(5.0, .25 / (sd * np.sqrt(365))))
    return sorted(best.values(), reverse=True)[:4]
res = {}
for P in (1, 2, 3):
    per = []; q = pd.Timestamp("2021-01-01", tz="UTC")
    while q < pd.Timestamp("2026-11-01", tz="UTC"):
        q2 = min(q + pd.DateOffset(months=P), pd.Timestamp("2026-11-01", tz="UTC"))
        a, b = int(q.timestamp() * 1000), int(q2.timestamp() * 1000); per.append((q, a, b, pick(a))); q = q2
    for lev in (1, 2, 3):
        for S_ in (None, .10, .15, .20):
            eq = 500.; pa = 500.; mdd = 0.; yr = {}; stops = 0
            for q, a, b, sel in per:
                te = (days >= a) & (days < b)
                r = lev * np.mean([M[te, k] * s for _, k, s in sel], axis=0) if sel else np.zeros(te.sum())
                pk = eq; off = False
                for t, x in zip(days[te], np.clip(r, -.99, None)):
                    if not off:
                        eq *= 1 + x; pk = max(pk, eq)
                        if S_ and eq <= pk * (1 - S_): off = True; stops += 1
                    pa = max(pa, eq); mdd = max(mdd, 1 - eq / pa); yr[pd.Timestamp(t, unit="ms").year] = round(eq)
            lab = f"每{P}個月｜{lev}倍｜" + ("不停損" if S_ is None else f"回落{S_:.0%}就停")
            print(f"{lab}: 500U → {eq:,.0f}U，最大回落 {mdd:.0%}，停了 {stops} 期，各年底 {yr}", flush=True)
            res[lab] = {"end": round(eq), "max_dd": round(mdd, 3), "stops": stops, "year_end": yr}
    if P == 2:
        for q, a, b, sel in per: print(f"   {q:%Y-%m}: " + "、".join(NM[k] for _, k, _ in sel))
json.dump(res, open("eqw_stop.json", "w"), ensure_ascii=False, indent=1)
