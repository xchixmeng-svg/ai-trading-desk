"""iter3 (2026-11-18): change HOW strategies are picked (iteration 1 = 12-month Sharpe, best 1 per family):
rank by Sharpe or by return / max fall (Calmar), 1 or 2 per family, look back 12 or 6 months, keep 4 or 6, equal
weight, leverage by distance from the account high (6/3/1, 8/4/2, 10/5/2), rebuild every 1 or 2 months."""
import json, itertools
import numpy as np, pandas as pd
M = np.load("final_streams.npy"); meta = json.load(open("final_streams.json")); NM, FAM = meta["names"], meta["fam"]
d = pd.read_parquet("daily.parquet"); NC = set(open("noncrypto.txt").read().split()); d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
days = np.sort(d.day.unique())
def mdd_(v):
    x = np.cumprod(1 + v); return float((1 - x / np.maximum.accumulate(x)).max())
def picks(a, look, rank, per_fam, topn):
    t0 = int((pd.Timestamp(a, unit="ms", tz="UTC") - pd.DateOffset(months=look)).timestamp() * 1000); tr = (days >= t0) & (days < a)
    cand = []
    for k, nm in enumerate(NM):
        v = M[tr, k]; sd = v.std()
        if sd <= 0 or (v != 0).sum() < 20 or v.sum() <= 0: continue
        sh = v.mean() / sd * np.sqrt(365)
        if sh < .5: continue
        sc = min(5.0, .25 / (sd * np.sqrt(365)))
        score = sh if rank == "sharpe" else float(np.prod(1 + v * sc) - 1) / max(mdd_(v * sc), .02)
        cand.append((score, k, sc, FAM[nm]))
    cand.sort(reverse=True); out = []; cnt = {}
    for c in cand:
        if cnt.get(c[3], 0) >= per_fam: continue
        cnt[c[3]] = cnt.get(c[3], 0) + 1; out.append(c[:3])
        if len(out) == topn: break
    return out
START = pd.Timestamp("2021-01-01", tz="UTC"); END = pd.Timestamp("2026-11-01", tz="UTC")
def plan(P, look, rank, pf, topn):
    q = START; lst = []
    while q < END:
        q2 = min(q + pd.DateOffset(months=P), END); a = int(q.timestamp() * 1000); lst.append((a, int(q2.timestamp() * 1000), picks(a, look, rank, pf, topn))); q = q2
    return lst
def sim(pl, tiers):
    eq = 500.; pa = 500.; mdd = 0.; yr = {}
    for a, b, sel in pl:
        for i in np.nonzero((days >= a) & (days < b))[0]:
            if sel:
                dd = 1 - eq / pa; lev = tiers[0] if dd < .1 else (tiers[1] if dd < .25 else tiers[2])
                eq *= 1 + max(lev * np.mean([M[i, k] * sc for _, k, sc in sel]), -.99)
            pa = max(pa, eq); mdd = max(mdd, 1 - eq / pa); yr[pd.Timestamp(int(days[i]), unit="ms").year] = round(eq)
    return eq, mdd, yr
res = []
for P, look, rank, pf, topn in itertools.product((1, 2), (12, 6), ("sharpe", "calmar"), (1, 2), (4, 6)):
    pl = plan(P, look, rank, pf, topn)
    for tiers in ((6, 3, 1), (8, 4, 2), (10, 5, 2)):
        eq, mdd, yr = sim(pl, tiers)
        res.append({"P": P, "look": look, "rank": rank, "per_fam": pf, "topn": topn, "tiers": tiers, "end": eq, "mdd": mdd, "yr": yr})
    print(P, look, rank, pf, topn, flush=True)
json.dump(res, open("iter3_grid.json", "w"), ensure_ascii=False, default=str)
print("\n回落 ≤60% 前 15：")
for r in sorted([r for r in res if r["mdd"] <= .6], key=lambda r: -r["end"])[:15]:
    print(f"每{r['P']}月 看{r['look']}月 {r['rank']} 每類{r['per_fam']} 前{r['topn']} 放大{r['tiers']}: 500U → {r['end']:,.0f}U 回落 {r['mdd']:.0%} {r['yr']}")
