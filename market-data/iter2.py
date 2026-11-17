"""iter2 (2026-11-18): same marked-to-market streams (final_streams.npy) and the same monthly / bi-monthly re-pick,
but leverage set by the mix's own recent volatility (target 40-120 % a year, using the last 30 days only, cap 10x)
instead of by the distance from the account high, plus an optional brake (x0.5 below -20 %, x0.25 below -35 % from the
account high).  No stop inside the period (iteration 1 showed it hurts)."""
import json, itertools
import numpy as np, pandas as pd
M = np.load("final_streams.npy"); meta = json.load(open("final_streams.json")); NM, FAM = meta["names"], meta["fam"]
d = pd.read_parquet("daily.parquet"); NC = set(open("noncrypto.txt").read().split()); d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
days = np.sort(d.day.unique()); assert len(days) == len(M)
def picks(a, thr=.5, topn=6):
    t0 = int((pd.Timestamp(a, unit="ms", tz="UTC") - pd.DateOffset(months=12)).timestamp() * 1000); tr = (days >= t0) & (days < a)
    best = {}
    for k, nm in enumerate(NM):
        v = M[tr, k]; sd = v.std()
        if sd <= 0 or (v != 0).sum() < 20 or v.sum() <= 0: continue
        sh = v.mean() / sd * np.sqrt(365)
        if sh < thr: continue
        f = FAM[nm]
        if f not in best or sh > best[f][0]: best[f] = (sh, k, min(5.0, .25 / (sd * np.sqrt(365))))
    return sorted(best.values(), reverse=True)[:topn]
START = pd.Timestamp("2021-01-01", tz="UTC"); END = pd.Timestamp("2026-11-01", tz="UTC")
PL = {}
for P in (1, 2):
    q = START; lst = []
    while q < END:
        q2 = min(q + pd.DateOffset(months=P), END); a = int(q.timestamp() * 1000); lst.append((q, a, int(q2.timestamp() * 1000), picks(a))); q = q2
    PL[P] = lst
def sim(P, topn, tv, brake, cap=10.0):
    eq = 500.; pa = 500.; mdd = 0.; hist = []; yr = {}
    for q, a, b, sel in PL[P]:
        sel = sel[:topn]; te = np.nonzero((days >= a) & (days < b))[0]
        for i in te:
            base = np.mean([M[i, k] * sc for _, k, sc in sel]) if sel else 0.0
            rv = np.std(hist[-30:]) * np.sqrt(365) if len(hist) >= 10 else .25
            lev = min(cap, tv / max(rv, 1e-3))
            dd = 1 - eq / pa
            if brake and dd > .35: lev *= .25
            elif brake and dd > .20: lev *= .5
            eq *= 1 + max(lev * base, -.99); hist.append(base)
            pa = max(pa, eq); mdd = max(mdd, 1 - eq / pa); yr[pd.Timestamp(int(days[i]), unit="ms").year] = round(eq)
    return eq, mdd, yr
res = []
for P, topn, tv, brake in itertools.product((1, 2), (3, 4, 6), (.4, .6, .8, 1.0, 1.2), (False, True)):
    eq, mdd, yr = sim(P, topn, tv, brake); res.append({"P": P, "topn": topn, "tv": tv, "brake": brake, "end": eq, "mdd": mdd, "yr": yr})
json.dump(res, open("iter2_grid.json", "w"), ensure_ascii=False, default=str)
for r in sorted([r for r in res if r["mdd"] <= .6], key=lambda r: -r["end"])[:15]:
    print(f"每{r['P']}月 前{r['topn']} 目標波動{r['tv']:.0%} {'有煞車' if r['brake'] else '無煞車'}: 500U → {r['end']:,.0f}U 回落 {r['mdd']:.0%} {r['yr']}")
print("--- 鄰近值（每1月 前4 有煞車）---")
for r in res:
    if r["P"] == 1 and r["topn"] == 4 and r["brake"]: print(f"目標波動{r['tv']:.0%}: {r['end']:,.0f}U 回落 {r['mdd']:.0%}")
