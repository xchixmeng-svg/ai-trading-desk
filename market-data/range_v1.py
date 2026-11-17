"""range_v1 (2026-11-17, Chi): only well-known coins, long range, buy at the bottom, never chase the top.
Rules written before looking:
  universe   : top 20 by 30-day quote volume that day, crypto only, at least 365 days of data (coins already trading
               when the data starts in 2020-01 count as old)
  range      : the last N daily bars including today: high / low - 1 <= 30 %
  low zone   : today's close in the bottom Z of the range ((close - low) / (high - low) <= Z)
  entry      : next day's open, long only; stop = range low - 3 %; target = low + 80 % of the range; max 30 days
               (stop checked first if both are hit on one day); skip if the open is already past stop or target
  account    : 500U, risk 2 % of cash per trade, at most 10 open, total size <= 3 x cash, one trade per coin at a time
Base N = 30, Z = 20 %; neighbours N 20 / 30 / 45 x Z 15 / 20 / 25 %. Cost 0.1 % per side + funding (argument 2 doubles).
Front 2020-09..2022-12, back 2023-01..2026-10.  usage: python range_v1.py [cost_per_side]"""
import json
import os
import sys

import numpy as np
import pandas as pd

from load import load

D = 86_400_000
SIDE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
FRONT = (ts_("2020-09-01"), ts_("2023-01-01")); BACK = (ts_("2023-01-01"), ts_("2026-11-01"))
NC = set(open("noncrypto.txt").read().split())

if not os.path.exists("daily.parquet"):
    parts = []
    for y in range(2020, 2027):
        k = load("klines_1h", f"{y}-01", f"{y}-12" if y < 2026 else "2026-10", columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"])
        k = k.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"]); k["day"] = k.open_time // D * D
        parts.append(k.groupby(["symbol", "day"]).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
                                                       q=("quote_volume", "sum"), n=("close", "size")).reset_index())
    pd.concat(parts).to_parquet("daily.parquet")
d = pd.read_parquet("daily.parquet")
d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
P = {x: d.pivot(index="day", columns="symbol", values=x) for x in "ohlcq"}
days = P["c"].index.to_numpy(); syms = P["c"].columns
fu = load("funding", "2020-01", "2026-10", columns=["symbol", "ts", "rate"]); fu["day"] = fu.ts // D * D
FD = fu.groupby(["symbol", "day"]).rate.sum().unstack(0).reindex(index=days, columns=syms).fillna(0).to_numpy()
O, H, L, C, Q = (P[x].to_numpy() for x in "ohlcq")
first = np.argmax(~np.isnan(C), axis=0)
old = (first == 0)
q30 = pd.DataFrame(Q).rolling(30, min_periods=25).sum().to_numpy()
TOP = np.zeros_like(C, dtype=bool)
for i in range(len(days)):
    v = q30[i].copy(); age_ok = old | (i - first >= 365); v[~age_ok | np.isnan(v)] = -1
    top = np.argsort(-v)[:20]; TOP[i, top[v[top] > 0]] = True


def trades(N, Z):
    hiN = pd.DataFrame(H).rolling(N, min_periods=N).max().to_numpy(); loN = pd.DataFrame(L).rolling(N, min_periods=N).min().to_numpy()
    out = []
    for j in range(C.shape[1]):
        busy = -1
        for i in np.flatnonzero(TOP[:, j]):
            if i <= busy or i + 1 >= len(days) or not (FRONT[0] <= days[i] < BACK[1]): continue
            hi, lo, c = hiN[i, j], loN[i, j], C[i, j]
            if np.isnan(hi) or hi / lo - 1 > .30 or (c - lo) / (hi - lo) > Z: continue
            e = O[i + 1, j]; stop = lo * .97; tgt = lo + .8 * (hi - lo)
            if np.isnan(e) or e <= stop or e >= tgt: continue
            dist = (e - stop) / e; x = None; marks = []
            for k in range(i + 1, min(len(days), i + 31)):
                if np.isnan(C[k, j]): x = C[k - 1, j]; break
                if L[k, j] <= stop: x = min(O[k, j], stop); break
                if H[k, j] >= tgt: x = max(O[k, j], tgt); break
                marks.append((k, (C[k, j] / e - 1) / dist))
            if x is None: x = C[k, j]
            fund = FD[i + 1:k + 1, j].sum()
            R = (x / e - 1 - 2 * SIDE - fund) / dist
            out.append({"s": syms[j], "i": i, "xk": k, "t": int(days[i]), "R": float(R), "d": float(dist), "marks": marks,
                        "ret": float(x / e - 1)})
            busy = k
    return out


def account(T, a0, a1, risk=.02):
    T = sorted([r for r in T if a0 <= r["t"] < a1], key=lambda r: r["i"])
    by_in = {}; by_out = {}
    for n, r in enumerate(T): by_in.setdefault(r["i"] + 1, []).append(n); by_out.setdefault(r["xk"], []).append(n)
    cash = 500.; opn = {}; curve = []; yr = {}; taken = 0
    i0 = np.searchsorted(days, a0); i1 = np.searchsorted(days, a1)
    mk = {n: dict(r["marks"]) for n, r in enumerate(T)}
    for k in range(i0, min(len(days), i1 + 31)):
        for n in by_in.get(k, []):
            r = T[n]
            if len(opn) >= 10 or any(T[m]["s"] == r["s"] for m in opn): continue
            rk = cash * risk; size = rk / r["d"]
            if size > cash or sum(v[1] for v in opn.values()) + size > 3 * cash: continue
            opn[n] = (rk, size); taken += 1
        eq = cash + sum(v[0] * mk[n].get(k, 0) for n, v in opn.items())
        for n in by_out.get(k, []):
            if n in opn: cash += opn.pop(n)[0] * T[n]["R"]
        curve.append(eq); yr[pd.Timestamp(days[k], unit="ms").year] = round(eq)
    v = np.array(curve)
    return round(cash), round(float((1 - v / np.maximum.accumulate(v)).max()), 3), yr, taken


def hold_btc(a0, a1):
    j = list(syms).index("BTCUSDT"); i0 = np.searchsorted(days, a0); i1 = np.searchsorted(days, a1) - 1
    v = C[i0:i1 + 1, j] / O[i0, j] * 500
    return round(float(v[-1])), round(float((1 - v / np.maximum.accumulate(v)).max()), 3)


res = {"cost_per_side": SIDE}
print(f"成本每邊 {SIDE:.1%}")
for N in tuple(int(x) for x in os.environ.get("NS", "20,30,45").split(",")):
    for Z in (.15, .20, .25):
        T = trades(N, Z); row = {}
        line = f"{N}天 低位{Z:.0%}"
        for pn, per in (("前段", FRONT), ("後段", BACK)):
            tt = [r for r in T if per[0] <= r["t"] < per[1]]
            R = np.array([r["R"] for r in tt]); end, dd, yr, n = account(T, *per)
            row[pn] = {"n": len(tt), "avgR": round(float(R.mean()), 3) if len(R) else None, "win": round(float((R > 0).mean()), 3) if len(R) else None,
                       "account": end, "max_dd": dd, "taken": n, "year_end": yr}
            line += f" | {pn} {len(tt)}筆 平均{R.mean():+.2f}R 勝率{(R > 0).mean():.0%} 500U→{end:,}U 回落{dd:.0%}"
        if N == 30 and Z == .20:
            by = {}
            for r in T: by.setdefault(pd.Timestamp(r["t"], unit="ms").year, []).append(r["R"])
            row["per_year"] = {y: (len(x), round(float(np.mean(x)), 3), round(float(np.mean(np.array(x) > 0)), 3)) for y, x in sorted(by.items())}
            line += "\n   基準逐年(筆,平均R,勝率): " + str(row["per_year"])
            line += f"\n   帳戶各年底 前段 {row['前段']['year_end']} 後段 {row['後段']['year_end']}"
        res[f"{N}_{Z}"] = row; print(line, flush=True)
for pn, per in (("前段", FRONT), ("後段", BACK)):
    res["hold_btc_" + pn] = hold_btc(*per); print(f"一直拿 BTC {pn}: 500U → {res['hold_btc_' + pn][0]:,}U，回落 {res['hold_btc_' + pn][1]:.0%}")
json.dump(res, open(f"range_v1_side{SIDE}_{os.environ.get('NS', 'base')}.json", "w"), ensure_ascii=False, indent=1, default=str)
