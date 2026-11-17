"""Robustness test of the v12 playbook (盤整突破 A／B + 抄底 v11, rules in 策略_AB.md), 2026-11-17 (Claude).
usage: python robust_v12.py            (needs ~5 GB RAM; reads the shared dataset through load.py)

1) Unseen history: 2020-09 .. 2022-12 was never used to pick any rule. Rules are NOT changed.
   Binance has no long/short-ratio data before 2021-12, so B (needs it) cannot fire there -> only A (fail closed).
2) Neighbouring parameters: range 8 / 10 / 12 days x width 12 / 15 / 18 % x volume 1.5 / 2 / 2.5 x (27 sets).
3) Account exactly as 策略_AB.md section 4 (500U, A 5.75 %, B 2.25 %, DIP 7.5 %, margin <= 68 %, stop risk <= 40 %,
   DIP <= 3 open, <= 30 open, one per coin, stop opening new trades after a 60 % fall from the peak).
1h exits (stop checked before the best price is updated), cost 0.4 % per round trip + funding."""
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

from load import load, load_meta, months

H = 3_600_000; H4 = 4 * H; D = 24 * H; M = 60_000
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
FIRST_M, LAST_M = "2020-01", "2026-10"
NEW = (ts_("2020-05-01"), ts_("2023-01-01"))
OLD = (ts_("2023-01-01"), ts_("2026-11-01"))
COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.004
NC = set(open("noncrypto.txt").read().split())

k = load("klines_1h", FIRST_M, LAST_M, columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"])
k = k[~k.symbol.isin(NC)].drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
B = {s: g[["open_time", "open", "high", "low", "close", "quote_volume"]].to_numpy("float64") for s, g in k.groupby("symbol")}
del k
print("coins", len(B), flush=True)

# ---- point-in-time top 100 by 30-day quote volume (720 complete 1h bars ending at the 4h boundary)
T0 = ts_(FIRST_M + "-01"); T1 = ts_("2026-11-01")
nh = (T1 - T0) // H
syms = sorted(B); Q = np.zeros((nh + 1, len(syms)), np.float32); P = np.zeros((nh + 1, len(syms)), np.float32)
for j, s in enumerate(syms):
    a = B[s]; ii = ((a[:, 0] - T0) // H).astype(int); ok = (ii >= 0) & (ii < nh)
    Q[ii[ok], j] = a[ok, 5]; P[ii[ok], j] = 1
cq = np.cumsum(Q, 0, dtype=np.float64); cp = np.cumsum(P, 0, dtype=np.float64)
del Q, P
cq = np.vstack([np.zeros((1, len(syms))), cq]); cp = np.vstack([np.zeros((1, len(syms))), cp])
TOP = {}
for t in range(T0 + 720 * H, T1 + 1, H4):
    r = (t - T0) // H; v = cq[r] - cq[r - 720]; n = cp[r] - cp[r - 720]
    v = np.where(n >= 720, v, -1); top = np.argsort(-v)[:100]
    TOP[t] = {syms[j] for j in top if v[j] > 0}
del cq, cp
inTop = lambda s, t: s in TOP.get(t - t % H4, ())
print("top100 ready", flush=True)

fu = load("funding", FIRST_M, LAST_M, columns=["symbol", "ts", "rate"])
FU = {s: (g.ts.to_numpy(np.int64), g.rate.to_numpy(float)) for s, g in fu.sort_values("ts").groupby("symbol")}
del fu


def fund(s, a_, b_):
    if s not in FU: return 0.0
    a, v = FU[s]
    return float(v[np.searchsorted(a, a_, side="right"):np.searchsorted(a, b_, side="right")].sum())


# ---- indicators (exactly as breakout.py: EMA seeded with the first value, RSI with EMA(27))
def ema(x, n):
    return pd.Series(x).ewm(alpha=2 / (n + 1), adjust=False).mean().to_numpy()


def rsi(c):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    return 100 - 100 / (1 + ema(up, 27) / np.maximum(ema(dn, 27), 1e-12))


def agg(b, kh):
    ot = b[:, 0]; grp = (ot // (kh * H)).astype(np.int64)
    idx = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]]); ends = np.r_[idx[1:], len(b)]
    full = (ends - idx) == kh
    h = np.maximum.reduceat(b[:, 2], idx); l = np.minimum.reduceat(b[:, 3], idx); q = np.add.reduceat(b[:, 5], idx)
    ct = grp[idx] * kh * H + kh * H
    return ct[full], h[full], l[full], b[ends - 1, 4][full], q[full]



# ---- DIP v11 (same code as robust_v12.py / oos_2023.py), with the reason each crash was not traded
first = dict(zip(*load_meta("symbols.csv")[["symbol", "first_open_time"]].T.values))
btc = B["BTCUSDT"]


def hammer(o_, h_, l_, c_):
    rg = h_ - l_
    if rg <= 0: return False
    body = abs(c_ - o_); up_ = h_ - max(o_, c_); lw = min(o_, c_) - l_
    return lw >= 2 * body and up_ <= max(body, .15 * rg) and lw >= .5 * rg


DIPc = []; WHY = []
for s, b in B.items():
    ot = b[:, 0]
    st4 = np.flatnonzero(ot % H4 == 0)
    blocks = [j for j in st4 if j + 3 < len(b) and ot[j + 3] == ot[j] + 3 * H]
    if len(blocks) < 182: continue
    F = np.array([[ot[j], b[j, 1], b[j:j + 4, 2].max(), b[j:j + 4, 3].min(), b[j + 3, 4], b[j:j + 4, 5].sum()] for j in blocks])
    for n in range(181, len(F)):
        T = int(F[n, 0]) + H4
        if T < NEW[0] or T >= OLD[1] or not inTop(s, T): continue
        hi180 = F[n - 180:n, 2].max(); med = np.median(F[n - 30:n, 5]); pc = F[n - 1, 4]; cc = F[n, 4]
        if not (cc <= .5 * hi180 and med > 0 and F[n, 5] / med >= 6 and cc / pc - 1 <= -.10): continue
        if first.get(s, ot[0]) > T - 120 * D: WHY.append((T, s, "上市未滿120天")); continue
        stop0 = cc * .85; i = np.searchsorted(ot, T); why = "12小時內沒有錘子線"
        for j in range(i, min(len(b), i + 12)):
            if ot[j] >= T + 12 * H: break
            if b[j, 3] <= stop0: why = "錘子線前又跌15%"; break
            if hammer(*b[j, 1:5]):
                tb = int(ot[j]) + H
                bi = np.searchsorted(btc[:, 0], tb) - 1
                if btc[bi, 4] >= btc[bi::-24, 4][:51].mean(): why = "BTC在50日均線上"; break
                if j < 720 or b[j, 4] / b[j - 720, 4] - 1 <= -.527: why = "30天已跌超過52.7%"; break
                DIPc.append({"s": s, "t": tb, "j": int(j)}); why = None
                break
        if why: WHY.append((T, s, why))
mm = {}
for r in DIPc: mm.setdefault(dt.datetime.fromtimestamp(r["t"] / 1000, dt.UTC).strftime("%Y-%m"), []).append(r)
DIP = []; busy = {}
for m, rs in sorted(mm.items()):
    met = load("metrics_5m", m, m, symbols=sorted({r["s"] for r in rs}), columns=["symbol", "ts", "count_long_short_ratio"])
    G = {s: (g.ts.to_numpy(np.int64), g.count_long_short_ratio.to_numpy(float)) for s, g in met.sort_values("ts").groupby("symbol")} if len(met) else {}
    for r in sorted(rs, key=lambda z: z["t"]):
        if r["t"] < busy.get(r["s"], 0): continue        # one DIP per coin until it exits (as oos_2023.py)
        a = G.get(r["s"]); acc = np.nan
        if a is not None:
            i = np.searchsorted(a[0], r["t"], side="right") - 1
            if i >= 0 and r["t"] - a[0][i] <= 15 * M: acc = a[1][i]
        b = B[r["s"]]; j = r["j"]; e = b[j, 4]; sl = e * .85; lock = False; hourly = []; x = None
        for kk in range(j + 1, min(len(b), j + 73)):
            if b[kk, 3] <= sl: x = min(b[kk, 1], sl); xt = int(b[kk, 0]) + H; break
            if not lock and b[kk, 2] >= e * 1.15: lock = True; sl = e * 1.075
            hourly.append((int(b[kk, 0]) + H, (b[kk, 4] / e - 1) / .15))
        if x is None: kk = min(len(b) - 1, j + 72); x = b[kk, 4]; xt = int(b[kk, 0]) + H
        busy[r["s"]] = xt
        DIP.append({"s": r["s"], "t": r["t"], "xt": xt, "d": .15, "lev": 4, "R": (x / e - 1 - COST - fund(r["s"], r["t"], xt)) / .15,
                    "hourly": hourly, "kind": "DIP", "acc": float(acc)})
print("DIP trades", len(DIP), flush=True)



# ---- 3) account (策略_AB.md section 4)
def pool(T, a0, a1, risk={"A": .0575, "B": .0225, "DIP": .075}, kill=.60):
    T = [r for r in T if a0 <= r["t"] < a1]
    ev = sorted([(r["t"], 1, n) for n, r in enumerate(T)] + [(r["xt"], 0, n) for n, r in enumerate(T)])
    marks = {}
    for n, r in enumerate(T):
        for hh, v in r["hourly"]: marks.setdefault(hh, []).append((n, v))
    hs = sorted(marks); hi = 0; cash = 500.; opn = {}; peak = 500.; dd = 0.; dead = False; yearly = {}; taken = 0
    for tm, typ, n in ev:
        while hi < len(hs) and hs[hi] <= tm:
            eq = cash + sum(opn[m][0] * v for m, v in marks[hs[hi]] if m in opn)
            peak = max(peak, eq); dd = max(dd, 1 - eq / peak)
            if 1 - eq / peak >= kill: dead = True
            yearly[dt.datetime.fromtimestamp(hs[hi] / 1000, dt.UTC).year] = eq; hi += 1
        r = T[n]
        if typ == 0:
            if n in opn: cash += opn.pop(n)[0] * r["R"]; peak = max(peak, cash)
            continue
        if dead or len(opn) >= 30 or any(T[m]["s"] == r["s"] for m in opn): continue
        if r["kind"] == "DIP" and sum(T[m]["kind"] == "DIP" for m in opn) >= 3: continue
        rk = cash * risk[r["kind"]]
        lev = r.get("lev") or max(1, min(20, int(.8 / r["d"])))
        mg = rk / r["d"] / lev
        if sum(v[1] for v in opn.values()) + mg > .68 * cash: continue
        if sum(v[0] for v in opn.values()) + rk > .40 * cash: continue
        opn[n] = (rk, mg); taken += 1
    return cash, dd, yearly, taken



yr = lambda t: dt.datetime.fromtimestamp(t / 1000, dt.UTC).year
A = [r for r in DIP if r["acc"] <= 2.0]                 # the official rule (needs the ratio; none before 2021-12)
NA = DIP                                                 # same rule without the ratio
P_FRONT, P_BACK, P_ALL = NEW, OLD, (NEW[0], OLD[1])
print("\n=== 被擋掉的崩跌（前段 2020-05～2022 / 後段）===")
w = pd.DataFrame(WHY, columns=["T", "s", "why"]); w["段"] = np.where(w["T"] < NEW[1], "前段", "後段")
print(w.groupby(["段", "why"]).size().to_string())
x = pd.DataFrame([dict(t=r["t"], acc=r["acc"]) for r in DIP]); x["段"] = np.where(x.t < NEW[1], "前段", "後段")
x["多空比"] = np.where(x.acc.isna(), "沒有資料", np.where(x.acc <= 2.0, "<=2 通過", ">2 擋掉"))
print(x.groupby(["段", "多空比"]).size().to_string())
for y in sorted({yr(r["t"]) for r in DIP}):
    v = [r["R"] for r in NA if yr(r["t"]) == y]; v2 = [r["R"] for r in A if yr(r["t"]) == y]
    print(f"{y}: 不看多空比 {len(v)}筆 平均 {np.mean(v):+.2f}R 勝率 {np.mean(np.array(v) > 0):.0%} | 定案規則 {len(v2)}筆 " + (f"{np.mean(v2):+.2f}R" if v2 else ""))
for lab, T in (("定案規則（要多空比）", A), ("不看多空比", NA)):
    for rk in (.075, .02):
        for nm, per in (("前段 2020-05～2022", P_FRONT), ("後段 2023～2026-10", P_BACK), ("全部", P_ALL)):
            for kill in (.60, 9):
                c_, d_, y_, n_ = pool(T, *per, risk={"DIP": rk}, kill=kill)
                print(f"{lab} 風險{rk:.1%} {nm} {'60%停機' if kill < 1 else '不停機'}: 500U → {c_:,.0f}U，最大回落 {d_:.0%}，{n_} 筆，各年底 {({k: round(v) for k, v in y_.items()})}")
