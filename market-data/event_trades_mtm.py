"""walkforward_v1 (2026-11-17, Chi): every quarter pick the strategy from the previous 12 months only, trade it for the
next 3 months, roll on.  Candidate trades are generated once with the code of robust_v12.py (below its original header).

Robustness test of the v12 playbook (盤整突破 A／B + 抄底 v11, rules in 策略_AB.md), 2026-11-17 (Claude).
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


# ---- 1) breakout candidates for every range length, loosest width / volume, filtered later
LS, WS, VS = (192, 240, 288), (0.12, 0.15, 0.18), (1.5, 2.0, 2.5)
C = []
for s, b in B.items():
    ot, o, h, l, c, q = b.T
    n = len(b)
    if n < 1400: continue
    s4 = np.convolve(q, np.ones(4), "full")[3:n + 3 - 3]           # s4[j] = q[j..j+3]
    s4 = np.r_[s4, np.zeros(n - len(s4))]
    hs = pd.Series(h); ls_ = pd.Series(l)
    ind = None
    for L in LS:
        hi = hs.rolling(L).max().shift(1).to_numpy(); lo = ls_.rolling(L).min().shift(1).to_numpy()
        cont = np.r_[np.zeros(L), ot[L:] - ot[:-L]] == L * H
        prev = np.r_[np.nan, c[:-1]]
        with np.errstate(invalid="ignore", divide="ignore"):
            up = (c > hi) & (prev <= hi); dn = (c < lo) & (prev >= lo); w = hi / lo - 1
        cand = np.flatnonzero((up | dn) & (w <= max(WS)) & cont)
        for i in cand:
            if i < 1300 or i + 1 >= n or ot[i + 1] != ot[i] + H: continue
            t = int(ot[i]) + H
            if not (NEW[0] <= t < OLD[1]) or not inTop(s, t): continue
            med = np.median(s4[i - L:i:4])
            vx = q[i - 3:i + 1].sum() / med if med > 0 else 0
            if vx < min(VS): continue
            side = 1 if up[i] else -1; e = o[i + 1]; stop = lo[i] if side > 0 else hi[i]
            d = (e - stop) / e * side
            if d <= 0.005 or d > 0.25: continue
            sl = stop; best = e; hourly = []; x = None
            for j in range(i + 1, min(n, i + 1 + 720)):
                if side > 0:
                    if l[j] <= sl: x = min(o[j], sl); xt = int(ot[j]) + H; break
                    best = max(best, h[j]); sl = max(sl, best * (1 - d))
                else:
                    if h[j] >= sl: x = max(o[j], sl); xt = int(ot[j]) + H; break
                    best = min(best, l[j]); sl = min(sl, best * (1 + d))
                hourly.append((int(ot[j]) + H, (c[j] / e - 1) * side / d))
            if x is None:
                j = min(n - 1, i + 720); x = c[j]; xt = int(ot[j]) + H
            R = ((x / e - 1) * side - COST - fund(s, t, xt) * side) / d
            if ind is None:                                         # 4h / daily indicators, once per coin
                c4t, _, _, c4, _ = agg(b, 4); cdt, hd, ldd, cd, qd = agg(b, 24)
                md = ema(cd, 12) - ema(cd, 26)
                ind = (c4t, rsi(c4), cdt, hd, ldd, qd, md - ema(md, 9), rsi(cd))
            c4t, r4, cdt, hd, ldd, qd, hist, rd = ind
            j4 = np.searchsorted(c4t, t, side="right") - 1; jd = np.searchsorted(cdt, t, side="right") - 1
            if j4 < 45 or jd < 45: continue
            a_, b_ = slice(jd - 19, jd + 1), slice(jd - 39, jd - 19)
            vdiv = (ldd[a_].min() < ldd[b_].min() if side > 0 else hd[a_].max() > hd[b_].max()) and qd[a_].sum() < qd[b_].sum()
            A = bool(vdiv and (r4[j4] > 70 if side > 0 else r4[j4] < 30))
            Bp = bool(((hist[jd] > 0) if side > 0 else (hist[jd] <= 0)) and
                      ((50 < rd[jd] <= 70) if side > 0 else (30 <= rd[jd] < 50)))
            C.append({"s": s, "t": t, "xt": xt, "side": side, "d": float(d), "R": float(R), "L": L, "w": float(w[i]),
                      "vx": float(vx), "A": A, "Bp": Bp, "hourly": hourly})
print("breakout candidates", len(C), flush=True)

# ---- all-account long/short ratio at each candidate (B needs it; none before 2021-12)
need = {}
for r in C:
    if r["Bp"]:
        need.setdefault(dt.datetime.fromtimestamp(r["t"] / 1000, dt.UTC).strftime("%Y-%m"), []).append(r)
for m, rs in sorted(need.items()):
    met = load("metrics_5m", m, m, symbols=sorted({r["s"] for r in rs}), columns=["symbol", "ts", "count_long_short_ratio"])
    G = {s: (g.ts.to_numpy(np.int64), g.count_long_short_ratio.to_numpy(float)) for s, g in met.sort_values("ts").groupby("symbol")} if len(met) else {}
    for r in rs:
        a = G.get(r["s"]); r["acc"] = None
        if a is None: continue
        i = np.searchsorted(a[0], r["t"], side="right") - 1
        if i >= 0 and r["t"] - a[0][i] <= 15 * M and not np.isnan(a[1][i]): r["acc"] = float(a[1][i])
for r in C:
    acc = r.get("acc")
    r["B"] = bool(r["Bp"] and acc is not None and ((acc <= 1.0) if r["side"] > 0 else (acc >= 2.0)))

# ---- 2) DIP v11 (same code as oos_2023.py)
first = dict(zip(*load_meta("symbols.csv")[["symbol", "first_open_time"]].T.values))
btc = B["BTCUSDT"]


def hammer(o_, h_, l_, c_):
    rg = h_ - l_
    if rg <= 0: return False
    body = abs(c_ - o_); up_ = h_ - max(o_, c_); lw = min(o_, c_) - l_
    return lw >= 2 * body and up_ <= max(body, .15 * rg) and lw >= .5 * rg


DIPc = []
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
        if first.get(s, ot[0]) > T - 120 * D: continue
        stop0 = cc * .85; i = np.searchsorted(ot, T)
        for j in range(i, min(len(b), i + 12)):
            if ot[j] >= T + 12 * H or b[j, 3] <= stop0: break
            if hammer(*b[j, 1:5]):
                tb = int(ot[j]) + H
                bi = np.searchsorted(btc[:, 0], tb) - 1
                if btc[bi, 4] >= btc[bi::-24, 4][:51].mean(): break
                if j < 720 or b[j, 4] / b[j - 720, 4] - 1 <= -.527: break
                DIPc.append({"s": s, "t": tb, "j": int(j)})
                break
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




import pickle
D_ = 86_400_000
def keep(r):
    out = {k: v for k, v in r.items() if k != "hourly"}
    out["dm"] = [(int(t), float(v)) for t, v in r.get("hourly", []) if t % D_ == 0]   # mark (in R) at each UTC day close
    return out
pickle.dump({"C": [keep(r) for r in C], "DIP": [keep(r) for r in DIP]}, open("event_trades_mtm.pkl", "wb"))
print("saved", len(C), len(DIP))
