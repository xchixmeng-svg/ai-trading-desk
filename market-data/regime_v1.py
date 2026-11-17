"""regime_v1 (2026-11-17, Chi): which strategy in which market. Built on robust_v12.py (same data, same strategy code).
Below: the original robust_v12 header.

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
NEW = (ts_("2020-09-01"), ts_("2023-01-01"))
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
LS, WS, VS = (240,), (0.15,), (2.0,)
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
    G = {s: (g.ts.to_numpy(np.int64), g.count_long_short_ratio.to_numpy(float)) for s, g in met.sort_values("ts").groupby("symbol")}
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
        if not (acc <= 2.0): continue                   # no ratio -> no trade (fail closed), as in oos_2023.py
        b = B[r["s"]]; j = r["j"]; e = b[j, 4]; sl = e * .85; lock = False; hourly = []; x = None
        for kk in range(j + 1, min(len(b), j + 73)):
            if b[kk, 3] <= sl: x = min(b[kk, 1], sl); xt = int(b[kk, 0]) + H; break
            if not lock and b[kk, 2] >= e * 1.15: lock = True; sl = e * 1.075
            hourly.append((int(b[kk, 0]) + H, (b[kk, 4] / e - 1) / .15))
        if x is None: kk = min(len(b) - 1, j + 72); x = b[kk, 4]; xt = int(b[kk, 0]) + H
        busy[r["s"]] = xt
        DIP.append({"s": r["s"], "t": r["t"], "xt": xt, "d": .15, "lev": 4, "R": (x / e - 1 - COST - fund(r["s"], r["t"], xt)) / .15,
                    "hourly": hourly, "kind": "DIP"})
print("DIP trades", len(DIP), flush=True)



# ---- 4) more existing strategies from the shared repo: 飆幣做空 (featgen.py) and turtle S1 20/10 (turtle_test.py)
FADE = []
for s, b in B.items():
    ot, o, h, l, c, q = b.T; n = len(b); last_f = -1e18
    if n < 1400: continue
    for i in range(1300, n - 1):
        t = int(ot[i]) + H
        if not (NEW[0] <= t < OLD[1]): continue
        if not (c[i] < l[i - 6:i].min() and c[i - 1] >= l[i - 7:i - 1].min()): continue
        if not (c[i - 72:i].max() / l[i - 72:i].min() - 1 >= .5 and t - last_f >= 24 * H and h[i - 72:i].argmax() >= 36): continue
        if not inTop(s, t): continue
        last_f = t
        e = o[i + 1]; stop = h[i - 12:i + 1].max(); d = (stop - e) / e
        if d <= .004 or d > .30: continue
        sl = stop; best = e; hourly = []; x = None
        for j in range(i + 1, min(n, i + 1 + 168)):
            if h[j] >= sl: x = max(o[j], sl); xt = int(ot[j]) + H; break
            best = min(best, l[j]); sl = min(sl, best * (1 + d))
            hourly.append((int(ot[j]) + H, (e / c[j] - 1) if False else (1 - c[j] / e) / d))
        if x is None: j = min(n - 1, i + 168); x = c[j]; xt = int(ot[j]) + H
        FADE.append({"s": s, "t": t, "xt": xt, "side": -1, "d": float(d), "R": float(((1 - x / e) - COST + fund(s, t, xt)) / d), "hourly": hourly})
print("fade", len(FADE), flush=True)
TUR = []
for s, b in B.items():
    ct, dh, dl, dc, _ = agg(b, 24)
    if len(dc) < 80: continue
    do_ = np.r_[np.nan, dc[:-1]]                                   # open of day ~ previous close (1h opens are contiguous)
    oi = {int(x): k for k, x in enumerate(ct)}
    tr = np.maximum(dh[1:], dc[:-1]) - np.minimum(dl[1:], dc[:-1]); atr = pd.Series(np.r_[np.nan, tr]).rolling(20).mean().to_numpy()
    i = 21
    while i < len(dc) - 1:
        t = int(ct[i])
        if t < NEW[0] or t >= OLD[1] or np.isnan(atr[i]) or not inTop(s, t) or ct[i + 1] != t + D: i += 1; continue
        hi = dc[i - 20:i].max(); lo = dc[i - 20:i].min()
        side = 1 if dc[i] > hi else -1 if dc[i] < lo else 0
        if not side: i += 1; continue
        e = dc[i]; dd_ = 2 * atr[i]; d = dd_ / e; sl = e - side * dd_; x = None; hourly = []
        if not (0.005 < d <= 0.5): i += 1; continue
        for j in range(i + 1, len(dc)):
            if (side > 0 and dl[j] <= sl) or (side < 0 and dh[j] >= sl):
                x = sl; break
            ex = dc[j - 10:j].min() if side > 0 else dc[j - 10:j].max()
            if j > i + 1 and ((side > 0 and dc[j] < ex) or (side < 0 and dc[j] > ex)): x = dc[j]; break
            hourly.append((int(ct[j]), (dc[j] / e - 1) * side / d))
        if x is None: j = len(dc) - 1; x = dc[j]
        xt = int(ct[j])
        TUR.append({"s": s, "t": t, "xt": xt, "side": side, "d": float(d), "R": float(((x / e - 1) * side - COST - fund(s, t, xt) * side) / d), "hourly": hourly})
        i = j + 1
print("turtle", len(TUR), flush=True)

# ---- 5) strategy pool (each one is a fixed, already existing rule; split by direction)
base = [r for r in C if r["L"] == 240 and r["w"] <= .15 and r["vx"] >= 2]
POOL = {
    "突破A 多": [r for r in base if r["A"] and r["side"] > 0], "突破A 空": [r for r in base if r["A"] and r["side"] < 0],
    "突破B 多": [r for r in base if r["B"] and not r["A"] and r["side"] > 0], "突破B 空": [r for r in base if r["B"] and not r["A"] and r["side"] < 0],
    "突破(全部) 多": [r for r in base if r["side"] > 0], "突破(全部) 空": [r for r in base if r["side"] < 0],
    "抄底 多": [dict(r, side=1) for r in DIP], "飆幣做空 空": FADE,
    "海龜20日 多": [r for r in TUR if r["side"] > 0], "海龜20日 空": [r for r in TUR if r["side"] < 0],
}
for k_, v in POOL.items():
    for r in v: r["strat"] = k_

# ---- 6) market state at each signal, from BTC only (known at that moment)
bt = B["BTCUSDT"]; bct, _, _, bdc, _ = agg(bt, 24)
ma50 = pd.Series(bdc).rolling(50).mean().to_numpy()
ret = np.r_[np.nan, np.diff(np.log(bdc))]; vol20 = pd.Series(ret).rolling(20).std().to_numpy()
volmed = pd.Series(vol20).rolling(365, min_periods=180).median().to_numpy()


def state(t):
    j = np.searchsorted(bct, t, side="right") - 1                   # last closed BTC daily bar
    if j < 0 or np.isnan(ma50[j]) or np.isnan(volmed[j]): return None
    return ("多頭" if bdc[j] > ma50[j] else "空頭") + ("・波動大" if vol20[j] > volmed[j] else "・波動小")


STATES = ["多頭・波動小", "多頭・波動大", "空頭・波動小", "空頭・波動大"]
for v in POOL.values():
    for r in v: r["st"] = state(r["t"])
days = {}
for t in range(NEW[0], OLD[1], D):
    days.setdefault((t < NEW[1], state(t)), 0); days[(t < NEW[1], state(t))] += 1
print("days per state (前段 / 後段):", {st: (days.get((True, st), 0), days.get((False, st), 0)) for st in STATES})


def stats(rs):
    v = np.array([r["R"] for r in rs]); return (len(v), float(v.mean()) if len(v) else float("nan"))


print("\n=== 每個策略在每種行情的每筆平均 R（前段 2020-09～2022-12 / 後段 2023-01～2026-10）===")
TAB = {}
for k_, v in POOL.items():
    line = f"{k_:10s}"
    for st in STATES:
        a = stats([r for r in v if r["st"] == st and r["t"] < NEW[1]]); b_ = stats([r for r in v if r["st"] == st and r["t"] >= OLD[0]])
        TAB[(k_, st)] = (a, b_)
        line += f" | {st} 前 {a[0]}筆 {a[1]:+.2f} 後 {b_[0]}筆 {b_[1]:+.2f}"
    print(line)

print("\n=== 時段（UTC 訊號時間；亞洲 0-8、歐洲 8-16、美國 16-24），全部期間每筆平均 R ===")
for k_, v in POOL.items():
    line = f"{k_:10s}"
    for nm, (h0, h1) in (("亞洲", (0, 8)), ("歐洲", (8, 16)), ("美國", (16, 24))):
        n_, m_ = stats([r for r in v if h0 <= (r["t"] // H) % 24 < h1])
        line += f" | {nm} {n_}筆 {m_:+.2f}"
    print(line)


# ---- 7) account: same risk for every trade so strategies compare fairly
def pool(T, a0, a1, risk=.02, kill=9.0):
    T = sorted([r for r in T if a0 <= r["t"] < a1], key=lambda r: r["t"])
    ev = sorted([(r["t"], 1, n) for n, r in enumerate(T)] + [(r["xt"], 0, n) for n, r in enumerate(T)])
    marks = {}
    for n, r in enumerate(T):
        for hh, v in r["hourly"]: marks.setdefault(hh, []).append((n, v))
    hs = sorted(marks); hi = 0; cash = 500.; opn = {}; peak = 500.; dd = 0.; yearly = {}; taken = 0; dead = False
    for tm, typ, n in ev:
        while hi < len(hs) and hs[hi] <= tm:
            eq = cash + sum(opn[m][0] * v for m, v in marks[hs[hi]] if m in opn)
            peak = max(peak, eq); dd = max(dd, 1 - eq / peak); dead = dead or (1 - eq / peak >= kill)
            yearly[dt.datetime.fromtimestamp(hs[hi] / 1000, dt.UTC).year] = eq; hi += 1
        r = T[n]
        if typ == 0:
            if n in opn: cash += opn.pop(n)[0] * r["R"]; peak = max(peak, cash)
            continue
        if dead or len(opn) >= 30 or any(T[m]["s"] == r["s"] for m in opn): continue
        rk = cash * risk; lev = 4 if r["strat"].startswith("抄底") else max(1, min(20, int(.8 / r["d"])))
        mg = rk / r["d"] / lev
        if sum(v[1] for v in opn.values()) + mg > .68 * cash or sum(v[0] for v in opn.values()) + rk > .40 * cash: continue
        opn[n] = (rk, mg); taken += 1
    yearly = {k: round(v) for k, v in yearly.items() if a0 <= ts_(f"{k}-01-01") + 0 or True}
    return cash, dd, yearly, taken


MIN_N = 15                                                         # pre-registered: a cell is used if its fit period has >= 15 trades and mean R > 0
res = {"cost": COST, "min_n": MIN_N}
for fit, test, fn, tn in ((NEW, OLD, "前段", "後段"), (OLD, NEW, "後段", "前段")):
    on = [(k_, st) for (k_, st), ab in TAB.items() for x in [ab[0] if fn == "前段" else ab[1]] if x[0] >= MIN_N and x[1] > 0]
    print(f"\n=== 用{fn}定對照表，拿{tn}考試 ===")
    for st in STATES:
        print(f"  {st} → {'、'.join(k_ for k_, s_ in on if s_ == st) or '不做'}")
    T = [r for k_, st in on for r in POOL[k_] if r["st"] == st]
    out = {}
    for nm, TT in [("看行情選策略", T), ("全部策略一直開", [r for v in POOL.values() for r in v])] + [(f"只做 {k_}", v) for k_, v in POOL.items()]:
        c_, d_, y_, n_ = pool(TT, *test)
        out[nm] = {"end": round(c_), "max_dd": round(d_, 3), "trades": n_, "year_end": y_}
        print(f"  {tn} {nm}: 500U → {c_:,.0f}U，最大回落 {d_:.0%}，{n_} 筆，各年底 {y_}")
    res[f"fit_{fn}_test_{tn}"] = {"table": on, "accounts": out}
json.dump(res, open(f"regime_v1_cost{COST}.json", "w"), ensure_ascii=False, indent=1)
