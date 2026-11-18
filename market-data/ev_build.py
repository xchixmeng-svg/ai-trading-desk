"""ev_build.py (2026-11-18, Claude, for Chi): candidate events for the per-period "find a v12-like strategy" search.

Every hour, for every coin in the point-in-time top 100 (30-day quote volume, crypto only), these triggers are recorded
(entry = next 1h open):
  BRK  close breaks the L-hour range (L = 120 / 240 / 480), range width <= 30 %, stop = other side of the range
  SPK  4h move >= +8 %  -> long (follow) and short (fade)
  CRS  4h move <= -8 %  -> long (rebound, like 抄底) and short (follow)
  PB   trend pullback: daily EMA20 > EMA50 and close > EMA50, 1h close crosses back above the 4h EMA20 -> long (mirror short)
  EXT  7-day move >= +50 % and 1h close crosses below the 4h EMA20 -> short; <= -35 % and crosses above -> long
Non-BRK stop distance = 1.5 x daily ATR %, clipped to 2..25 %.  Cool-down 12h per coin / trigger / side (none for BRK, as v12).
Exits (R = result / initial stop distance, cost 0.4 % + funding):
  X1 trailing stop at the stop distance, max 30 days (v12 exit)
  X2 trailing stop at 2 x the stop distance, max 30 days
  X3 fixed stop, take profit +2R, max 5 days
  X4 fixed stop, after +1R lock +0.5R, max 72 h (抄底-like)
Features at the signal close (only data up to then): see FEAT below.  Output ev.npz."""
import datetime as dt
import numpy as np
import pandas as pd
from numba import njit

from load import load, load_meta

H = 3_600_000; H4 = 4 * H; D = 24 * H; M = 60_000
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
FIRST_M, LAST_M = "2021-03", "2026-10"
EV0, EV1 = ts_("2022-01-01"), ts_("2026-11-01")
COST = 0.004
NC = set(open("noncrypto.txt").read().split())

k = load("klines_1h", FIRST_M, LAST_M, columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"])
k = k[~k.symbol.isin(NC)].drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
B = {s: g[["open_time", "open", "high", "low", "close", "quote_volume"]].to_numpy("float64") for s, g in k.groupby("symbol")}
del k
print("coins", len(B), flush=True)

T0 = ts_(FIRST_M + "-01"); T1 = ts_("2026-11-01"); nh = (T1 - T0) // H
syms = sorted(B); Q = np.zeros((nh + 1, len(syms)), np.float32); P = np.zeros((nh + 1, len(syms)), np.float32)
for j, s in enumerate(syms):
    a = B[s]; ii = ((a[:, 0] - T0) // H).astype(int); ok = (ii >= 0) & (ii < nh)
    Q[ii[ok], j] = a[ok, 5]; P[ii[ok], j] = 1
cq = np.cumsum(Q, 0, dtype=np.float64); del Q
cq = np.vstack([np.zeros((1, len(syms))), cq])
cp = np.cumsum(P, 0, dtype=np.float32); del P
cp = np.vstack([np.zeros((1, len(syms)), np.float32), cp])
TOP = {}
for t in range(T0 + 720 * H, T1 + 1, H4):
    r = (t - T0) // H; v = cq[r] - cq[r - 720]; n = cp[r] - cp[r - 720]
    v = np.where(n >= 720, v, -1); top = np.argsort(-v)[:100]
    TOP[t] = {syms[j]: rk for rk, j in enumerate(top) if v[j] > 0}
del cq, cp
EVER = set().union(*[set(v) for t, v in TOP.items() if t >= EV0])
print("top100 ready; coins ever in top100:", len(EVER), flush=True)

fu = load("funding", FIRST_M, LAST_M, columns=["symbol", "ts", "rate"])
FU = {s: (g.ts.to_numpy(np.int64), np.cumsum(g.rate.to_numpy(float))) for s, g in fu.sort_values("ts").groupby("symbol")}
del fu


def fund(s, a_, b_):
    if s not in FU: return 0.0
    a, cs = FU[s]; i = np.searchsorted(a, a_, side="right") - 1; j = np.searchsorted(a, b_, side="right") - 1
    return float((cs[j] if j >= 0 else 0) - (cs[i] if i >= 0 else 0))


first = dict(zip(*load_meta("symbols.csv")[["symbol", "first_open_time"]].T.values))


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


def hammer(o_, h_, l_, c_, side):
    rg = h_ - l_
    if rg <= 0: return False
    body = abs(c_ - o_); up_ = h_ - max(o_, c_); lw = min(o_, c_) - l_
    if side < 0: up_, lw = lw, up_
    return lw >= 2 * body and up_ <= max(body, .15 * rg) and lw >= .5 * rg


@njit(cache=True)
def walk(o, h, l, c, i, side, Dd, trail, tp, lock, maxb, marks):
    """enter at o[i]; returns exit price, exit bar; daily marks (R) written into marks"""
    e = o[i]; sl = e * (1 - side * Dd); best = e; locked = False; m = 0; n = len(o)
    for j in range(i, min(n, i + maxb)):
        if side > 0:
            if l[j] <= sl: return min(o[j], sl), j
            if tp > 0 and h[j] >= e * (1 + tp * Dd): return max(o[j], e * (1 + tp * Dd)), j
            if h[j] > best: best = h[j]
            if trail > 0:
                ns = best * (1 - trail * Dd)
                if ns > sl: sl = ns
            if lock and not locked and h[j] >= e * (1 + Dd): locked = True; sl = max(sl, e * (1 + .5 * Dd))
        else:
            if h[j] >= sl: return max(o[j], sl), j
            if tp > 0 and l[j] <= e * (1 - tp * Dd): return min(o[j], e * (1 - tp * Dd)), j
            if l[j] < best: best = l[j]
            if trail > 0:
                ns = best * (1 + trail * Dd)
                if ns < sl: sl = ns
            if lock and not locked and l[j] <= e * (1 - Dd): locked = True; sl = min(sl, e * (1 - .5 * Dd))
        if (j - i + 1) % 24 == 0 and m < marks.shape[0]:
            marks[m] = (c[j] / e - 1) * side / Dd; m += 1
    j = min(n - 1, i + maxb - 1)
    return c[j], j


EXITS = [("X1", 1.0, 1.0, 0.0, False, 720), ("X2", 2.0, 1.0, 0.0, False, 720),
         ("X3", 1.0, 0.0, 2.0, False, 120), ("X4", 1.0, 0.0, 0.0, True, 72)]
FAM = {"BRK": 0, "SPK": 1, "CRS": 2, "PB": 3, "EXT": 4}
FEAT = ["fam", "side", "L", "w", "d", "vx", "r4", "rd", "hist_dir", "vdiv", "acc", "topacc", "taker", "oi24", "fund",
        "btc_up", "btc_r7", "r7", "r30", "dd30", "age", "atr", "ham", "vrank", "r4h", "rd_dir"]

btc = B["BTCUSDT"]; bct, _, _, bcd, _ = agg(btc, 24); bma = pd.Series(bcd).rolling(50).mean().to_numpy()
ROWS = []; OUT = []; MK = []; meta_s = []
for s in sorted(EVER):
    b = B.get(s)
    if b is None or len(b) < 1500: continue
    ot, o, h, l, c, q = b.T; n = len(b)
    c4t, h4, l4, c4, q4 = agg(b, 4); cdt, hd, ldd, cd, qd = agg(b, 24)
    if len(cd) < 60: continue
    r4a = rsi(c4); e20_4 = ema(c4, 20)
    md = ema(cd, 12) - ema(cd, 26); hist = md - ema(md, 9); rda = rsi(cd)
    e20d, e50d = ema(cd, 20), ema(cd, 50)
    atr = pd.Series((hd - ldd) / cd).rolling(14).mean().to_numpy()
    tclose = ot + H
    j4 = np.searchsorted(c4t, tclose, side="right") - 1; jd = np.searchsorted(cdt, tclose, side="right") - 1
    s4 = np.r_[np.zeros(3), np.convolve(q, np.ones(4), "valid")]       # s4[i] = q[i-3..i]
    cont = np.r_[np.zeros(4, bool), ot[4:] - ot[:-4] == 4 * H]
    r4h = np.r_[np.full(4, np.nan), c[4:] / c[:-4] - 1]
    ema4_now = np.where(j4 >= 0, e20_4[np.maximum(j4, 0)], np.nan)
    prev_c = np.r_[np.nan, c[:-1]]; prev_e = np.r_[np.nan, ema4_now[:-1]]
    ok_t = (tclose >= EV0) & (tclose < EV1) & (np.r_[ot[1:] - ot[:-1] == H, False]) & (np.arange(n) >= 1300)
    trig = []                                                            # (i, fam, side, L, w, stop)
    hs, ls_ = pd.Series(h), pd.Series(l)
    for L in (120, 240, 480):
        hi = hs.rolling(L).max().shift(1).to_numpy(); lo = ls_.rolling(L).min().shift(1).to_numpy()
        cl = np.r_[np.zeros(L, bool), ot[L:] - ot[:-L] == L * H]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = hi / lo - 1
            up = (c > hi) & (prev_c <= hi); dn = (c < lo) & (prev_c >= lo)
        for i in np.flatnonzero((up | dn) & (w <= .30) & cl & ok_t):
            sd = 1 if up[i] else -1
            trig.append((i, "BRK", sd, L, w[i], lo[i] if sd > 0 else hi[i]))
    jdc = np.maximum(jd, 0)
    with np.errstate(invalid="ignore"):
        r7 = np.where(jd >= 7, cd[jdc] / cd[np.maximum(jdc - 7, 0)] - 1, np.nan)
        trend_up = (e20d[jdc] > e50d[jdc]) & (c > e50d[jdc]); trend_dn = (e20d[jdc] < e50d[jdc]) & (c < e50d[jdc])
        cu = (prev_c < prev_e) & (c > ema4_now); cdn = (prev_c > prev_e) & (c < ema4_now)
        base_ok = ok_t & cont & (jd >= 60) & (j4 >= 60)
        for i in np.flatnonzero(base_ok & (r4h >= .08)): trig += [(i, "SPK", 1, 0, np.nan, None), (i, "SPK", -1, 0, np.nan, None)]
        for i in np.flatnonzero(base_ok & (r4h <= -.08)): trig += [(i, "CRS", 1, 0, np.nan, None), (i, "CRS", -1, 0, np.nan, None)]
        for i in np.flatnonzero(base_ok & trend_up & cu): trig.append((i, "PB", 1, 0, np.nan, None))
        for i in np.flatnonzero(base_ok & trend_dn & cdn): trig.append((i, "PB", -1, 0, np.nan, None))
        for i in np.flatnonzero(base_ok & (r7 >= .5) & cdn): trig.append((i, "EXT", -1, 0, np.nan, None))
        for i in np.flatnonzero(base_ok & (r7 <= -.35) & cu): trig.append((i, "EXT", 1, 0, np.nan, None))
    trig.sort(key=lambda z: z[0]); last = {}
    for i, fam, sd, L, w, stop in trig:
        t = int(tclose[i])
        key = (fam, sd, L)
        if fam != "BRK" and t - last.get(key, -10 ** 15) < 12 * H: continue
        top = TOP.get(t - t % H4)
        if top is None or s not in top: continue
        if jd[i] < 45 or j4[i] < 45: continue
        a4, ad = j4[i], jd[i]
        e = o[i + 1]
        if fam == "BRK":
            dd_ = (e - stop) / e * sd
            if dd_ <= .005 or dd_ > .25: continue
        else:
            if not np.isfinite(atr[ad]): continue
            dd_ = float(np.clip(1.5 * atr[ad], .02, .25))
        last[key] = t
        med = np.median(s4[max(4, i - 240):i:4]) if i > 244 else 0
        vx = s4[i] / med if med > 0 else 0
        a_, b_ = slice(ad - 19, ad + 1), slice(ad - 39, ad - 19)
        vdiv = (ldd[a_].min() < ldd[b_].min() if sd > 0 else hd[a_].max() > hd[b_].max()) and qd[a_].sum() < qd[b_].sum()
        bi = np.searchsorted(bct, t, side="right") - 1
        hi30 = hd[max(0, ad - 29):ad + 1].max()
        ham = any(hammer(o[jj], h[jj], l[jj], c[jj], sd) for jj in range(i - 11, i + 1))
        row = [FAM[fam], sd, L, w, dd_, vx, r4a[a4], rda[ad], float((hist[ad] > 0) == (sd > 0)), float(vdiv), np.nan, np.nan, np.nan,
               np.nan, np.nan, float(bcd[bi] > bma[bi]) if bi >= 50 else np.nan,
               bcd[bi] / bcd[bi - 7] - 1 if bi >= 7 else np.nan, r7[i] * sd, (cd[ad] / cd[ad - 30] - 1) * sd,
               c[i] / hi30 - 1, (t - first.get(s, ot[0])) / D, atr[ad], float(ham), top[s], r4h[i] * sd,
               float((rda[ad] > 50) == (sd > 0))]
        outs = []; mks = []
        for nm, mult, trail, tp, lk, mb in EXITS:
            Dd = min(dd_ * mult, .40)
            mk = np.full(30, np.nan, np.float32)
            x, jx = walk(o, h, l, c, i + 1, sd, Dd, trail, tp, lk, mb, mk)
            xt = int(ot[jx]) + H
            R = ((x / e - 1) * sd - COST - fund(s, t, xt) * sd) / Dd
            outs += [R, xt, Dd]; mks.append(mk)
        ROWS.append(row); OUT.append(outs); MK.append(np.stack(mks)); meta_s.append((s, t))
    print(s, len(ROWS), flush=True) if len(meta_s) and len(EVER) and sorted(EVER).index(s) % 50 == 0 else None

F = np.array(ROWS, np.float64); O = np.array(OUT, np.float64); MKA = np.stack(MK).astype(np.float32)
S = np.array([m[0] for m in meta_s]); T = np.array([m[1] for m in meta_s], np.int64)
print("events", len(F), {k_: int((F[:, 0] == v).sum()) for k_, v in FAM.items()}, flush=True)

# ---- metrics at the signal (all-account L/S, top-trader L/S, taker ratio, OI change 24h) and last funding
mon = np.array([dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m") for t in T])
iA, iT, iK, iO, iF = FEAT.index("acc"), FEAT.index("topacc"), FEAT.index("taker"), FEAT.index("oi24"), FEAT.index("fund")
prevG = {}
for m in sorted(set(mon)):
    idx = np.flatnonzero(mon == m)
    met = load("metrics_5m", m, m, symbols=sorted(set(S[idx])),
               columns=["symbol", "ts", "count_long_short_ratio", "count_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio", "sum_open_interest_value"])
    G = {s: (g.ts.to_numpy(np.int64), g.iloc[:, 2:].to_numpy(float)) for s, g in met.sort_values("ts").groupby("symbol")} if len(met) else {}
    for n_ in idx:
        a = G.get(S[n_]); t = T[n_]
        if a is not None:
            i = np.searchsorted(a[0], t, side="right") - 1
            if i >= 0 and t - a[0][i] <= 15 * M:
                v = a[1][i]; F[n_, iA], F[n_, iT], F[n_, iK] = v[0], v[1], v[2]
                j = np.searchsorted(a[0], t - D, side="right") - 1
                if j >= 0 and v[3] > 0 and a[1][j][3] > 0: F[n_, iO] = v[3] / a[1][j][3] - 1
        if S[n_] in FU:
            fa, cs = FU[S[n_]]; i = np.searchsorted(fa, t, side="right") - 1
            if i >= 1: F[n_, iF] = (cs[i] - cs[i - 1]) * F[n_, 1]
    del met, G
np.savez_compressed("ev.npz", F=F, O=O, MK=MKA, S=S, T=T, FEAT=np.array(FEAT))
print("saved ev.npz", F.shape, flush=True)
