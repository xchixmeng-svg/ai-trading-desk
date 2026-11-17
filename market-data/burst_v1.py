"""burst_v1 (2026-11-17, Chi's question): when is a 1h volume breakout on BTC / ETH worth switching to from the daily
strategy?  Spot 1h klines from data.binance.vision (spot_1h.parquet, built by burst_spot.py) + perp 1h from the shared
dataset.  Spot, no leverage, 250U per coin (500U total).  usage: python burst_v1.py [cost_per_side=0.001]

Signal S0 (closest rebuild of the 495-signal test; original script not in the repos): closed 1h spot bar
  close > highest high of the previous 24 bars, close / close 4 bars ago - 1 >= 3 %, quote volume >= 2 x the mean of
  the previous 24 bars, at most one signal per coin per 24 h.
Outcome: enter at the next bar's open; hit = +4 % reached before -2 % within 24 bars (both in one bar = miss).
Big swing: +10 % reached before -3 % within 72 bars (what a filter must not throw away).
Short trade: take profit +4 %, stop -2 % (stop first if both in one bar), else out after 24 bars.
Daily strategy (rebuilt from the description "MACD + Bollinger + rising 20-day MA"): long at the next day's open when
  the closed daily bar has MACD histogram > 0, close > 20-day MA (Bollinger middle) and the 20-day MA above its value 5
  days earlier; out at the next day's open when any of the three fails.  A short trade is only taken when the coin's
  daily strategy is flat; while a short trade is open the daily entry waits for the next day."""
import json
import sys

import numpy as np
import pandas as pd

from load import load

H = 3_600_000; D = 24 * H
SIDE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
FRONT = (ts_("2020-09-01"), ts_("2023-01-01")); BACK = (ts_("2023-01-01"), ts_("2026-11-01"))
spot = pd.read_parquet("spot_1h.parquet")
perp = load("klines_1h", "2020-01", "2026-10", symbols=["BTCUSDT", "ETHUSDT"], columns=["symbol", "open_time", "high", "close"])
perp = perp.drop_duplicates(["symbol", "open_time"])

S = {}
for s, g in spot.groupby("symbol"):
    g = g.sort_values("open_time").reset_index(drop=True)
    t = g.open_time.to_numpy(np.int64); o, h, l, c, q, v = (g[x].to_numpy(float) for x in ("open", "high", "low", "close", "quote_volume", "volume"))
    hi24 = pd.Series(h).rolling(24).max().shift(1).to_numpy()
    r4 = c / np.r_[[np.nan] * 4, c[:-4]] - 1
    q24 = pd.Series(q).rolling(24).mean().shift(1).to_numpy()
    hr = (t // H) % 24
    def same_hour(x):                                   # median of the same UTC hour over the previous 30 days
        out = np.full(len(x), np.nan)
        for k in range(24):
            ix = np.flatnonzero(hr == k)
            out[ix] = pd.Series(x[ix]).rolling(30).median().shift(1).to_numpy()
        return out
    qh, vh = same_hour(q), same_hour(v)
    p = perp[perp.symbol == s].sort_values("open_time")
    ph = pd.Series(p.high.to_numpy()).rolling(24).max().shift(1).to_numpy()
    pok = dict(zip(p.open_time.to_numpy(np.int64), p.close.to_numpy() > ph))
    perp_ok = np.array([bool(pok.get(int(x), False)) for x in t])
    S[s] = dict(t=t, o=o, h=h, l=l, c=c, hi24=hi24, brk=(c > hi24) & (r4 >= .03), vol_orig=q >= 2 * q24,
                vol_hq=q >= 2 * qh, vol_hv=v >= 2 * vh, perp=perp_ok)


def first_hit(s, k, up, dn, n):
    """from entry at open[k]: 1 if +up before -dn within n bars"""
    d = S[s]; e = d["o"][k]
    for j in range(k, min(len(d["o"]), k + n)):
        if d["l"][j] <= e * (1 - dn): return 0
        if d["h"][j] >= e * (1 + up): return 1
    return 0


def signals(vol="vol_orig", perp=False):
    out = []
    for s, d in S.items():
        m = d["brk"] & d[vol]
        if perp: m = m & d["perp"]
        last = -10 ** 18
        for i in np.flatnonzero(m):
            if d["t"][i] - last < D or i + 3 >= len(d["t"]): continue
            last = d["t"][i]
            out.append({"s": s, "i": int(i), "t": int(d["t"][i]) + H, "lvl": float(d["hi24"][i])})
    return out


BIG = {(r["s"], r["t"]) for r in signals() if first_hit(r["s"], r["i"] + 1, .10, .03, 72)}


def short_trades(sig, confirm=False, fail_exit=False):
    """entry: next open, or (confirm) the open after the next bar if that bar closed above the breakout level"""
    T = []
    for r in sig:
        d = S[r["s"]]; k = r["i"] + 1
        if confirm:
            if d["c"][k] <= r["lvl"]: continue
            k += 1
        e = d["o"][k]; x = None
        for j in range(k, min(len(d["o"]), k + 24)):
            if d["l"][j] <= e * .98: x = min(d["o"][j], e * .98); xt = d["t"][j] + H; break
            if d["h"][j] >= e * 1.04: x = max(d["o"][j], e * 1.04); xt = d["t"][j] + H; break
            if fail_exit and d["c"][j] < r["lvl"] and j + 1 < len(d["o"]): x = d["o"][j + 1]; xt = d["t"][j + 1]; break
        if x is None: j = min(len(d["o"]) - 1, k + 23); x = d["c"][j]; xt = d["t"][j] + H
        T.append(dict(r, et=int(d["t"][k]), e=float(e), xt=int(xt), x=float(x), hit=first_hit(r["s"], k, .04, .02, 24)))
    return T


# ---- daily strategy positions per coin (decided at the daily close, traded at the next day's first open)
DAY = {}
for s, d in S.items():
    t = d["t"]; day = t // D
    g = pd.DataFrame({"day": day, "c": d["c"], "n": 1}).groupby("day").agg(c=("c", "last"), n=("n", "sum"))
    c = g.c.to_numpy(); ma = pd.Series(c).rolling(20).mean().to_numpy()
    ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
    m = ema(c, 12) - ema(c, 26); hist = m - ema(m, 9)
    want = (hist > 0) & (c > ma) & (ma > np.r_[[np.nan] * 5, ma[:-5]])
    DAY[s] = {int(dd) + 1: bool(w) for dd, w in zip(g.index.to_numpy(), want)}   # position wanted on day dd+1


def account(a0, a1, T=(), daily_fail_exit=False, side=SIDE):
    """250U per coin; hourly loop; returns end value, max drawdown, per-year end"""
    T = sorted([r for r in T if a0 <= r["t"] < a1], key=lambda r: r["et"])
    eq_tot = {}
    for s, d in S.items():
        t = d["t"]; o, c = d["o"], d["c"]
        cash = 250.; qty = 0.; mode = None; short = None; byet = {}
        for r in T:
            if r["s"] == s: byet.setdefault(r["et"], r)
        fails = {}
        if daily_fail_exit:                               # a short signal that closes back under its level within 24 h
            for r in T:
                if r["s"] != s: continue
                for j in range(r["i"] + 1, min(len(c), r["i"] + 25)):
                    if c[j] < r["lvl"]: fails[int(t[j + 1]) if j + 1 < len(t) else None] = True; break
        blocked_day = None
        for j in range(len(t)):
            tm = int(t[j])
            if tm < a0 or tm >= a1 + 3 * D: continue
            # short trade exit
            if mode == "short" and tm >= short["xt"]:
                cash = qty * short["x"] * (1 - side); qty = 0.; mode = None; short = None
            # daily decision at the first bar of each UTC day
            if tm % D == 0 and tm < a1:
                want = DAY[s].get(tm // D, False)
                if mode == "daily" and not want:
                    cash = qty * o[j] * (1 - side); qty = 0.; mode = None
                elif mode is None and want:
                    qty = cash * (1 - side) / o[j]; cash = 0.; mode = "daily"
            if mode == "daily" and fails.get(tm):
                cash = qty * o[j] * (1 - side); qty = 0.; mode = None; blocked_day = tm // D
            if mode is None and tm in byet:
                r = byet[tm]; qty = cash * (1 - side) / o[j]; cash = 0.; mode = "short"; short = r
            eq_tot[tm] = eq_tot.get(tm, 0) + cash + qty * c[j]
    ts = sorted(eq_tot); v = np.array([eq_tot[x] for x in ts])
    dd = float((1 - v / np.maximum.accumulate(v)).max())
    yr = {}
    for x, val in zip(ts, v): yr[pd.Timestamp(x, unit="ms").year] = round(float(val))
    return round(float(v[-1])), round(dd, 3), yr


def report(nm, sig_rows, T, per):
    a0, a1 = per
    s_ = [r for r in sig_rows if a0 <= r["t"] < a1]; T_ = [r for r in T if a0 <= r["t"] < a1]
    big_all = [x for x in BIG if a0 <= x[1] < a1]
    caught = sum(1 for x in big_all if any(r["s"] == x[0] and abs(r["t"] - x[1]) <= 6 * H for r in T_))
    hit = np.mean([r["hit"] for r in T_]) if T_ else float("nan")
    by = {}
    for r in T_: by.setdefault(pd.Timestamp(r["t"], unit="ms").year, []).append(r["hit"])
    pnl = [r["x"] / r["e"] - 1 - 2 * SIDE for r in T_]
    return {"name": nm, "n": len(T_), "hit": round(float(hit), 3), "avg_ret": round(float(np.mean(pnl)), 4) if pnl else None, "big_caught": caught, "big_all": len(big_all),
            "per_year": {y: (len(v), round(float(np.mean(v)), 3)) for y, v in sorted(by.items())}}


VARS = {}
base_sig = signals()
VARS["S0 原版"] = (base_sig, short_trades(base_sig), {})
for nm, vol in (("V1a 量=同時段 USDT", "vol_hq"), ("V1b 量=同時段 幣量", "vol_hv")):
    sg = signals(vol); VARS[nm] = (sg, short_trades(sg), {})
sg = signals(perp=True); VARS["V2 原版+永續也突破"] = (sg, short_trades(sg), {})
VARS["V3 原版+等1根站穩"] = (base_sig, short_trades(base_sig, confirm=True), {})
VARS["V4 原版+收回就出場"] = (base_sig, short_trades(base_sig, fail_exit=True), {})
VARS["V5 原版+日線倉遇假突破也出"] = (base_sig, short_trades(base_sig), {"daily_fail_exit": True})

res = {"cost_per_side": SIDE}
for pn, per in (("前段 2020-09～2022-12", FRONT), ("後段 2023-01～2026-10", BACK)):
    d0 = account(*per)
    print(f"\n=== {pn}（每邊成本 {SIDE:.2%}）===\n只做日線: 500U → {d0[0]}U，最大回落 {d0[1]:.0%}，各年底 {d0[2]}")
    res[pn] = {"daily_only": d0, "variants": {}}
    for nm, (sg, T, kw) in VARS.items():
        st = report(nm, sg, T, per); acc = account(*per, T=T, **kw)
        st.update(account=acc[0], max_dd=acc[1], year_end=acc[2]); res[pn]["variants"][nm] = st
        print(f"{nm}: {st['n']} 筆，先漲4% {st['hit']:.1%}，每筆平均 {st['avg_ret']:+.2%}，大波段抓到 {st['big_caught']}/{st['big_all']}，日線+短線 500U → {acc[0]}U，回落 {acc[1]:.0%}，各年底 {acc[2]}，每年(筆,先漲4%) {st['per_year']}")
json.dump(res, open(f"burst_v1_side{SIDE}.json", "w"), ensure_ascii=False, indent=1, default=str)
