"""rotation_v1 (2026-11-17, Chi): two ideas taken from 熬鷹's interviews, as fixed rules written before looking.
Binance USDT perps, daily bars (UTC) built from the shared 1h data, crypto only, cost 0.1 % per side + funding.

Trend day (the trigger, same for both): ETH daily close above its highest close of the previous 20 days AND that day's
quote volume >= 1.5 x its 20-day average.  One trigger at most every 10 days.

1) 大行情才重倉: ETH long from the next day's open while the trend holds; out at the next open after a close below the
   lowest close of the previous 10 days (max 60 days).  Leverage 1 / 2 / 3.  Compared with always holding ETH (1x).
2) 輪動補漲: on each trigger, of the 50 most traded coins (30-day quote volume, excluding BTC, ETH) buy the 10 with the
   lowest 7-day return (laggards), equal weight, 1x, hold 5 days, stop -15 % per coin.  Compared with the 10 highest
   (leaders), all 50, and ETH alone over the same 5 days.
Front 2020-09..2022-12, back 2023-01..2026-10."""
import json

import numpy as np
import pandas as pd

from load import load

D = 86_400_000
SIDE = 0.001
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
PER = {"前段 2020-09～2022-12": (ts_("2020-09-01"), ts_("2023-01-01")), "後段 2023-01～2026-10": (ts_("2023-01-01"), ts_("2026-11-01"))}
NC = set(open("noncrypto.txt").read().split())

parts = []
for y in range(2020, 2027):
    k = load("klines_1h", f"{y}-01", f"{y}-12" if y < 2026 else "2026-10", columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"])
    k = k[~k.symbol.isin(NC)].drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    k["day"] = k.open_time // D * D
    parts.append(k.groupby(["symbol", "day"]).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
                                                   q=("quote_volume", "sum"), n=("close", "size")).reset_index())
    print(y, flush=True)
d = pd.concat(parts)
d = d[d.n >= 20]
C = d.pivot(index="day", columns="symbol", values="c"); O = d.pivot(index="day", columns="symbol", values="o")
L = d.pivot(index="day", columns="symbol", values="l"); Q = d.pivot(index="day", columns="symbol", values="q")
days = C.index.to_numpy()
fu = load("funding", "2020-01", "2026-10", columns=["symbol", "ts", "rate"])
fu["day"] = fu.ts // D * D
FD = fu.groupby(["symbol", "day"]).rate.sum().unstack(0).reindex(days).fillna(0)

# ---- trigger
e = C["ETHUSDT"].to_numpy(); eq_ = Q["ETHUSDT"].to_numpy()
hi20 = pd.Series(e).rolling(20).max().shift(1).to_numpy(); q20 = pd.Series(eq_).rolling(20).mean().shift(1).to_numpy()
TRIG = []; last = -99
for i in range(len(days) - 1):
    if e[i] > hi20[i] and eq_[i] >= 1.5 * q20[i] and i - last >= 10:
        TRIG.append(i); last = i
print("trend days", len(TRIG))


def per_of(t):
    for nm, (a, b) in PER.items():
        if a <= t < b: return nm


# ---- 1) trend leverage on ETH: daily equity path
def eth_path(lev, a, b, always=False):
    lo10 = pd.Series(e).rolling(10).min().shift(1).to_numpy(); o = O["ETHUSDT"].to_numpy(); f = FD["ETHUSDT"].to_numpy()
    eq = 500.; pos = False; entry_i = None; curve = []; trig = set(TRIG); held = 0
    for i in range(len(days) - 1):
        t = days[i]
        if t < a or t >= b: continue
        if always:
            if not pos: pos = True; eq *= 1 - SIDE * lev
        if pos:                                          # day i's return (open i -> open i+1)
            r = o[i + 1] / o[i] - 1
            eq *= max(0.0, 1 + lev * (r - f[i])); held += 1
            if not always and (e[i] < lo10[i] or i - entry_i >= 60):
                pos = False; eq *= 1 - SIDE * lev
        if not pos and not always and (i in trig):
            pos = True; entry_i = i + 1; eq *= 1 - SIDE * lev        # enter at open of i+1 (applied from next loop)
        curve.append(eq)
        if eq <= 0: break
    v = np.array(curve); dd = float((1 - v / np.maximum.accumulate(v)).max()) if len(v) else 0
    return round(float(v[-1])), round(dd, 3), held


# ---- 2) rotation basket per trigger
def basket(i, pick):
    t = days[i]; r7 = C.iloc[i] / C.iloc[i - 7] - 1
    vol30 = Q.iloc[i - 29:i + 1].sum(min_count=25).drop(["BTCUSDT", "ETHUSDT"], errors="ignore").dropna()
    uni = vol30.sort_values(ascending=False).index[:50]
    r = r7[uni].dropna()
    if pick == "laggards": coins = r.sort_values().index[:10]
    elif pick == "leaders": coins = r.sort_values().index[-10:]
    elif pick == "all": coins = r.index
    else: coins = ["ETHUSDT"]
    rets = []
    for s in coins:
        if i + 6 >= len(days) or pd.isna(O[s].iloc[i + 1]): continue
        en = O[s].iloc[i + 1]; out = None
        for j in range(i + 1, i + 6):
            if L[s].iloc[j] <= en * .85: out = en * .85; break
        if out is None: out = O[s].iloc[i + 6] if not pd.isna(O[s].iloc[i + 6]) else C[s].iloc[i + 5]
        fund = FD[s].iloc[i + 1:i + 6].sum() if s in FD else 0
        rets.append(out / en - 1 - 2 * SIDE - fund)
    return float(np.mean(rets)) if rets else 0.0, list(coins)


res = {"trend_days": [str(pd.Timestamp(days[i], unit="ms").date()) for i in TRIG]}
for nm, (a, b) in PER.items():
    print(f"\n=== {nm} ===  觸發 {sum(1 for i in TRIG if a <= days[i] < b)} 次")
    r_ = {}
    for lab, kw in (("一直拿 ETH 1倍", dict(lev=1, always=True)), ("趨勢日才進場 1倍", dict(lev=1)), ("趨勢日才進場 2倍", dict(lev=2)), ("趨勢日才進場 3倍", dict(lev=3))):
        v, dd, held = eth_path(a=a, b=b, **kw)
        print(f"  {lab}: 500U → {v:,}U，最大回落 {dd:.0%}，持有 {held} 天"); r_[lab] = (v, dd, held)
    for pick, lab in (("laggards", "輪動：落後 10 檔"), ("leaders", "對照：領漲 10 檔"), ("all", "對照：前 50 全買"), ("eth", "對照：只買 ETH")):
        rs = [(days[i], *basket(i, pick)) for i in TRIG if a <= days[i] < b]
        eq = 500.; curve = [500.]; yr = {}
        for t, r, _ in rs:
            eq *= 1 + r; curve.append(eq); yr.setdefault(pd.Timestamp(t, unit="ms").year, []).append(r)
        v = np.array(curve); dd = float((1 - v / np.maximum.accumulate(v)).max())
        win = np.mean([r > 0 for _, r, _ in rs]) if rs else 0
        by = {y: f"{len(x)}次 {np.mean(x):+.1%}" for y, x in sorted(yr.items())}
        print(f"  {lab}: 每次平均 {np.mean([r for _, r, _ in rs]):+.2%}，賺錢 {win:.0%}，500U 每次全押 → {eq:,.0f}U，最大回落 {dd:.0%}，各年 {by}")
        r_[lab] = (round(eq), round(dd, 3), by)
    res[nm] = r_
json.dump(res, open("rotation_v1.json", "w"), ensure_ascii=False, indent=1, default=str)
