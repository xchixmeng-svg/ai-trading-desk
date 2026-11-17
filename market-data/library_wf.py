"""library_wf (2026-11-17, Chi): a library of well-known strategy families, re-built every quarter from the previous 12
months only, traded for the next 3 months, rolled from 2021-01 to 2026-10.  Daily bars (UTC) from the shared 1h data,
crypto only (noncrypto.txt), cost 0.1 % per side on turnover, funding paid / received.

Library (each one a daily return stream; positions decided on the close, applied to the next day):
  trend      BTC 200-day MA long / flat; BTC 50-day MA long / short; ETH 50-day MA long / flat;
             time-series momentum on the top 20 (sign of the 20 / 60-day return, inverse-vol weights)
  cross-sec  momentum top 50 (long best 10 / short worst 10 by 30-day or 7-day return, weekly);
             1-day reversal top 50 (long worst 10 / short best 10, daily)
  carry      funding carry top 50 (short the 10 highest 3-day funding, long the 10 lowest, daily)
  bear       short the top 20 equally while BTC is under its 200-day MA
  hold       BTC buy and hold
  events     盤整突破 v12 (A or B) / all breakouts, 抄底 v11 (no ratio condition), 區間低點 45 days: 1 % risk per trade,
             P&L booked on the exit day (from event_trades.pkl built by event_trades.py)
Every quarter: each stream is scaled to 25 % yearly volatility using the previous 12 months; streams with a 12-month
Sharpe >= 0.5 and a positive 12-month return are kept, best 4 by Sharpe, equal weight; none -> cash.
Account: 500U, multiplier 1x (base) and 2x."""
import json
import pickle

import numpy as np
import pandas as pd

D = 86_400_000
SIDE = 0.001
NC = set(open("noncrypto.txt").read().split())
d = pd.read_parquet("daily.parquet")
d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
P = {x: d.pivot(index="day", columns="symbol", values=x) for x in "ohlcq"}
days = P["c"].index.to_numpy(); syms = list(P["c"].columns); S = len(syms)
C = P["c"].to_numpy(); Q = P["q"].to_numpy(); L_ = P["l"].to_numpy(); H_ = P["h"].to_numpy(); O = P["o"].to_numpy()
from load import load
fu = load("funding", "2020-01", "2026-10", columns=["symbol", "ts", "rate"]); fu["day"] = fu.ts // D * D
FD = fu.groupby(["symbol", "day"]).rate.sum().unstack(0).reindex(index=days, columns=syms).fillna(0).to_numpy()
R = np.vstack([np.full(S, np.nan), C[1:] / C[:-1] - 1]); R = np.nan_to_num(R, nan=0.0, posinf=0, neginf=0)
R = np.clip(R, -0.9, 3)
first = np.argmax(~np.isnan(C), axis=0)
q30 = pd.DataFrame(Q).rolling(30, min_periods=25).sum().to_numpy()


def top(i, n, exclude=()):
    v = q30[i].copy(); v[np.isnan(v) | (i - first < 60) | np.isnan(C[i])] = -1
    for e in exclude: v[syms.index(e)] = -1
    o = np.argsort(-v)[:n]; return o[v[o] > 0]


def run(W):
    """W[i] = weights decided at close of day i -> P&L on day i+1 (return, funding, turnover cost)."""
    out = np.zeros(len(days)); prev = np.zeros(S)
    for i in range(len(days) - 1):
        w = W[i]
        out[i + 1] = float(np.nansum(w * R[i + 1]) - np.nansum(w * FD[i + 1]) - SIDE * np.abs(w - prev).sum())
        prev = w
    return out


def ma(col, n): return pd.Series(C[:, col]).rolling(n).mean().to_numpy()


bi, ei = syms.index("BTCUSDT"), syms.index("ETHUSDT")
STR = {}
W = np.zeros((len(days), S)); m = ma(bi, 200); W[:, bi] = np.where(C[:, bi] > m, 1, 0); STR["BTC 站上200日線做多"] = run(W)
W = np.zeros((len(days), S)); m = ma(bi, 50); W[:, bi] = np.where(np.isnan(m), 0, np.where(C[:, bi] > m, 1, -1)); STR["BTC 50日線多空"] = run(W)
W = np.zeros((len(days), S)); m = ma(ei, 50); W[:, ei] = np.where(C[:, ei] > m, 1, 0); STR["ETH 站上50日線做多"] = run(W)
W = np.zeros((len(days), S)); W[:, bi] = 1; STR["BTC 買入持有"] = run(W)
vol = pd.DataFrame(R).rolling(30, min_periods=20).std().to_numpy()
for n in (20, 60):
    W = np.zeros((len(days), S))
    for i in range(80, len(days)):
        u = top(i, 20)
        if not len(u): continue
        sg = np.sign(C[i, u] / C[i - n, u] - 1); iv = 1 / np.maximum(vol[i, u], 1e-3); iv = np.nan_to_num(iv)
        if iv.sum() > 0: W[i, u] = np.nan_to_num(sg * iv / iv.sum())
    STR[f"時序動能 {n}日 前20大"] = run(W)
for n in (30, 7):
    W = np.zeros((len(days), S)); last = None
    for i in range(80, len(days)):
        if last is not None and i - last < 7: W[i] = W[i - 1]; continue
        u = top(i, 50); r = C[i, u] / C[i - n, u] - 1; ok = ~np.isnan(r); u, r = u[ok], r[ok]
        if len(u) < 20: continue
        o = np.argsort(r); W[i, u[o[-10:]]] = .05; W[i, u[o[:10]]] = -.05; last = i
    STR[f"強者恆強 {n}日 前50大"] = run(W)
W = np.zeros((len(days), S))
for i in range(80, len(days)):
    u = top(i, 50); r = R[i, u]; o = np.argsort(r)
    if len(u) >= 20: W[i, u[o[:10]]] = .05; W[i, u[o[-10:]]] = -.05
STR["短線反轉 1日 前50大"] = run(W)
W = np.zeros((len(days), S)); f3 = pd.DataFrame(FD).rolling(3).sum().to_numpy()
for i in range(80, len(days)):
    u = top(i, 50); f = f3[i, u]; o = np.argsort(f)
    if len(u) >= 20: W[i, u[o[:10]]] = .05; W[i, u[o[-10:]]] = -.05
STR["資金費套利 前50大"] = run(W)
W = np.zeros((len(days), S)); m = ma(bi, 200)
for i in range(80, len(days)):
    if not np.isnan(m[i]) and C[i, bi] < m[i]:
        u = top(i, 20, exclude=("BTCUSDT",)); W[i, u] = -1 / max(len(u), 1)
STR["熊市做空 前20大"] = run(W)

# ---- event strategies: 1 % risk per trade, booked on the exit day
ev = pickle.load(open("event_trades.pkl", "rb"))
dix = {int(t): k for k, t in enumerate(days)}
def book(trs):
    out = np.zeros(len(days))
    for r in trs:
        k = dix.get(int(r["xt"]) // D * D)
        if k is not None: out[k] += .01 * r["R"]
    return out
base = [r for r in ev["C"] if r["L"] == 240 and r["w"] <= .15 and r["vx"] >= 2]
STR["盤整突破 v12（A或B）"] = book([r for r in base if r["A"] or r["B"]])
STR["盤整突破 全部"] = book(base)
STR["抄底 v11"] = book(ev["DIP"])
# 區間低點 45 天 (range_v1 rules, top 20)
hiN = pd.DataFrame(H_).rolling(45, min_periods=45).max().to_numpy(); loN = pd.DataFrame(L_).rolling(45, min_periods=45).min().to_numpy()
rg = []
TOPS = [set(top(i, 20)) for i in range(len(days))]
for j in range(S):
    busy = -1
    for i in range(60, len(days) - 1):
        if i <= busy or j not in TOPS[i]: continue
        hi, lo, c = hiN[i, j], loN[i, j], C[i, j]
        if np.isnan(hi) or hi / lo - 1 > .30 or (c - lo) / (hi - lo) > .2: continue
        e = O[i + 1, j]; stop = lo * .97; tgt = lo + .8 * (hi - lo)
        if np.isnan(e) or e <= stop or e >= tgt: continue
        dist = (e - stop) / e; x = None
        for k in range(i + 1, min(len(days), i + 31)):
            if np.isnan(C[k, j]): x = C[k - 1, j]; break
            if L_[k, j] <= stop: x = min(O[k, j], stop); break
            if H_[k, j] >= tgt: x = max(O[k, j], tgt); break
        if x is None: x = C[k, j]
        rg.append({"xt": int(days[k]), "R": (x / e - 1 - 2 * SIDE - FD[i + 1:k + 1, j].sum()) / dist}); busy = k
STR["區間低點 45天"] = book(rg)
print("strategies", len(STR), flush=True)

# ---- 資金流 (money-flow) strategies, 2026-11-17 night; metrics only exist from 2021-12 (zero before)
import os
if os.environ.get("FLOW"):
    md = pd.read_parquet("metrics_daily.parquet"); md = md[md.symbol.isin(syms)].drop_duplicates(["symbol", "day"], keep="last")
    MP = {x: md.pivot(index="day", columns="symbol", values=x).reindex(index=days, columns=syms).to_numpy() for x in ("oi", "acc", "top", "taker")}
    OI = MP["oi"]
    W = np.zeros((len(days), S))                                   # 1) new listings: short every coin in its first 30 days
    for i in range(1, len(days)):
        nw = np.nonzero((first > 30) & (np.arange(len(days))[i] - first >= 1) & (np.arange(len(days))[i] - first <= 30) & ~np.isnan(C[i]) & (Q[i] > 0))[0]   # Q, not q30: q30 is empty in the first 25 days (bug found 11-17)
        if len(nw): W[i, nw] = -1 / max(len(nw), 5)
    STR["新幣上市做空"] = run(W)
    W = np.zeros((len(days), S)); oi1 = OI / np.vstack([np.full((1, S), np.nan), OI[:-1]]) - 1   # 2) liquidation flush -> bounce
    for i in range(80, len(days)):
        u = top(i, 50)
        for j in u:
            if R[i, j] <= -.08 and oi1[i, j] <= -.10: W[i:i + 5, j] += .1
    W = np.minimum(W, .3); g_ = W.sum(1, keepdims=True); W = np.where(g_ > 1, W / np.maximum(g_, 1e-9), W)
    STR["爆倉沖洗後做多"] = run(W)
    for nm_, key, sign in (("散戶多空比反向", "acc", -1), ("大戶多空比跟隨", "top", 1)):
        W = np.zeros((len(days), S)); X_ = MP[key]
        for i in range(80, len(days)):
            u = top(i, 50); v = X_[i, u]; ok = np.isfinite(v); u, v = u[ok], v[ok]
            if len(u) < 20: continue
            o = np.argsort(v); W[i, u[o[-10:]]] = .05 * sign; W[i, u[o[:10]]] = -.05 * sign
        STR[nm_] = run(W)
    W = np.zeros((len(days), S)); tk3 = pd.DataFrame(MP["taker"]).rolling(3, min_periods=2).mean().to_numpy()
    for i in range(80, len(days)):
        u = top(i, 50); v = tk3[i, u]; ok = np.isfinite(v); u, v = u[ok], v[ok]
        if len(u) < 20: continue
        o = np.argsort(v); W[i, u[o[-10:]]] = .05; W[i, u[o[:10]]] = -.05
    STR["主動買力強者做多"] = run(W)
    W = np.zeros((len(days), S)); oi7 = OI / np.vstack([np.full((7, S), np.nan), OI[:-7]]) - 1
    for i in range(80, len(days)):
        u = top(i, 50); r7_ = C[i, u] / C[i - 7, u] - 1; sel_ = u[(r7_ > 0) & (oi7[i, u] > .2)]
        if len(sel_): W[i, sel_] = 1 / max(len(sel_), 5)
    STR["上漲且持倉大增做多"] = run(W)
    print("flow strategies added", flush=True)
import os
for k_ in [x for x in os.environ.get("DROP", "").split(",") if x]: STR.pop(k_)
NM = list(STR); M = np.vstack([STR[k] for k in NM]).T          # days x strategies
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
yr = pd.to_datetime(days, unit="ms").year
print("\n=== 每個策略每年的 Sharpe（年化報酬／年化波動；> 1 很好，< 0 虧錢）===")
tab = {}
for k, nm in enumerate(NM):
    row = {}
    for y in range(2020, 2027):
        v = M[yr == y, k]
        row[y] = round(float(v.mean() / (v.std() + 1e-12) * np.sqrt(365)), 2) if v.std() > 0 else None
    tab[nm] = row; print(f"{nm:18s} " + " ".join(f"{y}:{'  -  ' if row[y] is None else f'{row[y]:+.2f}'}" for y in row))

QS = []; q = pd.Timestamp("2021-01-01", tz="UTC")
while q < pd.Timestamp("2026-11-01", tz="UTC"):
    QS.append(q); q = q + pd.DateOffset(months=3)
LOG = []; port = np.zeros(len(days)); always = np.zeros(len(days))
for q in QS:
    a = int(q.timestamp() * 1000); b = int(min(q + pd.DateOffset(months=3), pd.Timestamp("2026-11-01", tz="UTC")).timestamp() * 1000)
    t0 = int((q - pd.DateOffset(months=12)).timestamp() * 1000)
    tr = (days >= t0) & (days < a); te = (days >= a) & (days < b)
    sel = []; scale = {}
    for k, nm in enumerate(NM):
        v = M[tr, k]; sd = v.std()
        if sd <= 0 or (v != 0).sum() < 20: continue
        sh = v.mean() / sd * np.sqrt(365); scale[k] = min(5.0, .25 / (sd * np.sqrt(365)))
        if sh >= .5 and v.sum() > 0: sel.append((sh, k))
    sel = sorted(sel, reverse=True)[:4]
    if sel: port[te] = np.mean([M[te, k] * scale[k] for _, k in sel], axis=0)
    if scale: always[te] = np.mean([M[te, k] * s for k, s in scale.items()], axis=0)
    qr = float(np.prod(1 + port[te]) - 1)
    LOG.append({"季": f"{q:%Y-%m}", "選中": [f"{NM[k]}（{sh:.1f}）" for sh, k in sel] or ["空手"], "這季": f"{qr:+.1%}"})
    print(f"{q:%Y-%m} 選 {', '.join(LOG[-1]['選中'])} → 這季 {qr:+.1%}", flush=True)

A0 = ts_("2021-01-01"); m = days >= A0
def acct(r, mult):
    v = 500 * np.cumprod(1 + mult * r[m]); y = {}
    for t, e in zip(days[m], v): y[pd.Timestamp(t, unit="ms").year] = round(float(e))
    return round(float(v[-1])), round(float((1 - v / np.maximum.accumulate(v)).max()), 3), y
print("\n=== 帳戶 500U，2021-01～2026-10（每季只用前 12 個月挑）===")
res = {"sharpe_by_year": tab, "quarters": LOG}
for lab, r_, mult in (("滾動挑策略 1倍", port, 1), ("滾動挑策略 2倍", port, 2), ("全部策略一直開 1倍", always, 1),
                      ("BTC 買入持有", STR["BTC 買入持有"], 1)):
    e_, dd_, y_ = acct(r_, mult); res[lab] = {"end": e_, "max_dd": dd_, "year_end": y_}
    print(f"{lab}: 500U → {e_:,}U，最大回落 {dd_:.0%}，各年底 {y_}")
json.dump(res, open(f"library_wf{os.environ.get('TAG', '')}.json", "w"), ensure_ascii=False, indent=1, default=str)
