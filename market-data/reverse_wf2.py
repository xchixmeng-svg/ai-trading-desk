"""reverse_wf2 (2026-11-17, Chi): reverse_wf.py with the rebuild every 1 / 2 / 3 months and an optional risk control
(at most 3 open trades per direction; after a 20 % fall from the peak inside a period, no new trades until the next
rebuild).  The search itself is unchanged (always the previous 12 months: 9 to find, 3 to check).

reverse_wf (2026-11-17, Chi): every quarter, work backwards from the moves that actually paid in the previous 12
months, find what they looked like the day before, trade that for the next 3 months, then do it again.
Only data up to the start of each quarter is used.  Daily bars (UTC) from the shared 1h data, crypto only.

Moves   : three kinds of trade, entered at the next day's open, long or short:
          quick (+10 % / -5 %, 3 days), swing (+20 % / -8 %, 7 days), big (+40 % / -15 %, 14 days);
          stop first if both are touched on one day; R = result / stop distance, minus 0.2 % cost and funding.
Clues   : 20 things visible at the close (returns 1/3/7/30 d, place in the 30-day range, 10/30-day range width, volume
          vs 20-day average, 20-day volatility, RSI, MACD, distance to 20/50-day MA, BTC 7-day return, BTC vs 50-day MA,
          3-day funding, 7-day return rank among the top 100, share of coins above their 20-day MA, coin age,
          fall from the 90-day high); each cut at the 20/40/60/80 % levels of the training months.
Rule    : one or two clues (different ones) + one kind of move + long or short, on the top 100 by 30-day volume.
Search  : first 9 of the 12 months: >= 30 trades on >= 15 different days, mean R >= 0.2, ranked by a t-value that
          counts each day once; then the last 3 months must also be positive (>= 5 trades).  The best 2 rules that
          pass are used for the next quarter, unchanged.  None pass -> no trades.
Account : 500U, 2 % of cash at risk per trade, at most 10 open, one per coin; marked to market daily."""
import itertools
import json

import numpy as np
import pandas as pd

D = 86_400_000
SIDE = 0.001
NC = set(open("noncrypto.txt").read().split())
d = pd.read_parquet("daily.parquet")
d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
P = {x: d.pivot(index="day", columns="symbol", values=x) for x in "ohlcq"}
days = P["c"].index.to_numpy(); syms = list(P["c"].columns); T_, S = len(days), len(syms)
O, H, L, C, Q = (P[x].to_numpy() for x in "ohlcq")
from load import load
fu = load("funding", "2020-01", "2026-10", columns=["symbol", "ts", "rate"]); fu["day"] = fu.ts // D * D
FD = fu.groupby(["symbol", "day"]).rate.sum().unstack(0).reindex(index=days, columns=syms).fillna(0).to_numpy()
CF = np.vstack([np.zeros(S), np.cumsum(FD, 0)])                 # CF[k] = funding summed over days < k
roll = lambda X, n, f, mp=None: getattr(pd.DataFrame(X).rolling(n, min_periods=mp or n), f)().to_numpy()
sh = lambda X, k: np.vstack([np.full((k, S), np.nan), X[:-k]])
first = np.argmax(~np.isnan(C), axis=0); age = np.arange(T_)[:, None] - first[None, :]
q30 = roll(Q, 30, "sum", 25)
U = np.zeros((T_, S), bool)
for i in range(T_):
    v = q30[i].copy(); v[np.isnan(v) | (age[i] < 30) | np.isnan(C[i])] = -1
    o = np.argsort(-v)[:100]; U[i, o[v[o] > 0]] = True

# ---- clues
with np.errstate(all="ignore"):
    r1 = C / sh(C, 1) - 1
    def rsi(X):
        dlt = np.nan_to_num(np.diff(X, axis=0, prepend=X[:1]))
        up = pd.DataFrame(np.where(dlt > 0, dlt, 0)).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        dn = pd.DataFrame(np.where(dlt < 0, -dlt, 0)).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))
    ema = lambda X, n: pd.DataFrame(X).ewm(span=n, adjust=False).mean().to_numpy()
    mac = ema(C, 12) - ema(C, 26); mh = (mac - ema(mac, 9)) / C
    hi30, lo30 = roll(H, 30, "max"), roll(L, 30, "min")
    bi = syms.index("BTCUSDT"); bc = C[:, bi]
    bt_r7 = bc / np.r_[[np.nan] * 7, bc[:-7]] - 1; bt_ma = bc / pd.Series(bc).rolling(50).mean().to_numpy() - 1
    ma20 = roll(C, 20, "mean")
    above = np.where(U, (C > ma20).astype(float), np.nan); breadth = np.nanmean(above, axis=1)
    r7 = C / sh(C, 7) - 1
    rank7 = pd.DataFrame(np.where(U, r7, np.nan)).rank(axis=1, pct=True).to_numpy()
    F = {"1日漲跌": r1, "3日漲跌": C / sh(C, 3) - 1, "7日漲跌": r7, "30日漲跌": C / sh(C, 30) - 1,
         "30日區間位置": (C - lo30) / (hi30 - lo30), "10日區間寬": roll(H, 10, "max") / roll(L, 10, "min") - 1,
         "30日區間寬": hi30 / lo30 - 1, "量比20日": Q / sh(roll(Q, 20, "mean"), 1), "20日波動": roll(r1, 20, "std"),
         "RSI": rsi(C), "MACD柱": mh, "離20日線": C / ma20 - 1, "離50日線": C / roll(C, 50, "mean") - 1,
         "BTC 7日漲跌": np.repeat(bt_r7[:, None], S, 1), "BTC離50日線": np.repeat(bt_ma[:, None], S, 1),
         "3日資金費": roll(FD, 3, "sum"), "7日強弱排名": rank7, "站上20日線的幣比例": np.repeat(breadth[:, None], S, 1),
         "上市天數": age.astype(float), "離90日高點": C / roll(H, 90, "max", 60) - 1}
FN = list(F)

# ---- outcome of every kind of move at every coin-day (entry = next open)
KINDS = {"快（+10%/-5%，3天）": (.10, .05, 3), "波段（+20%/-8%，7天）": (.20, .08, 7), "大波（+40%/-15%，14天）": (.40, .15, 14)}
OUT = {}
for kn, (tp, sl, hz) in KINDS.items():
    for side in (1, -1):
        Rm = np.full((T_, S), np.nan); X = np.full((T_, S), -1)
        e = np.vstack([O[1:], np.full((1, S), np.nan)])
        done = np.zeros((T_, S), bool); res = np.full((T_, S), np.nan)
        for h in range(1, hz + 1):
            hh = np.vstack([H[h:], np.full((h, S), np.nan)]); ll = np.vstack([L[h:], np.full((h, S), np.nan)])
            cc = np.vstack([C[h:], np.full((h, S), np.nan)])
            with np.errstate(invalid="ignore"):
                hit_sl = ~done & ((ll <= e * (1 - sl)) if side > 0 else (hh >= e * (1 + sl)))
                hit_tp = ~done & ~hit_sl & ((hh >= e * (1 + tp)) if side > 0 else (ll <= e * (1 - tp)))
            res[hit_sl] = -1; res[hit_tp] = tp / sl
            X[hit_sl | hit_tp] = (np.arange(T_)[:, None] + h)[hit_sl | hit_tp] if False else np.broadcast_to(np.arange(T_)[:, None] + h, (T_, S))[hit_sl | hit_tp]
            done |= hit_sl | hit_tp
            if h == hz:
                last = ~done & ~np.isnan(cc)
                res[last] = ((cc / e - 1) * side / sl)[last]; X[last] = np.broadcast_to(np.arange(T_)[:, None] + h, (T_, S))[last]
        ok = X >= 0
        xi = np.clip(X, 0, T_); fsum = np.zeros((T_, S))
        rows = np.broadcast_to(np.arange(T_)[:, None], (T_, S)); cols = np.broadcast_to(np.arange(S)[None, :], (T_, S))
        fsum[ok] = CF[np.minimum(xi[ok] + 1, T_), cols[ok]] - CF[np.minimum(rows[ok] + 1, T_), cols[ok]]
        Rm[ok] = res[ok] - (2 * SIDE + fsum[ok] * side) / sl
        OUT[(kn, side)] = (Rm, X)
print("labels ready", flush=True)

dix = np.arange(T_)
def t_of(s): return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def search(t0, t1, t2):
    """t0..t1 find (9 months), t1..t2 check (3 months).  Returns passing rules, best first."""
    tr = (days >= t0) & (days < t1); va = (days >= t1) & (days < t2)
    I_tr = U & tr[:, None]; I_va = U & va[:, None]
    edges = {}; conds = []
    for f in FN:
        v = F[f][I_tr]; v = v[np.isfinite(v)]
        if len(v) < 500: continue
        qs = np.quantile(v, [.2, .4, .6, .8]); edges[f] = qs
        for k, qv in enumerate(qs):
            lab = ["最低20%", "最低40%", "最高40%", "最高20%"][k]
            conds.append((f, "<=" if k < 2 else ">=", float(qv), lab))
    def mask(c):
        f, op, qv, _ = c
        with np.errstate(invalid="ignore"): return (F[f] <= qv) if op == "<=" else (F[f] >= qv)
    CM = [mask(c) for c in conds]
    ptr = np.nonzero(I_tr); pva = np.nonzero(I_va)
    CMt = [np.nan_to_num(c[ptr], nan=0).astype(bool) for c in CM]; CMv = [np.nan_to_num(c[pva], nan=0).astype(bool) for c in CM]
    dtr = ptr[0]
    combos = [(a,) for a in range(len(conds))] + [(a, b) for a, b in itertools.combinations(range(len(conds)), 2) if conds[a][0] != conds[b][0]]
    found = []
    for (kn, side), (Rm, X) in OUT.items():
        Rt = Rm[ptr]; Rv = Rm[pva]; gt = np.isfinite(Rt); gv = np.isfinite(Rv)
        for cb in combos:
            mt = (CMt[cb[0]] if len(cb) == 1 else CMt[cb[0]] & CMt[cb[1]]) & gt
            n = int(mt.sum())
            if n < 30: continue
            rr = Rt[mt]; mean = rr.mean()
            if mean < .2: continue
            di = dtr[mt]; ud, inv = np.unique(di, return_inverse=True); nd = len(ud)
            if nd < 15: continue
            per = np.bincount(inv, weights=rr) / np.bincount(inv)
            tv = per.mean() / (per.std() + 1e-9) * np.sqrt(nd)
            mv = (CMv[cb[0]] if len(cb) == 1 else CMv[cb[0]] & CMv[cb[1]]) & gv; nv = int(mv.sum())
            if nv < 5 or Rv[mv].mean() <= 0: continue
            found.append({"rule": cb, "kind": kn, "side": side, "n": n, "days": nd, "mean": float(mean), "t": float(tv),
                          "val_n": nv, "val_mean": float(Rv[mv].mean())})
    found.sort(key=lambda z: -z["t"])
    out = []; used = set()
    for z in found:                                   # best 2 that do not share a clue
        fs = {conds[c][0] for c in z["rule"]}
        if fs & used: continue
        z["text"] = ("做多" if z["side"] > 0 else "做空") + " " + z["kind"] + "：" + " 且 ".join(f"{conds[c][0]} {conds[c][3]}（{conds[c][1]} {conds[c][2]:.3g}）" for c in z["rule"])
        z["masks"] = [CM[c] for c in z["rule"]]
        out.append(z); used |= fs
        if len(out) == 2: break
    return out, len(found)



import sys
def build(P):
    QS = []; q = pd.Timestamp("2021-01-01", tz="UTC")
    while q < pd.Timestamp("2026-11-01", tz="UTC"):
        QS.append(q); q = q + pd.DateOffset(months=P)
    plan = []; log = []
    for q in QS:
        a = int(q.timestamp() * 1000); b = int(min(q + pd.DateOffset(months=P), pd.Timestamp("2026-11-01", tz="UTC")).timestamp() * 1000)
        t0 = int((q - pd.DateOffset(months=12)).timestamp() * 1000); t1 = int((q - pd.DateOffset(months=3)).timestamp() * 1000)
        rules, nfound = search(t0, t1, a)
        for z in rules: plan.append((a, b, z))
        log.append({"start": f"{q:%Y-%m}", "rules": [z["text"] for z in rules] or ["空手"]})
    return plan, log


def account(plan, risk=.02, control=False, a0=t_of("2021-01-01"), a1=t_of("2026-11-01")):
    i0, i1 = np.searchsorted(days, a0), np.searchsorted(days, a1)
    cash = 500.; opn = {}; curve = []; yr = {}; trades = []; per_peak = None; per_id = None; blocked = False
    for i in range(i0, i1):
        for key in [k for k, v in opn.items() if v["x"] <= i]:
            v = opn.pop(key); cash += v["rk"] * v["R"]; trades.append(v)
        eq = cash
        for j, v in opn.items():
            if np.isfinite(C[i, j]) and np.isfinite(v["e"]):
                eq += v["rk"] * float(np.clip((C[i, j] / v["e"] - 1) * v["side"] / v["sl"], -1, v["tp"] / v["sl"]))
        cur = [p for p in plan if p[0] <= days[i] < p[1]]
        pid = cur[0][0] if cur else None
        if pid != per_id: per_id, per_peak, blocked = pid, eq, False
        per_peak = max(per_peak, eq)
        if control and eq < .8 * per_peak: blocked = True
        if not blocked:
            for a, b, z in cur:
                Rm, X = OUT[(z["kind"], z["side"])]
                m = U[i].copy()
                for mm in z["masks"]: m &= np.nan_to_num(mm[i], nan=0).astype(bool)
                for j in np.nonzero(m & np.isfinite(Rm[i]))[0]:
                    if len(opn) >= 10 or j in opn: continue
                    if control and sum(1 for v in opn.values() if v["side"] == z["side"]) >= 3: continue
                    tp, sl, hz = KINDS[z["kind"]]
                    opn[j] = {"rk": cash * risk, "R": float(Rm[i, j]), "x": int(X[i, j]), "e": O[i + 1, j], "side": z["side"], "sl": sl,
                              "tp": tp, "rule": z["text"], "s": syms[j], "t": int(days[i]), "a": a}
        curve.append(eq); yr[pd.Timestamp(days[i], unit="ms").year] = round(eq)
    v = np.array(curve)
    return round(cash), round(float((1 - v / np.maximum.accumulate(v)).max()), 3), yr, trades


res = {}
for P in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "3,2,1").split(",")]:
    plan, log = build(P); res[P] = {"log": log}
    for ctl in (False, True):
        for rk in (.02, .04):
            c_, dd_, y_, tr_ = account(plan, rk, ctl)
            lab = f"每{P}個月重挑｜{'有風控' if ctl else '沒風控'}｜每筆{rk:.0%}"
            print(f"{lab}: 500U → {c_:,}U，最大回落 {dd_:.0%}，{len(tr_)} 筆，各年底 {y_}", flush=True)
            res[P][lab] = {"end": c_, "max_dd": dd_, "year_end": y_, "trades": len(tr_)}
            if P == 3 and ctl and rk == .02:
                mo = {}
                for t in tr_:
                    k = (pd.Timestamp(t["t"], unit="ms").to_period("M") - pd.Timestamp(t["a"], unit="ms").to_period("M")).n
                    mo.setdefault(k, []).append(t["R"])
                print("   每季第幾個月（筆數，平均R）:", {k + 1: (len(v), round(float(np.mean(v)), 2)) for k, v in sorted(mo.items())}, flush=True)
    json.dump(res, open("reverse_wf2.json", "w"), ensure_ascii=False, indent=1, default=str)
