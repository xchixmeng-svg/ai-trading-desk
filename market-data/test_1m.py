"""test_1m.py (2026-11-18, Claude, for Chi): v12 original vs today's version, exits on 1-minute bars vs 1-hour bars.
Candidates (signals) from dump_cands.py (same signals for both; 2023-01..2026-10). 1m exits: stop checked before the
best price is updated, every minute. Cost 0.4 % + funding. 1m files are deleted after each month (disk)."""
import glob
import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd

from load import load, months, CACHE

H = 3_600_000; D = 24 * H; M_ = 60_000
TS = lambda x: int(pd.Timestamp(x, tz="UTC").timestamp() * 1000)
COST = 0.004
C = pickle.load(open("cands_2023.pkl", "rb"))
fu = load("funding", "2023-01", "2026-10", columns=["symbol", "ts", "rate"])
FU = {s: (g.ts.to_numpy(np.int64), np.cumsum(g.rate.to_numpy(float))) for s, g in fu.sort_values("ts").groupby("symbol")}
del fu


def fund(s, a_, b_):
    if s not in FU: return 0.0
    a, cs = FU[s]; i = np.searchsorted(a, a_, side="right") - 1; j = np.searchsorted(a, b_, side="right") - 1
    return float((cs[j] if j >= 0 else 0) - (cs[i] if i >= 0 else 0))


class W:
    """one trade walked minute by minute"""
    def __init__(self, c, rule):
        self.c = c; self.rule = rule; self.e = None; self.done = False; self.hourly = []
        self.side = c["side"] if c["kind"] != "DIP" else 1
        self.d = c["d"] if c["kind"] != "DIP" else .15
        self.end = c["t"] + (72 * H if c["kind"] == "DIP" else 720 * H)
        self.locked = False; self.last_c = None; self.last_t = None

    def feed(self, ot, o, h, l, cl):
        c = self.c; i0 = np.searchsorted(ot, max(c["t"], ot[0] if len(ot) else 0)); i1 = np.searchsorted(ot, self.end)
        sd, d = self.side, self.d
        for i in range(i0, i1):
            if ot[i] < c["t"]: continue
            if self.e is None:
                self.e = o[i]; e = self.e
                self.sl = e * (1 - sd * d); self.best = e
            e = self.e
            if c["kind"] == "DIP":
                if l[i] <= self.sl: return self.close(min(o[i], self.sl), ot[i] + M_)
                if not self.locked and h[i] >= e * 1.15: self.locked = True; self.sl = e * 1.075
            else:
                if sd > 0:
                    if l[i] <= self.sl: return self.close(min(o[i], self.sl), ot[i] + M_)
                    self.best = max(self.best, h[i])
                    m = 2.0 if (self.rule == "wide" and self.best >= e * (1 + 2 * d)) else 1.0
                    self.sl = max(self.sl, self.best * (1 - m * d))
                else:
                    if h[i] >= self.sl: return self.close(max(o[i], self.sl), ot[i] + M_)
                    self.best = min(self.best, l[i])
                    m = 2.0 if (self.rule == "wide" and self.best <= e * (1 - 2 * d)) else 1.0
                    self.sl = min(self.sl, self.best * (1 + m * d))
            self.last_c, self.last_t = cl[i], ot[i] + M_
            if (ot[i] + M_) % H == 0: self.hourly.append((int(ot[i] + M_), (cl[i] / e - 1) * sd / d))
        if i1 < len(ot) or (len(ot) and ot[-1] + M_ >= self.end):
            if self.last_c is not None: return self.close(self.last_c, self.last_t)

    def close(self, x, xt):
        c = self.c; sd, d = self.side, self.d
        self.R = ((x / self.e - 1) * sd - COST - fund(c["s"], c["t"], xt) * sd) / d
        self.xt = int(xt); self.done = True


WK = []
for c in C:
    WK.append(W(c, "orig"))
    if c["kind"] != "DIP": WK.append(W(c, "wide"))
for m in months("2023-01", "2026-10"):
    a0 = TS(m + "-01"); a1 = TS(months(m, "2099-12")[1] + "-01")
    act = [w for w in WK if not w.done and w.c["t"] < a1 and w.end > a0]
    if not act: continue
    syms = sorted({w.c["s"] for w in act})
    k = load("klines_1m", m, m, symbols=syms, columns=["symbol", "open_time", "open", "high", "low", "close"])
    G = {s: g.sort_values("open_time")[["open_time", "open", "high", "low", "close"]].to_numpy("float64") for s, g in k.groupby("symbol")}
    del k
    for w in act:
        a = G.get(w.c["s"])
        if a is not None and len(a): w.feed(*a.T)
    for f in glob.glob(os.path.join(CACHE, f"m-{m}", "klines_1m_*.parquet")): os.remove(f)
    print(m, "active", len(act), "done", sum(w.done for w in WK), flush=True)
for w in WK:
    if not w.done and w.last_c is not None: w.close(w.last_c, w.last_t)
    elif not w.done: w.R, w.xt, w.done = 0.0, w.c["t"] + H, True
res = {}
for w in WK: res[(w.c["s"], w.c["t"], w.rule)] = (w.R, w.xt, w.hourly)
pickle.dump(res, open("test_1m_res.pkl", "wb"))

SPEC = {"A": .0575, "B": .0225, "DIP": .075}


def pool(T, risk, k=0.0, cap=None, kill=.60):
    ev = sorted([(r["t"], 1, n) for n, r in enumerate(T)] + [(r["xt"], 0, n) for n, r in enumerate(T)])
    marks = {}
    for n, r in enumerate(T):
        for hh, v in r["hourly"]: marks.setdefault(hh, []).append((n, v))
    hs = sorted(marks); hi = 0; cash = 500.; opn = {}; peak = 500.; dead = False; dd = 0.; nn = 0; yearly = {}
    for tm, typ, n in ev:
        while hi < len(hs) and hs[hi] <= tm:
            eq = cash + sum(opn[m][0] * v for m, v in marks[hs[hi]] if m in opn)
            peak = max(peak, eq); dd = max(dd, 1 - eq / peak)
            if 1 - eq / peak >= kill: dead = True
            yearly[dt.datetime.fromtimestamp(hs[hi] / 1000, dt.UTC).year] = eq; hi += 1
        r = T[n]
        if typ == 0:
            if n in opn: rk, mg, ic = opn.pop(n); cash += rk * r["R"] - ic; peak = max(peak, cash)
            continue
        if dead or len(opn) >= 30 or any(T[m]["s"] == r["s"] for m in opn): continue
        if r["kind"] == "DIP" and sum(T[m]["kind"] == "DIP" for m in opn) >= 3: continue
        rk = cash * risk[r["kind"]]; v24, vh, sig = r["liq"]; d = r["d"] if r["kind"] != "DIP" else .15
        notional = rk / d
        if cap is not None and vh > 0 and notional > cap * vh: notional = cap * vh; rk = notional * d
        lev = r.get("lev") or max(1, min(20, int(.8 / d))); mg = rk / d / lev
        if sum(v[1] for v in opn.values()) + mg > .68 * cash: continue
        if sum(v[0] for v in opn.values()) + rk > .40 * cash: continue
        ic = 2 * k * sig * np.sqrt(notional / v24) * notional if (k and v24 > 0) else 0.
        opn[n] = (rk, mg, ic); nn += 1
    return cash, dd, nn, yearly


def trades(version, bar):
    out = []
    for c in C:
        kd = c["kind"]
        if version == "new":
            if kd != "DIP" and not c["up"] and c["side"] > 0: continue
            if kd == "A" and c["up"]: kd = "A_up"
        rule = "wide" if (version == "new" and c["kind"] != "DIP") else "orig"
        if bar == "1h":
            R, xt, hourly = c["h1_wide"] if rule == "wide" else c["h1_orig"]
        else:
            R, xt, hourly = res[(c["s"], c["t"], rule)]
        out.append(dict(c, kind=kd, R=R, xt=xt, hourly=hourly))
    return out


RK = dict(SPEC, A_up=.10)
print("\n=== 2023-01～2026-10，500U ===", flush=True)
for ver, lab in (("v12", "v12 原樣"), ("new", "今天挑的版本")):
    for bar in ("1h", "1m"):
        T = trades(ver, bar)
        for k, cap, cl in ((0, None, "不算衝擊"), (1, .05, "分批＋5%上限")):
            c_, d_, n_, y_ = pool(T, RK if ver == "new" else SPEC, k, cap)
            print(f"{lab}｜{bar} 出場｜{cl}: 500U → {c_:,.0f}U，最大回落 {d_:.0%}，{n_} 筆，各年底 {({kk: round(v) for kk, v in y_.items()})}", flush=True)
