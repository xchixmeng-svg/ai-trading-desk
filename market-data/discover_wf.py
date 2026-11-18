"""discover_wf.py (2026-11-18, Claude, for Chi): every 2 months, redo the v12 discovery on the previous 12 months only.

Per period P (2023-01, 2023-03, ... 2026-09), using only trades that opened AND closed inside [P-12m, P):
 1. For each trigger family x side x exit x range length (ev_build.py; 56 groups) search entry rules = up to 3 feature
    conditions (thresholds at the window's own quantiles, beam search).  A rule counts only if it has >= 12 trades,
    both halves of the 12 months are positive, and the total without its best 2 trades is still positive.
    Score = total R without the best 2 trades (how much it would have grown the account, luck removed).
 2. Pick up to 5 rules (best score first, skip a rule whose trades are >= 50 % already covered).
 3. Risk per trade for each rule: tried 1 / 2 / 3 / 4.5 / 6 / 7.5 / 10 % on the same 12 months with the account rules;
    keep the setting with the highest end balance whose worst fall stays <= 50 %.
 4. Trade those rules for the next 2 months.  Open trades run to their own exit.
Account (same as v12): 500U start, margin <= 68 %, open stop-risk <= 40 %, <= 30 open, one per coin,
leverage = min(20, 0.8 / stop), a 60 % fall from the peak stops new trades until the next re-design."""
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

z = np.load("ev.npz", allow_pickle=True)
F, O, MK, S, T = z["F"], z["O"], z["MK"], z["S"], z["T"]; FE = list(z["FEAT"])
fi = {n: FE.index(n) for n in FE}
H = 3_600_000; D = 24 * H
ts_ = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
FAMS = ["突破", "急漲", "急跌", "趨勢回踩", "過度延伸"]
EXN = {0: "移動停利(1倍)", 1: "移動停利(2倍)", 2: "固定停利+2R(5天)", 3: "到+1R鎖+0.5R(72h)"}
LAB = {"w": "盤整寬度", "d": "停損距離", "vx": "4h量倍數", "r4": "4h RSI", "rd": "日RSI", "hist_dir": "日MACD柱同向",
       "vdiv": "日量價背離", "acc": "散戶多空比", "topacc": "大戶多空比", "taker": "主動買賣比", "oi24": "持倉24h變化",
       "fund": "資金費率(順向付)", "btc_up": "BTC在50日線上", "btc_r7": "BTC 7日漲跌", "r7": "7日順向漲跌",
       "r30": "30日順向漲跌", "dd30": "距30日高點", "age": "上市天數", "atr": "日波幅", "ham": "錘子線",
       "vrank": "成交量排名", "r4h": "4h順向漲跌", "rd_dir": "日RSI同向"}
NUM = ["w", "d", "vx", "r4", "rd", "acc", "topacc", "taker", "oi24", "fund", "btc_r7", "r7", "r30", "dd30", "age", "atr", "vrank", "r4h"]
BIN = ["hist_dir", "vdiv", "btc_up", "ham", "rd_dir"]
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LB = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PICK = sys.argv[3] if len(sys.argv) > 3 else "score"
K = 5; RISKS = (.01, .02, .03, .045, .06, .075, .10)

fam, side, L = F[:, fi["fam"]], F[:, fi["side"]], F[:, fi["L"]]
GROUPS = []
for f_ in range(5):
    for sd in (1, -1):
        for ex in range(4):
            for L_ in ((120, 240, 480) if f_ == 0 else (0,)):
                GROUPS.append((f_, sd, ex, L_))
gidx = {g: np.flatnonzero((fam == g[0]) & (side == g[1]) & (L == g[3])) for g in {(a, b, 0, d) for a, b, c, d in GROUPS}}


def conds(idx):
    """candidate conditions on the events idx: (text, mask)"""
    out = []
    for n in NUM:
        v = F[idx, fi[n]]; ok = np.isfinite(v)
        if ok.sum() < 30: continue
        for qv in np.unique(np.round(np.quantile(v[ok], (.2, .35, .5, .65, .8)), 4)):
            with np.errstate(invalid="ignore"):
                out.append(((n, "<=", float(qv)), v <= qv)); out.append(((n, ">=", float(qv)), v >= qv))
    for n in BIN:
        v = F[idx, fi[n]]
        out.append(((n, "==", 1.0), v == 1)); out.append(((n, "==", 0.0), v == 0))
    return out


def stats(R, half, M):
    """M: n x C masks -> score, n, half sums (vectorised)"""
    Mf = M.astype(np.float64)
    n = Mf.sum(0); tot = R @ Mf; h1 = (R * half) @ Mf; h2 = tot - h1
    W = np.where(M, R[:, None], -np.inf)
    top2 = -np.partition(-W, 1, axis=0)[:2] if len(R) >= 2 else np.zeros((2, M.shape[1]))
    top2 = np.where(np.isfinite(top2), top2, 0).sum(0)
    trim = tot - top2
    ok = (n >= 12) & (h1 > 0) & (h2 > 0) & (trim > 0)
    return np.where(ok, trim, -np.inf), n, tot


def search(a0, a1, keep=3):
    """all good rules in window [a0, a1)"""
    found = []
    mid = a0 + (a1 - a0) // 2
    for g in GROUPS:
        f_, sd, ex, L_ = g
        base = gidx[(f_, sd, 0, L_)]
        xt = O[base, 3 * ex + 1]
        idx = base[(T[base] >= a0) & (xt < a1)]
        if len(idx) < 12: continue
        R = O[idx, 3 * ex]; half = (T[idx] < mid).astype(float)
        cs = conds(idx)
        if not cs: continue
        CM = np.stack([m for _, m in cs], 1)
        beam = [((), np.ones(len(idx), bool))]
        sc0, n0, t0 = stats(R, half, beam[0][1][:, None])
        best = [(sc0[0], (), beam[0][1])] if np.isfinite(sc0[0]) else []
        for depth in range(3):
            cand = []
            for rule, m in beam:
                sc, n, tot = stats(R, half, CM & m[:, None])
                for c in np.argsort(-sc)[:10]:
                    if not np.isfinite(sc[c]): break
                    if any(cs[c][0][0] == r_[0] for r_ in rule): continue
                    cand.append((sc[c], rule + (cs[c][0],), CM[:, c] & m))
            if not cand: break
            cand.sort(key=lambda z_: -z_[0]); seen = set(); beam = []
            for sc_, rule, m in cand:
                key = tuple(sorted(rule))
                if key in seen: continue
                seen.add(key); beam.append((rule, m))
                best.append((sc_, rule, m))
                if len(beam) >= 8: break
        best.sort(key=lambda z_: -z_[0])
        for sc_, rule, m in best[:keep]:
            found.append({"g": g, "rule": rule, "score": float(sc_), "idx": idx[m], "R": R[m]})
    found.sort(key=lambda z_: -z_["score"])
    return found


def match(g, rule, a0, a1):
    """events of group g matching rule with signal in [a0, a1) (exit may be later)"""
    f_, sd, ex, L_ = g
    base = gidx[(f_, sd, 0, L_)]; idx = base[(T[base] >= a0) & (T[base] < a1)]
    m = np.ones(len(idx), bool)
    for n, op, v in rule:
        x = F[idx, fi[n]]
        with np.errstate(invalid="ignore"):
            m &= (x <= v) if op == "<=" else (x >= v) if op == ">=" else (x == v)
    return idx[m]


def trades(idx, ex, kind):
    return [{"s": S[i], "t": int(T[i]), "xt": int(O[i, 3 * ex + 1]), "R": float(O[i, 3 * ex]), "d": float(O[i, 3 * ex + 2]),
             "mk": MK[i, ex], "kind": kind, "ev": int(i)} for i in idx]


def pool(TR, risk, cash=500., kill=.60, resets=(), until=None):
    """v12 account rules; daily marks for the worst fall; kill stops new trades until the next reset time"""
    ev = []
    for n, r in enumerate(TR):
        ev.append((r["t"], 1, n)); ev.append((r["xt"], 0, n))
        for m, v in enumerate(r["mk"]):
            if not np.isfinite(v): break
            tm = r["t"] + (m + 1) * D
            if tm < r["xt"]: ev.append((tm, 2, n))
    for tm in resets: ev.append((tm, 3, -1))
    ev.sort(key=lambda z_: (z_[0], z_[1]))
    opn = {}; peak = cash; dd = 0.; dead = False; taken = 0; mark = {}; curve = []; coins = {}
    log = []
    for tm, typ, n in ev:
        if until is not None and tm >= until: break
        if typ == 3:
            dead = False; peak = cash + sum(o_[0] * mark.get(k_, 0) for k_, o_ in opn.items()); continue
        r = TR[n] if n >= 0 else None
        if typ == 2:
            if n in opn:
                mark[n] = float(r["mk"][(tm - r["t"]) // D - 1])
                eq = cash + sum(o_[0] * mark.get(k_, 0) for k_, o_ in opn.items())
                peak = max(peak, eq); dd = max(dd, 1 - eq / peak)
                if 1 - eq / peak >= kill: dead = True
                curve.append((tm, eq))
            continue
        if typ == 0:
            if n in opn:
                rk = opn.pop(n)[0]; mark.pop(n, None); coins.pop(r["s"], None)
                cash += rk * r["R"]
                eq = cash + sum(o_[0] * mark.get(k_, 0) for k_, o_ in opn.items())
                peak = max(peak, eq); dd = max(dd, 1 - eq / peak)
                if 1 - eq / peak >= kill: dead = True
                curve.append((tm, eq)); log.append(dict(r, risk_u=rk, pnl=rk * r["R"], cash_after=cash))
            continue
        if dead or len(opn) >= 30 or r["s"] in coins: continue
        rk = cash * risk[r["kind"]]
        lev = max(1, min(20, int(.8 / r["d"]))); mg = rk / r["d"] / lev
        if sum(v[1] for v in opn.values()) + mg > .68 * cash: continue
        if sum(v[0] for v in opn.values()) + rk > .40 * cash: continue
        opn[n] = (rk, mg); coins[r["s"]] = n; taken += 1
    return cash, dd, taken, log, curve


def desc(g, rule):
    f_, sd, ex, L_ = g
    s = f"{FAMS[f_]}{'做多' if sd > 0 else '做空'}" + (f"（{L_ // 24}天區間）" if f_ == 0 else "") + f"｜出場：{EXN[ex]}"
    cs = []
    for n, op, v in rule:
        if op == "==": cs.append(f"{LAB[n]}{'是' if v == 1 else '否'}")
        else: cs.append(f"{LAB[n]}{'≤' if op == '<=' else '≥'}{v:g}")
    return s + "｜條件：" + ("、".join(cs) if cs else "無（全部訊號）")


if __name__ == "__main__":
    P = []; q = pd.Timestamp("2023-01-01", tz="UTC")
    while q < pd.Timestamp("2026-11-01", tz="UTC"):
        P.append(q); q = q + pd.DateOffset(months=STEP)
    END = ts_("2026-11-01")
    ALLTR = []; RISK = {}; PLAN = []
    for pi, q in enumerate(P):
        a1 = int(q.timestamp() * 1000); a0 = int((q - pd.DateOffset(months=LB)).timestamp() * 1000)
        b1 = int(P[pi + 1].timestamp() * 1000) if pi + 1 < len(P) else END
        found = search(a0, a1, keep=10 if PICK == "minq" else 3)
        if PICK == "minq":                      # rank by the worst of 4 equal sub-periods (every part must hold up)
            for r in found:
                tt = T[r["idx"]]; qs_ = [r["R"][(tt >= a0 + k_ * (a1 - a0) // 4) & (tt < a0 + (k_ + 1) * (a1 - a0) // 4)].sum() for k_ in range(4)]
                r["minq"] = min(qs_)
            found.sort(key=lambda z_: -z_["minq"])
        chosen = []; cover = set()
        for r in found:
            keys = {(S[i], T[i]) for i in r["idx"]}
            if len(keys & cover) >= .5 * len(keys): continue
            chosen.append(r); cover |= keys
            if len(chosen) >= K: break
        # risk per rule on the same 12 months
        kinds = [f"{q:%Y-%m}#{k_ + 1}" for k_ in range(len(chosen))]
        win = []
        for r, kd in zip(chosen, kinds): win += trades(r["idx"], r["g"][2], kd)
        win.sort(key=lambda z_: z_["t"])
        rk = {kd: .02 for kd in kinds}
        def val(rk_):
            c_, d_, *_ = pool(win, rk_, kill=9)
            return c_ if d_ <= .50 else -d_
        cur = val(rk)
        for _ in range(2):
            for kd in kinds:
                for v in RISKS:
                    tr_ = dict(rk, **{kd: v}); x = val(tr_)
                    if x > cur: cur, rk = x, tr_
        wc, wd, wn, *_ = pool(win, rk, kill=9)
        RISK.update(rk)
        oos = []
        for r, kd in zip(chosen, kinds):
            oos += trades(match(r["g"], r["rule"], a1, b1), r["g"][2], kd)
        ALLTR += oos
        PLAN.append({"period": f"{q:%Y-%m}", "window": f"{pd.Timestamp(a0, unit='ms'):%Y-%m}～{pd.Timestamp(a1 - 1, unit='ms'):%Y-%m}",
                     "in_window_500U": round(wc), "in_window_dd": round(wd, 3),
                     "rules": [{"kind": kd, "text": desc(r["g"], r["rule"]), "risk": rk[kd], "win_trades": len(r["R"]),
                                "win_R": round(float(r["R"].sum()), 1), "score": round(r["score"], 1),
                                "g": list(map(int, r["g"])), "rule": r["rule"], "oos_signals": int(sum(t_["kind"] == kd for t_ in oos))}
                               for r, kd in zip(chosen, kinds)]})
        print(f"\n== {q:%Y-%m}（用 {PLAN[-1]['window']}）過去12個月模擬 500U→{wc:,.0f}U 回落{wd:.0%}", flush=True)
        for x in PLAN[-1]["rules"]:
            print(f"  {x['kind']} 風險{x['risk']:.1%} 過去{x['win_trades']}筆 {x['win_R']:+.1f}R → 本期訊號{x['oos_signals']}｜{x['text']}", flush=True)
        ALLTR.sort(key=lambda z_: z_["t"])
        resets = [int(p.timestamp() * 1000) for p in P[1:pi + 1]]
        c_, d_, n_, log, _ = pool(ALLTR, RISK, resets=resets, until=b1)
        print(f"  ▶ 帳戶到 {pd.Timestamp(b1 - 1, unit='ms'):%Y-%m} 底：{c_:,.0f}U（已平倉部分），最大回落 {d_:.0%}，{n_} 筆", flush=True)
    resets = [int(p.timestamp() * 1000) for p in P[1:]]
    c_, d_, n_, log, curve = pool(ALLTR, RISK, resets=resets)
    print(f"\n=== 2023-01～2026-10 每{STEP}個月重擬：500U → {c_:,.0f}U，最大回落 {d_:.0%}，{n_} 筆", flush=True)
    tag = f"_s{STEP}" + (f"_lb{LB}_{PICK}" if (LB, PICK) != (12, "score") else "")
    json.dump(PLAN, open(f"discover_plan{tag}.json", "w"), ensure_ascii=False, indent=1, default=float)
    pd.DataFrame([{k_: v for k_, v in r.items() if k_ != "mk"} for r in log]).to_csv(f"discover_trades{tag}.csv", index=False)
    pd.DataFrame(curve, columns=["t", "eq"]).to_csv(f"discover_curve{tag}.csv", index=False)
