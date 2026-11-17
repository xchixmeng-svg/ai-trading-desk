"""final_opt (2026-11-18, Chi: 好好算、找到最優策略、不停的改、出詳細報告)
Same monthly / bi-monthly re-design as eqw_stop.py + dynamic leverage, but the trade strategies (盤整突破 variants,
抄底) are now MARKED TO MARKET every day (their open trades' profit and loss counts every day, not only on the exit
day), so drawdowns are real.  Then a search over the settings, the best one is chosen (most money with the worst fall
<= 60 %), and a full report is written: every period's strategies and rules, every trade, ROI, CAGR, RR, PF ...

Settings searched: rebuild every 1 / 2 months; strategies kept if 12-month Sharpe >= 0.5 / 1.0; best 2 / 3 / 4
families; equal or Sharpe-weighted; leverage by distance from the account high (<10 % / 10-25 % / >25 %):
4/2/1, 6/3/1, 8/4/2, 10/5/2, 12/6/2; stop inside a period at a 25 % fall or none.  Look-back always 12 months."""
import os, sys, json, pickle, itertools
os.environ.setdefault("FLOW", "1")
src = open("library_wf.py").read().split("NM = list(STR); M = np.vstack")[0]
exec(src)
for k in ("盤整突破 v12（A或B）", "盤整突破 全部", "抄底 v11"):
    STR.pop(k, None)
exec(open("dynamic_wf.py").read().split('FAM = {k: k for k in STR}')[0].split('ev = pickle.load')[1].split('\n', 1)[1])  # DESC dict
FAM = {k: k for k in STR}
ev = pickle.load(open("event_trades_mtm.pkl", "rb"))
dix = {int(t): k for k, t in enumerate(days)}


def mtm(trs):
    """daily stream (1 % risk per trade) and, per trade, its daily R increments {day_index: dR}"""
    out = np.zeros(len(days)); inc = []
    for r in trs:
        prev = 0.0; parts = {}
        for t, m in r["dm"]:
            k = dix.get(int(t) - D)
            if k is None or t >= r["xt"]: continue
            parts[k] = parts.get(k, 0) + (m - prev); prev = m
        k = dix.get(int((r["xt"] - 1) // D * D))
        if k is None: continue
        parts[k] = parts.get(k, 0) + (r["R"] - prev)
        for kk, v in parts.items(): out[kk] += .01 * v
        inc.append(parts)
    byday = {}
    for ti, parts in enumerate(inc):
        for kk, v in parts.items(): byday.setdefault(kk, []).append((ti, v))
    return out, (inc, byday)


TRD = {}
for L_ in (192, 240, 288):
    for W_ in (.12, .15, .18):
        for V_ in (1.5, 2.0, 2.5):
            rs = [r for r in ev["C"] if r["L"] == L_ and r["w"] <= W_ and r["vx"] >= V_]
            for fl, fn in (("AB", "A或B"), ("A", "只A"), ("ALL", "全部")):
                for sd, sn in ((1, "多"), (-1, "空"), (0, "多空")):
                    sel = [r for r in rs if (fl == "ALL" or r["A"] or (fl == "AB" and r["B"])) and (sd == 0 or r["side"] == sd)]
                    nm = f"盤整突破 {L_ // 24}天 寬{W_:.0%} 量{V_}x {fn} {sn}"
                    STR[nm], (inc, bd) = mtm(sel); FAM[nm] = "盤整突破"; TRD[nm] = (sel, inc, bd)
                    DESC[nm] = (f"成交額前 100 大的加密貨幣；過去 {L_ // 24} 天（{L_} 根 1 小時 K）最高／最低 ≤ {1 + W_:.2f} 倍（寬 ≤{W_:.0%}）；"
                                f"1 小時收盤突破區間上緣（做多）或跌破下緣（做空），最近 4 小時成交額 ≥ 平常 {V_} 倍；"
                                + {"AB": "且符合 A 或 B（A：4h RSI 過熱＋日線量價背離；B：散戶多空比反向＋日 MACD 柱同向＋日 RSI 中段）",
                                   "A": "且符合 A（4h RSI 做多 >70／做空 <30，日線量價背離）", "ALL": "不加 A／B 條件"}[fl]
                                + f"；{ {1: '只做多', -1: '只做空', 0: '多空都做'}[sd] }；下一根 K 開盤進場，停損在區間另一邊，之後停損跟著最好價格移動同樣距離，最多 30 天")
for nm, trs in (("抄底 v11", ev["DIP"]), ("抄底 v11（要多空比）", [r for r in ev["DIP"] if r.get("acc", 9) <= 2.0])):
    STR[nm], (inc, bd) = mtm(trs); FAM[nm] = "抄底"; TRD[nm] = (trs, inc, bd)
DESC["抄底 v11"] = ("成交額前 100 大、上市滿 120 天；4 小時 K 收盤 ≤ 過去 30 天最高價一半、單根跌 >10%、成交額 ≥ 前 30 根中位數 6 倍；"
                   "之後 12 小時內第一根 1 小時錘子線收盤買進（中途先跌 15% 取消）；BTC 在 50 日均線下、該幣 30 天跌幅 <52.7%；"
                   "停損 −15%，漲到 +15% 後停損拉到 +7.5%，最多 72 小時")
DESC["抄底 v11（要多空比）"] = DESC["抄底 v11"] + "；另外散戶多空比 ≤ 2"
NM = list(STR); M = np.vstack([STR[k] for k in NM]).T
np.save("final_streams.npy", M); json.dump({"names": NM, "fam": FAM}, open("final_streams.json", "w"), ensure_ascii=False)
print("streams", len(NM), flush=True)


def picks(a, thr, topn):
    t0 = int((pd.Timestamp(a, unit="ms", tz="UTC") - pd.DateOffset(months=12)).timestamp() * 1000); tr = (days >= t0) & (days < a)
    best = {}
    for k, nm in enumerate(NM):
        v = M[tr, k]; sd = v.std()
        if sd <= 0 or (v != 0).sum() < 20 or v.sum() <= 0: continue
        sh = v.mean() / sd * np.sqrt(365)
        if sh < thr: continue
        f = FAM[nm]
        if f not in best or sh > best[f][0]: best[f] = (sh, k, min(5.0, .25 / (sd * np.sqrt(365))))
    return sorted(best.values(), reverse=True)[:topn]


START = pd.Timestamp("2021-01-01", tz="UTC"); END = pd.Timestamp("2026-11-01", tz="UTC")
def periods(P):
    q = START; out = []
    while q < END:
        q2 = min(q + pd.DateOffset(months=P), END); out.append((q, int(q.timestamp() * 1000), int(q2.timestamp() * 1000))); q = q2
    return out
CACHE = {}
def plan(P, thr, topn):
    key = (P, thr)
    if key not in CACHE: CACHE[key] = [(q, a, b, picks(a, thr, 4)) for q, a, b in periods(P)]
    return [(q, a, b, sel[:topn]) for q, a, b, sel in CACHE[key]]


def sim(pl, wmode, tiers, stop, detail=False):
    Lmax, Lmid, Lmin = tiers
    eq = 500.; pa = 500.; mdd = 0.; curve = []; per_rows = []; contrib = {}; tl = {}
    for q, a, b, sel in pl:
        te = np.nonzero((days >= a) & (days < b))[0]
        if sel:
            w = np.array([s for s, _, _ in sel]) if wmode == "sharpe" else np.ones(len(sel)); w = w / w.sum()
        start = eq; pk = eq; off = False; lev_used = []
        for i in te:
            if not off and sel:
                dd = 1 - eq / pa; lev = Lmax if dd < .10 else (Lmid if dd < .25 else Lmin); lev_used.append(lev)
                parts = [lev * wi * sc * M[i, k] for wi, (_, k, sc) in zip(w, sel)]
                x = max(sum(parts), -.99)
                if detail:
                    for (_, k, sc), wi, pv in zip(sel, w, parts):
                        contrib[(q, NM[k])] = contrib.get((q, NM[k]), 0) + eq * pv
                        if NM[k] in TRD:
                            for ti, v in TRD[NM[k]][2].get(i, []):
                                key = (NM[k], ti); tl.setdefault(key, {"pnl": 0.0, "risk": None, "q": q})
                                tl[key]["pnl"] += eq * lev * wi * sc * .01 * v
                                if tl[key]["risk"] is None: tl[key]["risk"] = lev * wi * sc * .01
                eq *= 1 + x; pk = max(pk, eq)
                if stop and eq <= pk * (1 - stop): off = True
            pa = max(pa, eq); mdd = max(mdd, 1 - eq / pa); curve.append((int(days[i]), eq))
        if detail:
            per_rows.append({"期": f"{q:%Y-%m}", "策略": [(NM[k], float(wi), float(sh), float(sc)) for wi, (sh, k, sc) in zip(w, sel)] if sel else [],
                             "平均放大": float(np.mean(lev_used)) if lev_used else 0, "停損觸發": off, "期初": start, "期末": eq})
    return eq, mdd, curve, per_rows, contrib, tl


res = []
for P, thr, topn, wmode, tiers, stop in itertools.product((1, 2), (.5, 1.0), (2, 3, 4), ("equal", "sharpe"),
                                                          ((4, 2, 1), (6, 3, 1), (8, 4, 2), (10, 5, 2), (12, 6, 2)), (None, .25)):
    eq, mdd, curve, *_ = sim(plan(P, thr, topn), wmode, tiers, stop)
    res.append({"P": P, "thr": thr, "topn": topn, "w": wmode, "tiers": tiers, "stop": stop, "end": eq, "mdd": mdd})
json.dump(res, open("final_grid.json", "w"), ensure_ascii=False, default=str)
ok = sorted([r for r in res if r["mdd"] <= .60], key=lambda r: -r["end"])
print(f"\n設定組合 {len(res)} 組；最大回落 ≤60% 的有 {len(ok)} 組。前 15 名：", flush=True)
for r in ok[:15]:
    print(f"每{r['P']}月 Sharpe≥{r['thr']} 前{r['topn']} {r['w']} 放大{r['tiers']} 停損{r['stop']}: 500U → {r['end']:,.0f}U 回落 {r['mdd']:.0%}", flush=True)
print("全部最高：", sorted(res, key=lambda r: -r["end"])[0])
