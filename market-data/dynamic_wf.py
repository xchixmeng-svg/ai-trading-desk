"""dynamic_wf (2026-11-18, Chi): re-design the strategy every 1 / 2 / 3 months from the previous 12 months, from ALL
strategies we have, aiming at the most profit.  Built on library_wf.py (all its daily strategies + money-flow ones) plus
every 盤整突破 variant (3 range lengths x 3 widths x 3 volume levels x A or B / A only / all x long / short / both) and
抄底 (with / without the long-short ratio condition), from event_trades_full.pkl.

Each rebuild, using only the previous 12 months:
  1. every strategy is a daily return stream at a base size (trade strategies: 1 % of the account at risk per trade,
     booked on the exit day; daily strategies: their own weights);
  2. keep the ones that made money there (>= 20 active days), best variant per family, rank families by
     return / max drawdown, take the best 3;
  3. split the money by their 12-month profit (more to the one that made more);
  4. scale the whole mix so that it would have fallen at most 35 % in those 12 months (scale cap 10);
  5. trade exactly that for the next period.
Account: 500U, compounding daily, 2021-01 to 2026-10."""
import os
import sys
import pickle

os.environ.setdefault("FLOW", "1")
src = open("library_wf.py").read().split("NM = list(STR); M = np.vstack")[0]
exec(src)

ev = pickle.load(open("event_trades_full.pkl", "rb"))
DESC = {
    "BTC 站上200日線做多": "BTC 日收盤在 200 日均線上就持有，跌破就空手",
    "BTC 50日線多空": "BTC 在 50 日均線上做多、下面做空",
    "ETH 站上50日線做多": "ETH 日收盤在 50 日均線上就持有",
    "BTC 買入持有": "一直拿著 BTC",
    "時序動能 20日 前20大": "成交額前 20 大，過去 20 天漲的做多、跌的做空（波動小的下多一點）",
    "時序動能 60日 前20大": "成交額前 20 大，過去 60 天漲的做多、跌的做空",
    "強者恆強 30日 前50大": "前 50 大裡，買過去 30 天最強 10 個、空最弱 10 個，每週換",
    "強者恆強 7日 前50大": "前 50 大裡，買過去 7 天最強 10 個、空最弱 10 個，每週換",
    "短線反轉 1日 前50大": "前 50 大裡，買昨天跌最多的、空漲最多的",
    "資金費套利 前50大": "前 50 大裡，空資金費最高 10 個、買最低 10 個，收資金費",
    "熊市做空 前20大": "BTC 在 200 日線下時，平均做空前 20 大",
    "區間低點 45天": "前 20 大，45 天區間 ≤30%、收在區間下 20% 就買，停損區間低點 −3%",
    "新幣上市做空": "新上市 30 天內的幣平均做空",
    "爆倉沖洗後做多": "前 50 大，單日跌 ≥8% 且持倉減少 ≥10%，買進抱 5 天",
    "散戶多空比反向": "前 50 大，空散戶最看多 10 個、買散戶最看空 10 個，每天換",
    "大戶多空比跟隨": "前 50 大，買大戶最看多 10 個、空最看空 10 個",
    "主動買力強者做多": "前 50 大，買 3 天主動買最強 10 個、空最弱 10 個",
    "上漲且持倉大增做多": "前 50 大，7 天上漲且持倉增加 >20% 的做多",
}
FAM = {k: k for k in STR}
for k in ("盤整突破 v12（A或B）", "盤整突破 全部", "抄底 v11"):
    STR.pop(k, None)
for L_ in (192, 240, 288):
    for W_ in (.12, .15, .18):
        for V_ in (1.5, 2.0, 2.5):
            rs = [r for r in ev["C"] if r["L"] == L_ and r["w"] <= W_ and r["vx"] >= V_]
            for fl, fn in (("AB", "A或B"), ("A", "只A"), ("ALL", "全部")):
                for sd, sn in ((1, "多"), (-1, "空"), (0, "多空")):
                    sel = [r for r in rs if (fl == "ALL" or r["A"] or (fl == "AB" and r["B"])) and (sd == 0 or r["side"] == sd)]
                    nm = f"盤整突破 {L_ // 24}天 寬{W_:.0%} 量{V_}x {fn} {sn}"
                    STR[nm] = book(sel); FAM[nm] = "盤整突破"
                    DESC[nm] = (f"前 100 大，{L_ // 24} 天盤整區寬 ≤{W_:.0%}，1 小時收盤突破、4 小時量 ≥{V_} 倍，"
                                + {"AB": "且符合 A 或 B", "A": "且符合 A（4h RSI 過熱＋日線量價背離）", "ALL": "不加條件"}[fl]
                                + f"，{ {1: '只做多', -1: '只做空', 0: '多空都做'}[sd] }；停損在區間另一邊並跟著移動，最多 30 天")
STR["抄底 v11"] = book(ev["DIP"]); FAM["抄底 v11"] = "抄底"
DESC["抄底 v11"] = "前 100 大，4h 崩跌到 30 天高點一半以下、量 6 倍、單根跌 >10% 後 12 小時內出錘子線就買；BTC 在 50 日線下；停損 −15%，+15% 後拉到 +7.5%，最多 72 小時"
STR["抄底 v11（要多空比）"] = book([r for r in ev["DIP"] if r.get("acc", 9) <= 2.0]); FAM["抄底 v11（要多空比）"] = "抄底"
DESC["抄底 v11（要多空比）"] = DESC["抄底 v11"] + "；散戶多空比 ≤2"
NM = list(STR); M = np.vstack([STR[k] for k in NM]).T
print("strategies", len(NM), flush=True)


def maxdd(r):
    v = np.cumprod(1 + r); return float((1 - v / np.maximum.accumulate(v)).max())


def plan_for(a, b):
    t0 = int((pd.Timestamp(a, unit="ms", tz="UTC") - pd.DateOffset(months=12)).timestamp() * 1000)
    tr = (days >= t0) & (days < a)
    best = {}
    for k, nm in enumerate(NM):
        v = M[tr, k]
        if (v != 0).sum() < 20: continue
        ret = float(np.prod(1 + v) - 1)
        if ret <= 0: continue
        dd = max(maxdd(v), .01); cal = ret / dd
        f = FAM[nm]
        if f not in best or cal > best[f][0]: best[f] = (cal, k, ret, dd)
    pick = sorted(best.values(), reverse=True)[:3]
    if not pick: return [], 0.0, {}
    w = np.array([p[2] for p in pick]); w = w / w.sum()
    mix = M[tr][:, [p[1] for p in pick]] @ w
    lo, hi = 0.1, 10.0
    for _ in range(40):                                   # largest scale with a 12-month fall <= 35 %
        mid = (lo + hi) / 2
        if maxdd(np.clip(mid * mix, -.99, None)) <= .35: lo = mid
        else: hi = mid
    return [(p[1], float(wi), p[2], p[3]) for p, wi in zip(pick, w)], lo, {"train_mix_ret": float(np.prod(1 + lo * mix) - 1)}


res = {}
for P in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3").split(",")]:
    q = pd.Timestamp("2021-01-01", tz="UTC"); eq = 500.; curve = []; rows = []; peak = 500.; mdd = 0.
    while q < pd.Timestamp("2026-11-01", tz="UTC"):
        q2 = min(q + pd.DateOffset(months=P), pd.Timestamp("2026-11-01", tz="UTC"))
        a, b = int(q.timestamp() * 1000), int(q2.timestamp() * 1000)
        pick, k, info = plan_for(a, b)
        te = (days >= a) & (days < b); start = eq
        if pick:
            r = np.clip(k * (M[te][:, [p[0] for p in pick]] @ np.array([p[1] for p in pick])), -.99, None)
        else:
            r = np.zeros(te.sum())
        for x in r:
            eq *= 1 + x; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak); curve.append(eq)
        rows.append({"期": f"{q:%Y-%m}", "選": [f"{NM[i]}（權重 {w:.0%}，前12月 {ret:+.0%}／回落 {dd:.0%}）" for i, w, ret, dd in pick] or ["空手"],
                     "放大": round(k, 2), "這期": f"{eq / start - 1:+.1%}", "帳戶": round(eq)})
        q = q2
    yr = {}
    for t, e in zip(days[(days >= int(pd.Timestamp('2021-01-01', tz='UTC').timestamp() * 1000))][:len(curve)], curve):
        yr[pd.Timestamp(t, unit="ms").year] = round(e)
    print(f"\n######## 每 {P} 個月重擬：500U → {eq:,.0f}U，最大回落 {mdd:.0%}，各年底 {yr}", flush=True)
    for x in rows:
        print(f"{x['期']} 放大{x['放大']}倍 → 這期 {x['這期']}，帳戶 {x['帳戶']:,}U｜" + "；".join(x["選"]), flush=True)
    res[P] = {"end": round(eq), "max_dd": round(mdd, 3), "year_end": yr, "periods": rows}
used = sorted({s.split("（")[0] for P in res for x in res[P]["periods"] for s in x["選"] if s != "空手"})
print("\n=== 用到的策略內容 ===")
for u in used: print(f"・{u}：{DESC.get(u, '')}")
import json
json.dump({"results": res, "desc": {u: DESC.get(u, "") for u in used}}, open("dynamic_wf.json", "w"), ensure_ascii=False, indent=1, default=str)
