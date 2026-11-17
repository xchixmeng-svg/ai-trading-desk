"""report_final (2026-11-18): full report of the chosen setting from iter3.py
  rebuild on the 1st of every month, look back 12 months, rank by return / max fall (Calmar), best 1 per family,
  best 4, equal weight, leverage 10x / 5x / 2x when the account is < 10 % / 10-25 % / > 25 % below its high.
Writes: report_periods.csv, report_trades.csv, report_monthly.csv, report_summary.json (+ prints the checks)."""
import json, pickle
import numpy as np, pandas as pd
D = 86_400_000
M = np.load("final_streams.npy"); meta = json.load(open("final_streams.json")); NM, FAM = meta["names"], meta["fam"]
d = pd.read_parquet("daily.parquet"); NC = set(open("noncrypto.txt").read().split()); d = d[(d.n >= 20) & ~d.symbol.isin(NC)]
days = np.sort(d.day.unique()); assert len(days) == len(M)
dix = {int(t): k for k, t in enumerate(days)}
exec(open("iter3.py").read().split("START = ")[0].split("days = np.sort(d.day.unique())")[1])   # mdd_, picks
TIERS = (10, 5, 2)
START = pd.Timestamp("2021-01-01", tz="UTC"); END = pd.Timestamp("2026-11-01", tz="UTC")

# ---- strategy descriptions (same text as final_opt.py)
exec(open("dynamic_wf.py").read().split('FAM = {k: k for k in STR}')[0].split('ev = pickle.load')[1].split('\n', 1)[1])
def desc(nm):
    if nm in DESC: return DESC[nm]
    if nm.startswith("盤整突破"):
        _, dd_, ww, vv, fl, sd = nm.split(" ")
        L_ = int(dd_.replace("天", "")); W_ = ww.replace("寬", ""); V_ = vv.replace("量", "").replace("x", "")
        cond = {"A或B": "符合 A 或 B（A：4 小時 RSI 做多 >70／做空 <30，且日線量價背離；B：散戶多空比做多 ≤1／做空 ≥2，日 MACD 柱同向，日 RSI 做多 50～70／做空 30～50）",
                "只A": "符合 A（4 小時 RSI 做多 >70／做空 <30，且日線量價背離）", "全部": "不加 A／B 條件"}[fl]
        return (f"成交額前 100 大加密貨幣；過去 {L_} 天的最高價 ÷ 最低價 − 1 ≤ {W_}；1 小時 K 收盤突破區間上緣做多、跌破下緣做空"
                f"（{ {'多': '只做多', '空': '只做空', '多空': '多空都做'}[sd] }）；最近 4 小時成交額 ≥ 平常 {V_} 倍；{cond}；"
                "下一根 K 開盤進場，停損放區間另一邊，之後停損跟最好價格保持同樣距離往上（下）移，最多抱 30 天")
    if nm.startswith("抄底"):
        base = ("成交額前 100 大、上市滿 120 天；4 小時 K 收盤 ≤ 過去 30 天最高價的一半、這根跌 >10%、成交額 ≥ 前 30 根中位數 6 倍；"
                "之後 12 小時內第一根 1 小時錘子線收盤買進（中途先跌 15% 就取消）；BTC 在 50 日均線下、該幣 30 天跌幅 < 52.7%；"
                "停損 −15%，漲到 +15% 後停損拉到 +7.5%，最多抱 72 小時")
        return base + ("；另外散戶多空比 ≤ 2" if "要多空比" in nm else "")
    return ""

# ---- marked-to-market increments per trade, only for the trade strategies that get used
ev = pickle.load(open("event_trades_mtm.pkl", "rb"))
def trades_of(nm):
    if nm.startswith("抄底"): return ev["DIP"] if "要多空比" not in nm else [r for r in ev["DIP"] if r.get("acc", 9) <= 2.0]
    _, dd_, ww, vv, fl, sd = nm.split(" ")
    L_ = int(dd_.replace("天", "")) * 24; W_ = {"寬12%": .12, "寬15%": .15, "寬18%": .18}[ww]; V_ = float(vv.replace("量", "").replace("x", ""))
    side = {"多": 1, "空": -1, "多空": 0}[sd]; f = {"A或B": "AB", "只A": "A", "全部": "ALL"}[fl]
    return [r for r in ev["C"] if r["L"] == L_ and r["w"] <= W_ and r["vx"] >= V_ and (f == "ALL" or r["A"] or (f == "AB" and r["B"])) and (side == 0 or r["side"] == side)]
def incs(trs):
    out = []
    for r in trs:
        prev = 0.0; parts = {}
        for t, m in r["dm"]:
            k = dix.get(int(t) - D)
            if k is None or t >= r["xt"]: continue
            parts[k] = parts.get(k, 0) + (m - prev); prev = m
        k = dix.get(int((r["xt"] - 1) // D * D))
        if k is not None: parts[k] = parts.get(k, 0) + (r["R"] - prev)
        out.append(parts)
    return out

# ---- plan + detailed simulation
q = START; PL = []
while q < END:
    q2 = min(q + pd.DateOffset(months=1), END); a = int(q.timestamp() * 1000); PL.append((q, a, int(q2.timestamp() * 1000), picks(a, 12, "calmar", 1, 4))); q = q2
used = sorted({NM[k] for *_, sel in PL for _, k, _ in sel})
TR = {}
for nm in used:
    if FAM[nm] in ("盤整突破", "抄底"):
        trs = trades_of(nm); inc = incs(trs); bd = {}
        for ti, p in enumerate(inc):
            for kk, v in p.items(): bd.setdefault(kk, []).append((ti, v))
        chk = np.zeros(len(days))
        for p in inc:
            for kk, v in p.items(): chk[kk] += .01 * v
        assert np.allclose(chk, M[:, NM.index(nm)]), nm                # rebuilt trades == the stream that was optimised
        TR[nm] = (trs, bd)
eq = 500.; pa = 500.; curve = []; prow = []; contrib = {}; tl = {}; lev_by_day = []
for q, a, b, sel in PL:
    te = np.nonzero((days >= a) & (days < b))[0]; start = eq; levs = []
    for i in te:
        if sel:
            dd = 1 - eq / pa; lev = TIERS[0] if dd < .1 else (TIERS[1] if dd < .25 else TIERS[2]); levs.append(lev)
            n = len(sel); parts = []
            for _, k, sc in sel:
                f = lev / n * sc; pv = f * M[i, k]; parts.append(pv)
                contrib[(f"{q:%Y-%m}", NM[k])] = contrib.get((f"{q:%Y-%m}", NM[k]), 0) + eq * pv
                if NM[k] in TR:
                    for ti, v in TR[NM[k]][1].get(i, []):
                        key = (NM[k], ti)
                        if key not in tl: tl[key] = {"期": f"{q:%Y-%m}", "pnl": 0.0, "risk_pct": f * .01, "eq_at_entry": eq, "R_in": 0.0, "last": i}
                        tl[key]["pnl"] += eq * f * .01 * v; tl[key]["R_in"] += v; tl[key]["last"] = i
            eq *= 1 + max(sum(parts), -.99)
        pa = max(pa, eq); curve.append((int(days[i]), eq))
    prow.append({"期": f"{q:%Y-%m}", "期初": start, "期末": eq, "報酬": eq / start - 1, "平均放大": float(np.mean(levs)) if levs else 0,
                 "策略": [{"名稱": NM[k], "前12月分數": round(s, 2), "波動調整倍數": round(sc, 2), "每筆交易風險佔帳戶": f"{TIERS[0] / len(sel) * sc * .01:.2%}（最大放大時）" if FAM[NM[k]] in ("盤整突破", "抄底") else ""} for s, k, sc in sel]})
print(f"500U → {eq:,.0f}U", flush=True)

# ---- tables
cv = pd.DataFrame(curve, columns=["t", "nav"]); cv["d"] = pd.to_datetime(cv.t, unit="ms")
cv["peak"] = cv.nav.cummax(); cv["dd"] = 1 - cv.nav / cv.peak
mo = cv.set_index("d").nav.resample("ME").last(); mret = mo.pct_change(); mret.iloc[0] = mo.iloc[0] / 500 - 1
dr = cv.nav.pct_change().fillna(cv.nav.iloc[0] / 500 - 1)
yrs = (cv.d.iloc[-1] - cv.d.iloc[0]).days / 365.25
i_tr = cv.dd.idxmax(); pk_i = cv.nav[:i_tr + 1].idxmax()
under = (cv.dd > 0).astype(int); longest = int(max((len(s) for s in "".join(map(str, under)).split("0")), default=0))
rows = []
for (nm, ti), v in tl.items():
    r = TR[nm][0][ti]
    rows.append({"期": v["期"], "策略": nm, "幣": r["s"], "方向": "多" if r.get("side", 1) > 0 else "空",
                 "訊號時間(UTC)": pd.Timestamp(r["t"], unit="ms").strftime("%Y-%m-%d %H:%M"), "出場時間(UTC)": pd.Timestamp(r["xt"], unit="ms").strftime("%Y-%m-%d %H:%M"),
                 "停損距離": round(r["d"], 4), "整筆R": round(r["R"], 3), "R": round(v["R_in"], 3),
                 "提前結束": "是（下個月換掉策略，當天收盤平倉）" if abs(v["R_in"] - r["R"]) > 1e-6 else "", "風險佔帳戶": round(v["risk_pct"], 4), "進場時帳戶U": round(v["eq_at_entry"], 2), "賺賠U": round(v["pnl"], 2)})
T = pd.DataFrame(rows).sort_values("訊號時間(UTC)")
T.to_csv("report_trades.csv", index=False, encoding="utf-8-sig")
def tstats(x):
    w, l = x[x["賺賠U"] > 0], x[x["賺賠U"] <= 0]
    return {"筆數": len(x), "勝率": round(len(w) / max(len(x), 1), 3), "平均賺R": round(float(w.R.mean()), 2) if len(w) else None,
            "平均賠R": round(float(l.R.mean()), 2) if len(l) else None, "RR(平均賺÷平均賠)": round(float(w["賺賠U"].mean() / -l["賺賠U"].mean()), 2) if len(w) and len(l) and l["賺賠U"].sum() < 0 else None,
            "PF(總賺÷總賠)": round(float(w["賺賠U"].sum() / -l["賺賠U"].sum()), 2) if len(l) and l["賺賠U"].sum() < 0 else None,
            "每筆期望R": round(float(x.R.mean()), 3), "總賺賠U": round(float(x["賺賠U"].sum()), 0)}
seq = "".join("L" if v <= 0 else "W" for v in T.sort_values("出場時間(UTC)")["賺賠U"])
C_ = pd.DataFrame([{"期": k[0], "策略": k[1], "賺賠U": v} for k, v in contrib.items()])
summary = {
    "設定": "每月 1 號重擬；看前 12 個月；用「報酬 ÷ 最大回落」排名；每一類最多 1 個；取前 4 名平均分配；每個策略先調到年化波動 25%；帳戶離最高點 <10% 放大 10 倍、10～25% 放大 5 倍、>25% 放大 2 倍；成本：合約每邊 0.1%＋資金費（突破／抄底每筆來回 0.4%）",
    "期間": f"{cv.d.iloc[0]:%Y-%m-%d} ～ {cv.d.iloc[-1]:%Y-%m-%d}", "起始": 500, "期末": round(eq, 0), "總報酬": round(eq / 500 - 1, 2),
    "年化報酬(CAGR)": round((eq / 500) ** (1 / yrs) - 1, 3), "最大回落": round(float(cv.dd.max()), 3),
    "最大回落區間": f"{cv.d[pk_i]:%Y-%m-%d} 高點 {cv.nav[pk_i]:,.0f}U → {cv.d[i_tr]:%Y-%m-%d} 低點 {cv.nav[i_tr]:,.0f}U",
    "最長在水下(天)": longest, "Sharpe(日,年化)": round(float(dr.mean() / dr.std() * np.sqrt(365)), 2),
    "Sortino": round(float(dr.mean() / dr[dr < 0].std() * np.sqrt(365)), 2), "Calmar": round(((eq / 500) ** (1 / yrs) - 1) / float(cv.dd.max()), 2),
    "賺錢月份比例": round(float((mret > 0).mean()), 3), "最好月": f"{mret.idxmax():%Y-%m} {mret.max():+.1%}", "最差月": f"{mret.idxmin():%Y-%m} {mret.min():+.1%}",
    "各年報酬": {int(ix.year): round(float(v), 3) for ix, v in cv.set_index("d").nav.resample("YE").last().pipe(lambda s: s / s.shift(1).fillna(500) - 1).items()},
    "交易統計(突破＋抄底)": tstats(T), "最長連虧筆數": max((len(s) for s in seq.split("W")), default=0),
    "前10筆佔交易總獲利": round(float(T.nlargest(10, "賺賠U")["賺賠U"].sum() / T[T["賺賠U"] > 0]["賺賠U"].sum()), 3),
    "依策略類別賺賠U": {f: round(float(v), 0) for f, v in C_.assign(類=C_.策略.map(FAM)).groupby("類").賺賠U.sum().sort_values(ascending=False).items()},
    "各年交易統計": {int(y): tstats(g) for y, g in T.assign(年=T["訊號時間(UTC)"].str[:4]).groupby("年")},
}
json.dump(summary, open("report_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
P_ = []
for p in prow:
    cc = {r["策略"]: r["賺賠U"] for _, r in C_[C_.期 == p["期"]].iterrows()}
    P_.append({"期": p["期"], "期初U": round(p["期初"], 0), "期末U": round(p["期末"], 0), "報酬": f"{p['報酬']:+.1%}", "平均放大": round(p["平均放大"], 1),
               "策略與賺賠": "；".join(f"{s['名稱']}（{cc.get(s['名稱'], 0):+,.0f}U）" for s in p["策略"]) or "空手"})
pd.DataFrame(P_).to_csv("report_periods.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"月": [f"{x:%Y-%m}" for x in mret.index], "月報酬": mret.values, "月底帳戶U": mo.values}).to_csv("report_monthly.csv", index=False, encoding="utf-8-sig")
json.dump({nm: desc(nm) for nm in used}, open("report_rules.json", "w"), ensure_ascii=False, indent=1)
# ---- self checks
tot = C_["賺賠U"].sum()
print("檢查1 各策略賺賠合計 =", round(tot, 0), "；帳戶增加 =", round(eq - 500, 0))
ev_c = C_[C_.策略.map(FAM).isin(["盤整突破", "抄底"])]["賺賠U"].sum()
print("檢查2 突破＋抄底：策略層合計 =", round(ev_c, 0), "；逐筆合計 =", round(T["賺賠U"].sum(), 0))
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str)[:3000])
