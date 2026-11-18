"""When does v12 make / lose money? (2026-11-18, Claude, for Chi)
Market state at each v12 trade, using only data before the trade (previous completed UTC day / trades already closed)."""
import pickle, numpy as np, pandas as pd, datetime as dt
H=3_600_000; D=24*H
NC=set(open("noncrypto.txt").read().split())
d=pickle.load(open("event_trades_full.pkl","rb")); C=[r for r in d["C"] if r["s"] not in NC]; DIP=[r for r in d["DIP"] if r["s"] not in NC]
v12=[dict(r,kind="A" if r["A"] else "B") for r in C if r["L"]==240 and r["w"]<=.15 and r["vx"]>=2 and (r["A"] or r["B"])]+[dict(r,kind="DIP") for r in DIP]
v12.sort(key=lambda r:r["t"])
x=pd.read_parquet("daily.parquet"); x=x[~x.symbol.isin(NC)]
c=x.pivot(index="day",columns="symbol",values="c").sort_index(); q=x.pivot(index="day",columns="symbol",values="q").reindex(c.index)
q30=q.rolling(30,min_periods=30).sum(); rk=q30.rank(axis=1,ascending=False); top=rk<=100
ma50=c.rolling(50,min_periods=50).mean(); r30=c/c.shift(30)-1; r7=c/c.shift(7)-1
st=pd.DataFrame(index=c.index)
st["btc_up"]=(c.BTCUSDT>ma50.BTCUSDT).astype(float); st["btc_r30"]=r30.BTCUSDT; st["btc_r7"]=r7.BTCUSDT
st["breadth"]=((c>ma50)&top).sum(1)/top.sum(1).replace(0,np.nan)
st["alt_r30"]=r30.where(top).median(1); st["alt_vs_btc"]=st.alt_r30-st.btc_r30
qt=q.where(top).sum(1); st["vol_trend"]=qt.rolling(30).mean()/qt.rolling(180).mean()
st["btc_vol"]=np.log(c.BTCUSDT).diff().rolling(30).std()*np.sqrt(365)
st=st.shift(1)                                     # use the previous completed day only
# breakout market health: all breakout candidates (any range / width / volume) that closed in the last 30 days
ct=np.array([r["xt"] for r in C]); cR=np.array([r["R"] for r in C]); o=np.argsort(ct); ct,cR=ct[o],cR[o]; cs=np.r_[0,np.cumsum(cR)]
vt=np.array([r["xt"] for r in v12]); vR=np.array([r["R"] for r in v12]); o2=np.argsort(vt); vt,vR=vt[o2],vR[o2]; vs=np.r_[0,np.cumsum(vR)]
rows=[]
for r in v12:
    t=r["t"]; day=t-t%D; s_=st.loc[day] if day in st.index else None
    if s_ is None: continue
    i1=np.searchsorted(ct,t); i0=np.searchsorted(ct,t-30*D); j1=np.searchsorted(vt,t)
    rows.append(dict(t=t,s=r["s"],kind=r["kind"],R=r["R"],xt=r["xt"],d=r["d"],lev=r.get("lev"),**s_.to_dict(),
        brk30=(cs[i1]-cs[i0])/max(i1-i0,1), brk30n=i1-i0, v12_last20=vs[j1]-vs[max(0,j1-20)], v12_last50=vs[j1]-vs[max(0,j1-50)]))
X=pd.DataFrame(rows); X["day"]=pd.to_datetime(X.t,unit="ms")
X.to_pickle("regime_v12.pkl")
X["per"]=np.where(X.t<pd.Timestamp("2023-01-01").value//10**6,"2020-09～2022","2023～2026")
X=X[X.day>="2020-09-01"]
print("v12 trades:",X.groupby("per").R.agg(["count","mean","sum"]).round(2).to_string())
fit=X[X.per=="2023～2026"]
for f in ["btc_up","btc_r30","btc_r7","breadth","alt_r30","alt_vs_btc","vol_trend","btc_vol","brk30","v12_last20","v12_last50"]:
    if f=="btc_up": cuts=[.5]
    else: cuts=list(fit[f].quantile([1/3,2/3]).values)
    b=np.digitize(X[f],cuts); X["b"]=b
    out=X.groupby(["b","per"]).R.agg(["count","mean"]).unstack()
    print(f"\n{f}（切點 {', '.join(f'{v:.3g}' for v in cuts)}）"); print(out.round(2).to_string())
