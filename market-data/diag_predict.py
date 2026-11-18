"""Which past-12-month score predicts the next 2 months? (2026-11-18, Claude)"""
import numpy as np, pandas as pd
import discover_wf as d
from scipy.stats import spearmanr
P=[]; q=pd.Timestamp("2023-01-01",tz="UTC")
while q < pd.Timestamp("2026-09-01",tz="UTC"): P.append(q); q=q+pd.DateOffset(months=1)
rows=[]
for q in P:
    a1=int(q.timestamp()*1000); a0=int((q-pd.DateOffset(months=12)).timestamp()*1000); b1=int((q+pd.DateOffset(months=2)).timestamp()*1000)
    l3=int((q-pd.DateOffset(months=3)).timestamp()*1000)
    for r in d.search(a0,a1,keep=10):
        ex=r["g"][2]; t=d.T[r["idx"]]; R=r["R"]
        qs=[R[(t>=a0+k*(a1-a0)//4)&(t<a0+(k+1)*(a1-a0)//4)].sum() for k in range(4)]
        oi=d.match(r["g"],r["rule"],a1,b1); oR=d.O[oi,3*ex]
        rows.append(dict(p=f"{q:%Y-%m}",total=R.sum(),score=r["score"],mean=R.mean(),n=len(R),last3=R[t>=l3].sum(),last3m=R[t>=l3].mean() if (t>=l3).any() else np.nan,
            q4=sum(x>0 for x in qs),minq=min(qs),btc=any(c[0] in("btc_up","btc_r7") for c in r["rule"]),fam=r["g"][0],side=r["g"][1],ex=ex,
            on=len(oR),omean=oR.mean() if len(oR) else np.nan,osum=oR.sum()))
    print(q.strftime("%Y-%m"),len(rows),flush=True)
D=pd.DataFrame(rows); D.to_csv("diag_predict.csv",index=False)
X=D[D.on>=5]
print("rules with >=5 next-2m trades:",len(X)," avg next mean R",round(X.omean.mean(),3)," share positive",round((X.omean>0).mean(),2))
for c in ["total","score","mean","n","last3","last3m","q4","minq"]:
    rho=spearmanr(X[c],X.omean,nan_policy="omit").correlation
    top=X.sort_values(c,ascending=False).groupby("p").head(5)
    print(f"{c:7s} 與下期每筆平均R的相關 {rho:+.3f} | 每期取前5名的下期平均 {top.omean.mean():+.3f}R 正的比例 {(top.omean>0).mean():.2f}")
print("含BTC條件 vs 不含:",X.groupby("btc").omean.mean().round(3).to_dict())
print("by family:",X.groupby("fam").omean.mean().round(3).to_dict()); print("by exit:",X.groupby("ex").omean.mean().round(3).to_dict())
print("by side:",X.groupby("side").omean.mean().round(3).to_dict())
print("per period avg next mean R:");print(X.groupby("p").omean.mean().round(2).to_string())
