"""Out-of-sample test of the fixed playbook (resonance breakout + DIP v11) from the shared dataset only.
usage: python oos.py WARM_START SIG_START SIG_END [1m]   e.g. 2022-08 2023-01 2023-12 1m
Rules frozen 2026-11-16 (see 討論.md 15:40 and atd/trading_core/dipmeme.py)."""
import sys, json, bisect, datetime as dt, numpy as np, pandas as pd
from load import load, months
H = 3600000; H4 = 4 * H; M = 60000; D = 24 * H
W0, S0, S1 = sys.argv[1:4]; USE1M = len(sys.argv) > 4
nx = lambda m: months(m, '2099-12')[1]
END_M = nx(nx(S1))                                         # two extra months for exits
ts_ = lambda m: int(pd.Timestamp(m + '-01', tz='UTC').timestamp() * 1000)
SIG_A, SIG_B = ts_(S0), ts_(nx(S1))
NC = set(open('noncrypto.txt').read().split())
k = load('klines_1h', W0, END_M, columns=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'quote_volume'])
k = k[~k.symbol.isin(NC)].sort_values(['symbol', 'open_time'])
B = {s: g[['open_time', 'open', 'high', 'low', 'close', 'quote_volume']].to_numpy('float64') for s, g in k.groupby('symbol')}
del k
print('symbols', len(B), flush=True)
# ---- point-in-time crypto top 100 by 30-day quote volume (720 1h bars ending at the 4h boundary)
T0 = ts_(W0); grid = np.arange(T0, ts_(END_M) + 32 * D, H)
syms = sorted(B); Q = np.zeros((len(grid), len(syms)), np.float32); P = np.zeros_like(Q)
for j, s in enumerate(syms):
    a = B[s]; ii = ((a[:, 0] - T0) // H).astype(int); ok = (ii >= 0) & (ii < len(grid))
    Q[ii[ok], j] = a[ok, 5]; P[ii[ok], j] = 1
cq = np.vstack([np.zeros((1, len(syms))), np.cumsum(Q, 0, dtype=np.float64)]); cp = np.vstack([np.zeros((1, len(syms))), np.cumsum(P, 0)])
TOP = {}
for t in range(T0 + 720 * H, int(grid[-1]), H4):
    r = (t - T0) // H; v = cq[r] - cq[r - 720]; n = cp[r] - cp[r - 720]
    v = np.where(n >= 720, v, -1); top = np.argsort(-v)[:100]
    TOP[t] = {syms[j] for j in top if v[j] > 0}
del Q, P, cq, cp
# ---- metrics (retail all-account L/S) and funding
UNI = sorted(set().union(*TOP.values()))
B = {s: B[s] for s in set(UNI) | {'BTCUSDT'}}
parts = {}
for m in months(W0, END_M):
    met = load('metrics_5m', m, m, symbols=UNI, columns=['symbol', 'ts', 'count_long_short_ratio', 'sum_open_interest_value', 'sum_taker_long_short_vol_ratio'])
    for s, g in met.groupby('symbol'): parts.setdefault(s, []).append((g.ts.to_numpy(np.int64), g[['count_long_short_ratio', 'sum_open_interest_value', 'sum_taker_long_short_vol_ratio']].to_numpy(np.float32)))
    del met
MET = {}
for s, v in parts.items():
    a = np.concatenate([x for x, _ in v]); b_ = np.concatenate([y_ for _, y_ in v]); o_ = np.argsort(a, kind='stable'); MET[s] = (a[o_], b_[o_])
del parts
fu = load('funding', W0, END_M, columns=['symbol', 'ts', 'rate'])
FU = {s: (g.ts.to_numpy(np.int64), g.rate.to_numpy(float)) for s, g in fu.sort_values('ts').groupby('symbol')}
def acc_at(s, t):
    if s not in MET: return np.nan
    a, v = MET[s]; i = np.searchsorted(a, t, side='right') - 1
    return v[i] if i >= 0 and t - a[i] <= 15 * M else np.nan
def fund(s, a_, b_):
    if s not in FU: return 0.0
    a, v = FU[s]; return float(v[np.searchsorted(a, a_, side='right'):np.searchsorted(a, b_, side='right')].sum())

def met_at(s, t):
    if s not in MET: return None
    a, v = MET[s]; i = np.searchsorted(a, t, side='right') - 1
    if i < 0 or t - a[i] > 15 * M: return None
    i0 = np.searchsorted(a, t - 24 * H, side='right') - 1; lo = np.searchsorted(a, t - H, side='right')
    return {'acc': float(v[i, 0]), 'oi24': float(v[i, 1] / v[i0, 1] - 1) if i0 >= 0 and v[i0, 1] > 0 else np.nan,
            'tk': float(np.nanmean(v[lo:i + 1, 2]))}
def ema(x, n):
    a = 2 / (n + 1); out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out
def ind(o, h, l, c, v):
    """indicator arrays per bar: macd line, hist, rsi14, K, D, vol ratio"""
    m = ema(c, 12) - ema(c, 26); hist = m - ema(m, 9)
    dlt = np.diff(c, prepend=c[0]); up = np.where(dlt > 0, dlt, 0); dn = np.where(dlt < 0, -dlt, 0)
    au = ema(up, 27); ad = ema(dn, 27); rsi = 100 - 100 / (1 + au / np.maximum(ad, 1e-12))   # Wilder 14 ~ ema 27
    n = len(c); K = np.full(n, 50.); D = np.full(n, 50.)
    for i in range(n):
        lo, hi = l[max(0, i - 8):i + 1].min(), h[max(0, i - 8):i + 1].max()
        rsv = (c[i] - lo) / (hi - lo) * 100 if hi > lo else 50
        K[i] = (2 * K[i - 1] + rsv) / 3 if i else rsv; D[i] = (2 * D[i - 1] + K[i]) / 3 if i else K[i]
    va = np.convolve(v, np.ones(20) / 20, 'full')[:n]; vr = v / np.r_[np.nan, va[:-1]]
    return m, hist, rsi, K, D, vr
def agg(b, k):
    """resample 1h array (ot,o,h,l,c,q) to k hours aligned to UTC multiples; returns rows with close time"""
    ot = b[:, 0]; grp = (ot // (k * H)).astype(np.int64)
    idx = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]]); ends = np.r_[idx[1:], len(b)]
    full = (ends - idx) == k
    o = b[idx, 1]; c = b[ends - 1, 4]
    h = np.maximum.reduceat(b[:, 2], idx); l = np.minimum.reduceat(b[:, 3], idx); q = np.add.reduceat(b[:, 5], idx)
    ct = grp[idx] * k * H + k * H
    return ct[full], o[full], h[full], l[full], c[full], q[full]
def feats(TF, t, side):
    """directional booleans at signal close t (trade side +1/-1)"""
    f = {}
    for nm, (ct, o, h, l, c, q, m, hist, rsi, K, D, vr) in TF.items():
        i = np.searchsorted(ct, t, side='right') - 1
        if i < 45: return None
        p = side > 0
        f[nm + ' MACD柱同向'] = (hist[i] > 0) == p
        f[nm + ' MACD柱轉強'] = (hist[i] > hist[i - 1]) == p
        f[nm + ' MACD線同向'] = (m[i] > 0) == p
        f[nm + ' RSI過熱(順向)'] = rsi[i] > 70 if p else rsi[i] < 30
        f[nm + ' RSI中段順向'] = (50 < rsi[i] <= 70) if p else (30 <= rsi[i] < 50)
        f[nm + ' KD同向'] = (K[i] > D[i]) == p
        f[nm + ' KD過熱(順向)'] = K[i] > 80 if p else K[i] < 20
        a, b_ = slice(i - 19, i + 1), slice(i - 39, i - 19)
        hh = h[a].max() > h[b_].max(); ll = l[a].min() < l[b_].min()
        bear_m = hh and hist[a].max() < hist[b_].max(); bull_m = ll and hist[a].min() > hist[b_].min()
        bear_r = hh and rsi[a].max() < rsi[b_].max(); bull_r = ll and rsi[a].min() > rsi[b_].min()
        bear_v = hh and q[a].sum() < q[b_].sum(); bull_v = ll and q[a].sum() < q[b_].sum()
        f[nm + ' MACD背離(逆向)'] = bear_m if p else bull_m
        f[nm + ' MACD背離(順向)'] = bull_m if p else bear_m
        f[nm + ' RSI背離(逆向)'] = bear_r if p else bull_r
        f[nm + ' RSI背離(順向)'] = bull_r if p else bear_r
        f[nm + ' 量價背離(逆向)'] = bear_v if p else bull_v
        f[nm + ' 量價背離(順向)'] = bull_v if p else bear_v
        f[nm + ' 放量'] = vr[i] >= 1.5
    return f
TFC = {}
def tf_for(s):
    if s not in TFC:
        b = B[s]; out = {}
        for nm, k in (('1h', 1), ('4h', 4), ('日', 24)):
            ct, o, h, l, c, q = (b[:, 0] + H, b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5]) if k == 1 else agg(b, k)
            if len(c) < 60: out = None; break
            out[nm] = (ct, o, h, l, c, q) + ind(o, h, l, c, q)
        TFC.clear(); TFC[s] = out
    return TFC[s]
def path(b, k, e, side, stop, lim, trail):
    d = (e - stop) / e * side
    if d <= .004 or d > .30: return None
    sl = stop; best = e
    for j in range(k, min(len(b), k + lim)):
        o_, h_, l_, c_ = b[j, 1:5]
        if side > 0:
            if l_ <= sl: x = min(o_, sl); break
            best = max(best, h_)
            if trail: sl = max(sl, best * (1 - d))
        else:
            if h_ >= sl: x = max(o_, sl); break
            best = min(best, l_)
            if trail: sl = min(sl, best * (1 + d))
    else:
        j = min(len(b) - 1, k + lim - 1); x = b[j, 4]
    return x, int(b[j, 0]) + H, d
ROWS = []
for s in sorted(B):
    b = B[s]; ot = b[:, 0]; o, h, l, c, q = b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5]
    sigs = []
    last_f = -1e18
    for i in range(1300, len(b) - 1):
        t = int(ot[i]) + H
        if t < SIG_A or t >= SIG_B or s not in TOP.get(t - t % H4, ()): continue
        hi = h[i - 240:i].max(); lo = l[i - 240:i].min()
        if hi / lo - 1 <= .15:
            up = c[i] > hi and c[i - 1] <= hi; dn = c[i] < lo and c[i - 1] >= lo
            if (up or dn) and q[i - 3:i + 1].sum() >= 2 * np.median(q[i - 240:i].reshape(-1, 4).sum(1)):
                sd = 1 if up else -1
                sigs.append(('突破', i, t, sd, lo if up else hi, 720, True))
        if c[i - 72:i].max() / l[i - 72:i].min() - 1 >= .5 and t - last_f >= 24 * H and h[i - 72:i].argmax() >= 36 \
                and c[i] < l[i - 6:i].min() and c[i - 1] >= l[i - 7:i - 1].min():
            last_f = t; sigs.append(('飆幣做空', i, t, -1, h[i - 12:i + 1].max(), 168, True))
    if not sigs: continue
    TF = tf_for(s)
    if TF is None: continue
    for fam, i, t, sd, stop, lim, tr in sigs:
        r = path(b, i + 1, o[i + 1], sd, stop, lim, tr)
        if r is None: continue
        x, xt, d = r; e = o[i + 1]
        R = ((x / e - 1) * sd - .004 - fund(s, t, xt) * sd) / d
        f = feats(TF, t, sd)
        if f is None: continue
        mt = met_at(s, t) or {}
        a = mt.get('acc', np.nan)
        ma50 = c[i::-24][:50].mean()
        f['散戶反方向'] = bool((a <= 1.0) if sd > 0 else (a >= 2.0))
        f['日線趨勢同向'] = bool((c[i] > ma50) == (sd > 0))
        f['未平倉24h增加'] = bool(mt.get('oi24', 0) > 0)
        f['主動買賣同向'] = bool((mt.get('tk', 1) > 1.02) if sd > 0 else (mt.get('tk', 1) < .98))
        fr_ = fund(s, t - 8 * H, t)
        f['資金費不擁擠'] = bool((fr_ <= .0001) if sd > 0 else (fr_ >= 0))
        ROWS.append({'fam': fam, 's': s, 't': t, 'side': sd, 'R': float(R), 'f': {k: bool(v) for k, v in f.items()}})
print('rows', len(ROWS), flush=True)
json.dump(ROWS, open(f'feat_{S0}_{S1}.json', 'w'))
