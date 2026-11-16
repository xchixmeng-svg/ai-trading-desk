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
    met = load('metrics_5m', m, m, symbols=UNI, columns=['symbol', 'ts', 'count_long_short_ratio'])
    for s, g in met.groupby('symbol'): parts.setdefault(s, []).append((g.ts.to_numpy(np.int64), g.count_long_short_ratio.to_numpy(np.float32)))
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

# ---- fakeout fade (pre-registered 2026-11-16 19:10): breakout per the base rule (volume variant V: with / without the
# 2x volume condition); within 12 h a 1h close back inside the range -> enter opposite at next open; stop = extreme
# since the breakout; exits: A target = far side of range, max 7 d; B trail by d, max 30 d. Cost 0.4 % + funding.
FADE = []
for s, b in B.items():
    ot = b[:, 0]; o, h, l, c, q = b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5]
    for i in range(240 + 4 + 24 * 50, len(b) - 14):
        t = int(ot[i]) + H
        if t < SIG_A or t >= SIG_B or s not in TOP.get(t - t % H4, ()): continue
        hi = h[i - 240:i].max(); lo = l[i - 240:i].min()
        if hi / lo - 1 > .15: continue
        up = c[i] > hi and c[i - 1] <= hi; dn = c[i] < lo and c[i - 1] >= lo
        if not (up or dn): continue
        vol = q[i - 3:i + 1].sum() >= 2 * np.median(q[i - 240:i].reshape(-1, 4).sum(1))
        ext = h[i] if up else l[i]
        for j in range(i + 1, i + 13):
            ext = max(ext, h[j]) if up else min(ext, l[j])
            back = c[j] < hi if up else c[j] > lo
            if not back: continue
            if j + 1 >= len(b): break
            side = -1 if up else 1; e = o[j + 1]; stop = ext
            d = (stop - e) / e if side < 0 else (e - stop) / e
            if d <= .005 or d > .25: break
            tgt = lo if up else hi
            for mode in ('A', 'B'):
                sl = stop; best = e; hourly = []; x = None
                lim = 7 * 24 if mode == 'A' else 30 * 24
                for k in range(j + 1, min(len(b), j + 1 + lim)):
                    oo, hh, ll, cc = b[k, 1:5]
                    if side > 0:
                        if ll <= sl: x = min(oo, sl); break
                        if mode == 'A' and hh >= tgt: x = max(oo, tgt); break
                        best = max(best, hh)
                        if mode == 'B': sl = max(sl, best * (1 - d))
                    else:
                        if hh >= sl: x = max(oo, sl); break
                        if mode == 'A' and ll <= tgt: x = min(oo, tgt); break
                        best = min(best, ll)
                        if mode == 'B': sl = min(sl, best * (1 + d))
                    hourly.append((int(b[k, 0]) + H, (cc / e - 1) * side / d))
                else:
                    k = min(len(b) - 1, j + lim); x = b[k, 4]
                xt = int(b[k, 0]) + H
                FADE.append({'s': s, 't': int(ot[j]) + H, 'side': side, 'd': float(d), 'mode': mode, 'vol': bool(vol),
                             'R': ((x / e - 1) * side - .004 - fund(s, int(ot[j]) + H, xt) * side) / d, 'xt': xt, 'hourly': hourly,
                             'kind': 'FADE', 'brk_t': t})
            break
print('fade signals', len(FADE) // 2, flush=True)
y = lambda t: dt.datetime.fromtimestamp(t / 1000, dt.UTC).year
for mode in ('A', 'B'):
    for vv, vn in ((None, '全部'), (True, '有量突破')):
        for sd, sn in ((0, '多空'), (1, '多'), (-1, '空')):
            rs = [r for r in FADE if r['mode'] == mode and (vv is None or r['vol']) and (sd == 0 or r['side'] == sd)]
            line = f"{mode} {vn} {sn}"
            for yy in sorted({y(r['t']) for r in rs}):
                v = np.sort([r['R'] for r in rs if y(r['t']) == yy])
                line += f" | {yy} {len(v)}筆 {v.mean():+.2f}R 去3 {v[:-3].mean():+.2f}"
            print(line)
json.dump(FADE, open(f'fade_{S0}_{S1}.json', 'w'))
