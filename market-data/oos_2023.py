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
# ---- 1) range breakout + resonance
BRK = []
for s, b in B.items():
    ot = b[:, 0]; o, h, l, c, q = b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5]
    for i in range(240 + 4 + 24 * 50, len(b) - 1):
        t = int(ot[i]) + H
        if t < SIG_A or t >= SIG_B: continue
        if s not in TOP.get(t - t % H4, ()): continue
        hi = h[i - 240:i].max(); lo = l[i - 240:i].min()
        if hi / lo - 1 > .15: continue
        up = c[i] > hi and c[i - 1] <= hi; dn = c[i] < lo and c[i - 1] >= lo
        if not (up or dn): continue
        q4 = q[i - 3:i + 1].sum(); med = np.median(q[i - 240:i].reshape(-1, 4).sum(1))
        if q4 < 2 * med: continue
        side = 1 if up else -1; e = o[i + 1]; stop = lo if up else hi
        d = (e - stop) / e * side
        if d <= .005 or d > .25: continue
        ma50 = c[i - 24 * 49:i + 1:24].mean() if False else c[i::-24][:50].mean()
        a = acc_at(s, t)
        reso = bool(((a <= 1.0) if up else (a >= 2.0)) and ((c[i] > ma50) if up else (c[i] < ma50)))
        BRK.append({'s': s, 't': t, 'side': side, 'd': float(d), 'e1h': float(e), 'i': i, 'reso': reso, 'acc': float(a)})
print('breakouts', len(BRK), 'reso', sum(r['reso'] for r in BRK), flush=True)
def path1h(b, k, e, side, d):
    sl = e * (1 - d * side); best = e; hourly = []
    for j in range(k, min(len(b), k + 720)):
        o_, h_, l_, c_ = b[j, 1:5]
        if side > 0:
            if l_ <= sl: return min(o_, sl), int(b[j, 0]) + H, hourly
            best = max(best, h_); sl = max(sl, best * (1 - d))
        else:
            if h_ >= sl: return max(o_, sl), int(b[j, 0]) + H, hourly
            best = min(best, l_); sl = min(sl, best * (1 + d))
        hourly.append((int(b[j, 0]) + H, (c_ / e - 1) * side / d))
    j = min(len(b) - 1, k + 719); return b[j, 4], int(b[j, 0]) + H, hourly
for r in BRK:
    b = B[r['s']]; x, xt, hourly = path1h(b, r['i'] + 1, r['e1h'], r['side'], r['d'])
    r.update(xt=xt, hourly=hourly, R=((x / r['e1h'] - 1) * r['side'] - .004 - fund(r['s'], r['t'], xt) * r['side']) / r['d'])
if USE1M:   # 1m exit for resonance trades: entry close of 2nd minute, trail by d, max 30 days (X0)
    by = {}
    for r in BRK:
        if r['reso']: by.setdefault(datetime_m := dt.datetime.fromtimestamp(r['t'] / 1000, dt.UTC).strftime('%Y-%m'), []).append(r)
    for m, rs in sorted(by.items()):
        ss = sorted({r['s'] for r in rs})
        m1 = load('klines_1m', m, nx(m), symbols=ss, columns=['symbol', 'open_time', 'open', 'high', 'low', 'close'])
        G = {s: g.sort_values('open_time')[['open_time', 'open', 'high', 'low', 'close']].to_numpy('float64') for s, g in m1.groupby('symbol')}; del m1
        for r in rs:
            a = G.get(r['s']); 
            if a is None: continue
            j0 = np.searchsorted(a[:, 0], r['t']); seg = a[j0:j0 + 30 * 1440 + 2]
            if len(seg) < 10: continue
            e = seg[1, 4]; side, d = r['side'], r['d']; sl = e * (1 - d * side); best = e; x = None
            for bb in seg[2:]:
                if bb[0] - seg[2, 0] >= 30 * D: x, xt = bb[1], int(bb[0]); break
                if (side > 0 and bb[3] <= sl) or (side < 0 and bb[2] >= sl):
                    x = min(bb[1], sl) if side > 0 else max(bb[1], sl); xt = int(bb[0]) + M; break
                best = max(best, bb[2]) if side > 0 else min(best, bb[3])
                sl = max(sl, best * (1 - d)) if side > 0 else min(sl, best * (1 + d))
            if x is None: x, xt = seg[-1, 4], int(seg[-1, 0]) + M
            r.update(R=((x / e - 1) * side - .004 - fund(r['s'], r['t'], xt) * side) / d, xt=xt,
                     hourly=[(hh, v) for hh, v in r['hourly'] if hh < xt])
        print('1m', m, len(rs), flush=True)
# ---- 2) DIP v11
symmeta = pd.read_csv('symbols.csv') if __import__('os').path.exists('symbols.csv') else None
first = dict(zip(symmeta.symbol, symmeta.first_open_time)) if symmeta is not None else {}
btc = B['BTCUSDT']
def hammer(o_, h_, l_, c_):
    rg = h_ - l_
    if rg <= 0: return False
    body = abs(c_ - o_); up_ = h_ - max(o_, c_); lw = min(o_, c_) - l_
    return lw >= 2 * body and up_ <= max(body, .15 * rg) and lw >= .5 * rg
DIP = []
for s, b in B.items():
    ot = b[:, 0]
    st4 = np.flatnonzero(ot % H4 == 0)                     # 4h blocks starting at UTC 4h boundary with 4 contiguous bars
    blocks = [j for j in st4 if j + 3 < len(b) and ot[j + 3] == ot[j] + 3 * H]
    if len(blocks) < 182: continue
    F = np.array([[ot[j], b[j, 1], b[j:j + 4, 2].max(), b[j:j + 4, 3].min(), b[j + 3, 4], b[j:j + 4, 5].sum()] for j in blocks])
    busy = 0
    for n in range(181, len(F)):
        T = int(F[n, 0]) + H4
        if T < SIG_A or T >= SIG_B or T < busy: continue
        if s not in TOP.get(T, ()): continue
        hi180 = F[n - 180:n, 2].max(); med = np.median(F[n - 30:n, 5]); pc = F[n - 1, 4]; cc = F[n, 4]
        if not (cc <= .5 * hi180 and med > 0 and F[n, 5] / med >= 6 and cc / pc - 1 <= -.10): continue
        if first.get(s, ot[0]) > T - 120 * D: continue
        stop0 = cc * .85; i = np.searchsorted(ot, T)
        for j in range(i, min(len(b), i + 12)):
            if ot[j] >= T + 12 * H: break
            if b[j, 3] <= stop0: break
            if hammer(*b[j, 1:5]):
                tb = int(ot[j]) + H
                bi = np.searchsorted(btc[:, 0], tb) - 1
                bma = btc[bi::-24, 4][:51].mean()
                if btc[bi, 4] >= bma: break
                if j < 720 or b[j, 4] / b[j - 720, 4] - 1 <= -.527: break
                a = acc_at(s, tb)
                if not (a <= 2.0): break
                e = b[j, 4]; sl = e * .85; lock = False; hourly = []; x = None
                for kk in range(j + 1, min(len(b), j + 73)):
                    if b[kk, 3] <= sl: x = min(b[kk, 1], sl); xt = int(b[kk, 0]) + H; break
                    if not lock and b[kk, 2] >= e * 1.15: lock = True; sl = e * 1.075
                    hourly.append((int(b[kk, 0]) + H, (b[kk, 4] / e - 1) / .15))
                if x is None: kk = min(len(b) - 1, j + 72); x = b[kk, 4]; xt = int(b[kk, 0]) + H
                DIP.append({'s': s, 't': tb, 'xt': xt, 'R': (x / e - 1 - .004 - fund(s, tb, xt)) / .15, 'hourly': hourly, 'kind': 'DIP'})
                busy = xt
                break
print('DIP', len(DIP), flush=True)
# ---- pool: 500U, max 30 open, one per coin, size = cash x risk at the stop, hourly mark-to-market
def pool(T, risk):
    ev = sorted([(r['t'], 1, n) for n, r in enumerate(T)] + [(r['xt'], 0, n) for n, r in enumerate(T)])
    hours = {}
    for n, r in enumerate(T):
        for hh, v in r['hourly']: hours.setdefault(hh, []).append((n, v))
    hs = sorted(hours); hi = 0; cash = 500.; opn = {}; peak = 500.; dd = 0.
    for tm, typ, n in ev:
        while hi < len(hs) and hs[hi] <= tm:
            eq = cash + sum(opn[m] * v for m, v in hours[hs[hi]] if m in opn); peak = max(peak, eq); dd = max(dd, 1 - eq / peak); hi += 1
        r = T[n]
        if typ == 0:
            if n in opn: cash += opn.pop(n) * r['R']; peak = max(peak, cash)
            continue
        if len(opn) >= 30 or any(T[m]['s'] == r['s'] for m in opn): continue
        opn[n] = cash * risk[r['kind']]
    return cash, dd
y = lambda t: dt.datetime.fromtimestamp(t / 1000, dt.UTC).year
for nm, rs in (('全部突破', BRK), ('共振突破', [r for r in BRK if r['reso']]), ('共振做多', [r for r in BRK if r['reso'] and r['side'] > 0]),
               ('共振做空', [r for r in BRK if r['reso'] and r['side'] < 0]), ('抄底', DIP)):
    for yy in sorted({y(r['t']) for r in rs}):
        v = np.sort([r['R'] for r in rs if y(r['t']) == yy])
        print(f"{nm} {yy}: {len(v)}筆 平均 {v.mean():+.2f}R 去3 {v[:-3].mean() if len(v) > 5 else float('nan'):+.2f}R 勝率 {(v > 0).mean():.0%}")
RS = [dict(r, kind='BRK') for r in BRK if r['reso']]
for nm, rk in (('共振 2% + 抄底 5%', {'BRK': .02, 'DIP': .05}), ('共振 3% + 抄底 10%', {'BRK': .03, 'DIP': .10}), ('只有共振 3%', {'BRK': .03, 'DIP': 0})):
    c_, dd_ = pool(RS + DIP, rk); print(f"{nm}: 500U → {c_:,.0f}U ({c_/500:.1f}倍) 最大回落 {dd_:.0%}")
json.dump({'brk': [{k_: v for k_, v in r.items() if k_ != 'hourly'} for r in BRK], 'dip': [{k_: v for k_, v in r.items() if k_ != 'hourly'} for r in DIP]},
          open(f'oos_{S0}_{S1}.json', 'w'))
