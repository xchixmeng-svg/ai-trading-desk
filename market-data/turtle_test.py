"""Turtle-style trend following on daily bars (UTC), crypto, point-in-time top 100. Pre-registered 2026-11-16 19:30.
S1: enter on a daily close above the prior 20-day high close (short: below 20-day low close), exit on a close beyond the
prior 10-day opposite extreme; S2: 55 / 20. Initial stop 2 x ATR(20) from the entry (next day open); R = move / (2 ATR).
Cost 0.4 % + funding. Signals 2023-01..2026-09."""
import numpy as np, pandas as pd, json, datetime as dt
from load import load, load_meta, months
NC = set(open('noncrypto.txt').read().split())
D = 86400000
parts = []
for m in months('2022-06' if False else '2022-08', '2026-10'):
    k = load('klines_1h', m, m, columns=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'quote_volume'])
    k = k[~k.symbol.isin(NC)]; k['day'] = k.open_time // D * D
    parts.append(k.groupby(['symbol', 'day']).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), n=('close', 'size')).reset_index())
    print(m, flush=True)
d = pd.concat(parts); d = d[d.n >= 20].sort_values(['symbol', 'day'])
top = load_meta('top100_4h.parquet'); top = top[~top.symbol.isin(NC)]
top['day'] = top.ts // D * D; TOP = top[top.ts % D == 0].groupby('day').symbol.apply(set).to_dict()
fu = load('funding', '2022-08', '2026-10', columns=['symbol', 'ts', 'rate'])
FU = {s: (g.ts.to_numpy(), g.rate.to_numpy()) for s, g in fu.sort_values('ts').groupby('symbol')}
def fund(s, a, b):
    if s not in FU: return 0.
    t, v = FU[s]; return v[np.searchsorted(t, a, 'right'):np.searchsorted(t, b, 'right')].sum()
A0, A1 = 1672531200000, 1790812800000   # 2023-01-01 .. 2026-10-01
out = []
for s, g in d.groupby('symbol'):
    day = g.day.to_numpy(); o, h, l, c = (g[x].to_numpy() for x in 'ohlc')
    if len(c) < 80: continue
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1]); atr = pd.Series(np.r_[np.nan, tr]).rolling(20).mean().to_numpy()
    for sys_, (ne, nx) in (('S1', (20, 10)), ('S2', (55, 20))):
        i = ne + 1; busy = 0
        while i < len(c) - 1:
            if day[i] + D < A0 or day[i] + D >= A1 or i < busy or np.isnan(atr[i]) or s not in TOP.get(day[i] + D, ()):
                i += 1; continue
            hi = c[i - ne:i].max(); lo = c[i - ne:i].min()
            side = 1 if c[i] > hi else -1 if c[i] < lo else 0
            if not side: i += 1; continue
            e = o[i + 1]; dd_ = 2 * atr[i]; sl = e - side * dd_; x = None
            for j in range(i + 1, len(c)):
                if (side > 0 and l[j] <= sl) or (side < 0 and h[j] >= sl):
                    x = min(o[j], sl) if side > 0 else max(o[j], sl); break
                if j - nx >= 0 and j > i + 1:
                    ex = c[j - nx:j].min() if side > 0 else c[j - nx:j].max()
                    if (side > 0 and c[j] < ex) or (side < 0 and c[j] > ex): x = c[j]; break
            if x is None: j = len(c) - 1; x = c[j]
            xt = int(day[j]) + D
            R = ((x / e - 1) * side - .004 - fund(s, int(day[i]) + D, xt) * side) / (dd_ / e)
            out.append({'sys': sys_, 's': s, 't': int(day[i]) + D, 'xt': xt, 'side': side, 'R': float(R), 'days': j - i, 'hourly': [(int(day[k]) + D, float((c[k] / e - 1) * side / (dd_ / e))) for k in range(i + 1, j)]})
            busy = j + 1; i = j + 1
json.dump(out, open('turtle.json', 'w'))
o_ = pd.DataFrame(out); o_['y'] = pd.to_datetime(o_.t, unit='ms').dt.year
for (sy, sd), g in o_.groupby(['sys', 'side']):
    line = f"{sy} {'多' if sd > 0 else '空'}"
    for y, gg in g.groupby('y'):
        v = np.sort(gg.R.to_numpy()); line += f" | {y} {len(v)}筆 {v.mean():+.2f}R 去3 {v[:-3].mean():+.2f} ≥10R {int((v >= 10).sum())}"
    print(line)
