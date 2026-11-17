"""Completeness check of the shared 1h klines (anyone can re-run it).
usage: python check_gaps.py 2020-01 2026-10
Lists every hole inside a coin's life (missing 1h bars between its first and last bar) and the hours built from
fewer than 60 one-minute bars. A hole is only acceptable when Binance itself has no trading then (a halt or a
delist / relist); 2026-11-17 every remaining hole was checked against the daily archive."""
import sys
import pandas as pd
from load import load

H = 3_600_000
k = load("klines_1h", sys.argv[1], sys.argv[2], columns=["symbol", "open_time", "minutes"])
k = k.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
print("coins", k.symbol.nunique(), "1h bars", len(k), "hours with < 60 1m bars", int((k.minutes < 60).sum()))
rows = []
for s, g in k.groupby("symbol"):
    t = g.open_time.to_numpy()
    d = (t[1:] - t[:-1]) // H - 1
    for i in (d > 0).nonzero()[0]:
        rows.append((pd.to_datetime(t[i] + H, unit="ms", utc=True).strftime("%Y-%m-%d %H:00"), int(d[i]), s))
g = pd.DataFrame(rows, columns=["start_utc", "hours", "symbol"])
print("holes", len(g), "missing hours", int(g.hours.sum()) if len(g) else 0)
if len(g):
    print(g.sort_values(["start_utc", "symbol"]).to_string(index=False))
