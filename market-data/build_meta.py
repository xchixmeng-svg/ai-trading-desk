"""Build the cross-month files from the monthly klines_1h assets (downloaded with load.py).
  symbols.csv        symbol, first_open_time, last_open_time, first_month, last_month, months
  top100_4h.parquet  every 4h UTC boundary ts: the 100 symbols with the largest 30-day quote volume (720 closed
                     1h bars ending at ts, the bar that just closed included; symbols with < 720 bars are skipped),
                     rank 0 = largest. This is the point-in-time universe used by the desk's research.
Usage: python build_meta.py 2023-10 2026-10
"""
import os
import sys

import numpy as np
import pandas as pd

from load import load, months

H = 3_600_000
MIN_BARS = int(os.environ.get("MIN_BARS", 720))


def main():
    a, b = sys.argv[1], sys.argv[2]
    k = load("klines_1h", a, b, columns=["symbol", "open_time", "quote_volume"])
    k = k.sort_values(["symbol", "open_time"])
    os.makedirs("out/meta", exist_ok=True)
    k["month"] = pd.to_datetime(k.open_time, unit="ms", utc=True).dt.strftime("%Y-%m")
    s = k.groupby("symbol").agg(first_open_time=("open_time", "min"), last_open_time=("open_time", "max"),
                                first_month=("month", "min"), last_month=("month", "max"), months=("month", "nunique"))
    s.reset_index().to_csv("out/meta/symbols.csv", index=False)
    t0 = (k.open_time.min() // (4 * H) + 1) * 4 * H
    t1 = k.open_time.max() + H
    grid = np.arange(t0, t1 + 1, 4 * H)
    rows = []
    for sym, g in k.groupby("symbol"):
        ot = g.open_time.to_numpy(); q = g.quote_volume.to_numpy()
        c = np.concatenate([[0.0], np.cumsum(q)])
        close = ot + H
        j = np.searchsorted(close, grid, side="right")               # bars closed at or before each boundary
        lo = np.searchsorted(close, grid - 720 * H, side="right")
        n = j - lo
        ok = (n >= MIN_BARS) & (j > 0)
        v = c[j] - c[lo]
        rows.append(pd.DataFrame({"ts": grid[ok], "symbol": sym, "vol30d": v[ok]}))
    r = pd.concat(rows)
    r["rank"] = r.groupby("ts").vol30d.rank(ascending=False, method="first").astype(int) - 1
    r = r[r["rank"] < 100].sort_values(["ts", "rank"])
    r.to_parquet("out/meta/top100_4h.parquet", compression="zstd", index=False)
    print(len(s), "symbols;", r.ts.nunique(), "4h boundaries")


if __name__ == "__main__":
    main()
