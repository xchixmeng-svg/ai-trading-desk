"""Read the shared Binance USDT-perpetual dataset (Claude, GPT, Chi all use this same file).

    from load import load
    df = load("klines_1h", "2025-01", "2025-03")                       # all symbols
    df = load("klines_1m", "2025-10", "2025-10", symbols=["BTCUSDT"])  # one coin, 1-minute
    top = load_meta("top100_4h.parquet")

Datasets per month: klines_1m (split in parts), klines_5m, klines_1h, metrics_5m, funding, premium_1h.
Times are UTC milliseconds (open_time / ts). Files are cached in ./cache (delete it to free disk).
Only needs pandas + pyarrow; downloads over plain HTTPS from the public GitHub releases.
"""
import json
import os
import urllib.request

import pandas as pd

REPO = os.environ.get("MD_REPO", "xchixmeng-svg/ai-trading-desk")
BASE = f"https://github.com/{REPO}/releases/download"
CACHE = os.environ.get("MD_CACHE", "cache")


def months(a, b):
    y, m = map(int, a.split("-")); out = []
    while f"{y}-{m:02d}" <= b:
        out.append(f"{y}-{m:02d}"); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def fetch(tag, name):
    p = os.path.join(CACHE, tag, name)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{tag}/{name}", p + ".tmp")
        os.replace(p + ".tmp", p)
    return p


def manifest(month):
    return json.load(open(fetch(f"m-{month}", "manifest.json")))


def load(dataset, start, end, symbols=None, columns=None):
    """dataset: klines_1m | klines_5m | klines_1h | metrics_5m | funding | premium_1h; start/end 'YYYY-MM'."""
    parts = []
    want = set(symbols) if symbols else None
    for m in months(start, end):
        man = manifest(m)
        names = sorted(f for f in man["files"] if f.startswith(dataset + "_p") or f == dataset + ".parquet")
        for f in names:
            filt = [("symbol", "in", list(want))] if want else None
            d = pd.read_parquet(fetch(f"m-{m}", f), columns=columns, filters=filt)
            if "ts" in d and len(d) and d.ts.max() < 10 ** 12:      # metrics files built before 2026-11-16 store seconds
                d["ts"] = d.ts * 1000
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_meta(name):
    """symbols.csv or top100_4h.parquet"""
    p = fetch("meta", name)
    return pd.read_csv(p) if name.endswith(".csv") else pd.read_parquet(p)


def resample(k, minutes):
    """Resample a klines frame (one or more symbols) to a coarser bar size in minutes."""
    w = minutes * 60_000
    k = k.assign(open_time=k.open_time // w * w)
    return (k.sort_values(["symbol", "open_time"]).groupby(["symbol", "open_time"])
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                 volume=("volume", "sum"), quote_volume=("quote_volume", "sum")).reset_index())
