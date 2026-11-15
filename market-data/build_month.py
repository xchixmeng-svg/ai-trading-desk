"""Build one month of Binance USDT-margined perpetual data into Parquet files.

Source: Binance's public archive (data.binance.vision), every USDT perpetual including delisted ones.
Output (folder out/<YYYY-MM>/), all times UTC milliseconds:
  klines_1m_pNN.parquet  1-minute bars, symbols split alphabetically into parts of <= 120 symbols
  klines_5m.parquet      5-minute bars resampled from the 1-minute bars
  klines_1h.parquet      1-hour bars resampled from the 1-minute bars
  metrics_5m.parquet     open interest, long/short ratios, taker buy/sell ratio (5-minute)
  funding.parquet        funding rate settlements
  premium_1h.parquet     premium index 1-hour bars (futures vs index)
  manifest.json          file list with row counts and symbols

Usage: python build_month.py 2025-10 [SYMBOL ...]
Without symbols the full list comes from the archive's S3 index (needs network access to that index).
"""
import concurrent.futures as cf
import calendar
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DV = "https://data.binance.vision/data/futures/um/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix="
PART = 120
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
         "taker_buy_volume", "taker_buy_quote", "ignore"]


def get(url, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        time.sleep(min(30, 2 ** i))
    raise RuntimeError(f"failed: {url}")


def symbols():
    out, marker, prefix = [], "", "data/futures/um/monthly/klines/"
    while True:
        x = get(S3 + prefix + (f"&marker={marker}" if marker else "")).decode()
        p = re.findall(r"<Prefix>" + re.escape(prefix) + r"([^<]+)/</Prefix>", x)
        out += p
        if "<IsTruncated>true" not in x:
            break
        marker = prefix + p[-1] + "/"
    return sorted(s for s in out if s.endswith("USDT") and "_" not in s)


def read_zip(b, names):
    z = zipfile.ZipFile(io.BytesIO(b))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)
    df = df[df[0].str[0].str.isdigit()]                      # drop the header row when present
    df.columns = names[:df.shape[1]]
    return df


def ms(x):
    x = pd.to_numeric(x).astype("int64")
    return x.where(x < 10 ** 14, x // 1000)                  # microsecond files -> milliseconds


def klines(sym, month):
    b = get(f"{DV}monthly/klines/{sym}/1m/{sym}-1m-{month}.zip")
    if not b:
        return None
    d = read_zip(b, KCOLS)
    out = pd.DataFrame({"symbol": sym, "open_time": ms(d.open_time)})
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"):
        out[c] = pd.to_numeric(d[c]).astype("float64")
    out["trades"] = pd.to_numeric(d.trades).astype("int64")
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def resample(k, minutes):
    g = k.open_time // (minutes * 60_000) * (minutes * 60_000)
    a = k.groupby(g).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                         volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
                         taker_buy_quote=("taker_buy_quote", "sum"), trades=("trades", "sum"),
                         minutes=("open", "size")).reset_index().rename(columns={"open_time": "open_time"})
    a.insert(0, "symbol", k.symbol.iloc[0])
    a = a.rename(columns={a.columns[1]: "open_time"})
    return a


def metrics(sym, month):
    y, m = map(int, month.split("-"))
    days = [f"{month}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    parts = []
    for day in days:
        b = get(f"{DV}daily/metrics/{sym}/{sym}-metrics-{day}.zip")
        if not b:
            continue
        z = zipfile.ZipFile(io.BytesIO(b))
        parts.append(pd.read_csv(z.open(z.namelist()[0])))
    if not parts:
        return None
    d = pd.concat(parts)
    d["ts"] = (pd.to_datetime(d.create_time, utc=True).astype("int64") // 10 ** 6).astype("int64")
    keep = ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    out = pd.DataFrame({"symbol": sym, "ts": d.ts})
    for c in keep:
        out[c] = pd.to_numeric(d[c], errors="coerce").astype("float64") if c in d else float("nan")
    return out.drop_duplicates("ts").sort_values("ts")


def funding(sym, month):
    b = get(f"{DV}monthly/fundingRate/{sym}/{sym}-fundingRate-{month}.zip")
    if not b:
        return None
    d = read_zip(b, ["calc_time", "funding_interval_hours", "last_funding_rate"])
    return pd.DataFrame({"symbol": sym, "ts": ms(d.calc_time),
                         "interval_h": pd.to_numeric(d.funding_interval_hours).astype("int64"),
                         "rate": pd.to_numeric(d.last_funding_rate).astype("float64")})


def premium(sym, month):
    b = get(f"{DV}monthly/premiumIndexKlines/{sym}/1h/{sym}-1h-{month}.zip")
    if not b:
        return None
    d = read_zip(b, KCOLS)
    out = pd.DataFrame({"symbol": sym, "open_time": ms(d.open_time)})
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(d[c]).astype("float64")
    return out


def one(sym, month):
    k = klines(sym, month)
    if k is None:
        return sym, None, None, None, None, None
    return sym, k, resample(k, 5), resample(k, 60), metrics(sym, month), (funding(sym, month), premium(sym, month))


def write(df, path):
    t = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(t, path, compression="zstd", compression_level=9, row_group_size=500_000)
    return len(df)


def main():
    month = sys.argv[1]
    syms = sys.argv[2:] or symbols()
    out = f"out/{month}"
    os.makedirs(out, exist_ok=True)
    k1, k5, k60, met, fund, prem = [], [], [], [], [], []
    with cf.ThreadPoolExecutor(16) as ex:
        for sym, k, a5, a60, m, fp in ex.map(lambda s: one(s, month), syms):
            if k is None:
                continue
            k1.append(k); k5.append(a5); k60.append(a60)
            if m is not None: met.append(m)
            if fp[0] is not None: fund.append(fp[0])
            if fp[1] is not None: prem.append(fp[1])
    have = sorted(x.symbol.iloc[0] for x in k1)
    files = {}
    k1 = sorted(k1, key=lambda x: x.symbol.iloc[0])
    for p in range(0, len(k1), PART):
        name = f"klines_1m_p{p // PART + 1:02d}.parquet"
        files[name] = write(pd.concat(k1[p:p + PART]), f"{out}/{name}")
    for name, lst in (("klines_5m.parquet", k5), ("klines_1h.parquet", k60), ("metrics_5m.parquet", met),
                      ("funding.parquet", fund), ("premium_1h.parquet", prem)):
        if lst:
            files[name] = write(pd.concat(lst), f"{out}/{name}")
    man = {"month": month, "symbols": have, "files": files,
           "sizes_mb": {f: round(os.path.getsize(f"{out}/{f}") / 1e6, 1) for f in files},
           "source": "https://data.binance.vision (Binance public archive), USDT-margined perpetuals incl. delisted",
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("month", "files", "sizes_mb")}), len(have), "symbols")


if __name__ == "__main__":
    main()
