"""Builds spot_1h.parquet (BTCUSDT, ETHUSDT spot 1h, 2020-01..2026-10) from data.binance.vision monthly files."""
import io, urllib.request, zipfile
import numpy as np, pandas as pd
rows = []
for s in ("BTCUSDT", "ETHUSDT"):
    for y in range(2020, 2027):
        for m in range(1, 13):
            if (y, m) > (2026, 10): break
            b = urllib.request.urlopen(f"https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{y}-{m:02d}.zip").read()
            z = zipfile.ZipFile(io.BytesIO(b)); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
            d = d[d[0].astype(str).str[0].str.isdigit()].iloc[:, :11].astype(float)
            d[0] = np.where(d[0] > 1e14, d[0] // 1000, d[0]).astype("int64")      # 2025+ files are in microseconds
            d.insert(0, "symbol", s); rows.append(d)
k = pd.concat(rows)
k.columns = ["symbol", "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "tb_base", "tb_quote"]
k.drop(columns="close_time").drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"]).to_parquet("spot_1h.parquet")
