"""Daily summary of metrics_5m (2021-12..2026-10): end-of-day open interest value, day means of the account /
top-trader long-short ratios and the taker buy/sell volume ratio.  Writes metrics_daily.parquet."""
import pandas as pd
from load import load, months
D = 86_400_000; parts = []
for m in months("2021-12", "2026-10"):
    x = load("metrics_5m", m, m, columns=["symbol", "ts", "sum_open_interest_value", "count_long_short_ratio",
                                          "count_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    if not len(x): continue
    x["day"] = x.ts // D * D; x = x.sort_values("ts")
    g = x.groupby(["symbol", "day"]).agg(oi=("sum_open_interest_value", "last"), acc=("count_long_short_ratio", "mean"),
                                         top=("count_toptrader_long_short_ratio", "mean"), taker=("sum_taker_long_short_vol_ratio", "mean"),
                                         n=("ts", "size")).reset_index()
    parts.append(g); print(m, len(g), flush=True)
pd.concat(parts).to_parquet("metrics_daily.parquet")
