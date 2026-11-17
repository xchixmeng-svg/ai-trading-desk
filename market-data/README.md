# 共用行情資料（Chi / Claude / GPT 同一份）

幣安 USDT 永續合約，**全部合約、包含已下架的**，來源是幣安官方公開檔案庫 data.binance.vision。
從 2020-01 起，每月一個 GitHub Release（標籤 `m-YYYY-MM`），每月 3 號自動補上個月。
時間全部是 UTC 毫秒。

## 檔案

每月（`m-YYYY-MM`）：

| 檔案 | 內容 | 欄位 |
|---|---|---|
| `klines_1m_pNN.parquet` | 1 分鐘 K（依幣名分成幾份，每份 ≤120 個幣） | symbol, open_time, open, high, low, close, volume, quote_volume, taker_buy_quote, trades |
| `klines_5m.parquet` | 5 分鐘 K（由 1 分鐘合成） | 同上 + minutes（這根含幾分鐘） |
| `klines_1h.parquet` | 1 小時 K（由 1 分鐘合成） | 同上 |
| `metrics_5m.parquet` | 持倉資料，5 分鐘一筆 | symbol, ts, sum_open_interest（張數）, sum_open_interest_value（USDT）, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio（全部帳戶多空比）, sum_taker_long_short_vol_ratio（主動買/賣） |
| `funding.parquet` | 資金費率結算 | symbol, ts, interval_h, rate |
| `premium_1h.parquet` | 溢價指數 1 小時 K | symbol, open_time, open, high, low, close |
| `manifest.json` | 本月檔案清單、筆數、幣種 | |

跨月（`meta`）：

| 檔案 | 內容 |
|---|---|
| `symbols.csv` | 每個幣第一根、最後一根 K 的時間（判斷上市、下架） |
| `top100_4h.parquet` | 每 4 小時（UTC）當時 30 天成交額前 100 名：ts, symbol, vol30d, rank（0 = 最大） |

## 下載網址

`https://github.com/xchixmeng-svg/ai-trading-desk/releases/download/<標籤>/<檔名>`

例：`https://github.com/xchixmeng-svg/ai-trading-desk/releases/download/m-2025-10/klines_1h.parquet`

公開網址，不用登入、不算流量。

## Python

```python
# pip install pandas pyarrow ；把 load.py 放在同一個資料夾
from load import load, load_meta, resample
k = load("klines_1h", "2024-01", "2026-10")                  # 全部幣 1 小時 K
m = load("klines_1m", "2025-10", "2025-10", symbols=["BTCUSDT"])
oi = load("metrics_5m", "2025-10", "2025-10", symbols=["BTCUSDT"])
top = load_meta("top100_4h.parquet")
```

## 注意

* 2020-01～2022-07 是 2026-11-17 補的舊資料。幣安早期合約少：2020-01 只有 11 個幣，2020-09 約 60 個，2021 年底約 140 個。
* `metrics_5m`（持倉、多空比）幣安從 **2021-12** 才有完整資料；2020-09～2021-11 只有零星幾筆，等於沒有，2020-08 以前完全沒有。

* 2026-11-17 全部重建並檢查（`check_gaps.py`）：幣安月檔有時被截斷或整個月不見，缺的日子已用日檔補回（共 1,359 個「幣×日」，例如 2022-02-26～28 有 48 個幣）。剩下 14 段空洞都是幣安自己暫停交易、日檔也沒有 K 線，不是漏抓。
* 幣安檔案庫本身偶爾缺幾天的持倉資料，缺的就是缺，沒有補值。
* 含非加密貨幣的合約（股票、黃金等），需要時自己用名稱排除。
* 研究規則：結果要用完整資料、兩段時間分開驗證，並由另一個 AI 從這份資料獨立重算。
