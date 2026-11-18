# Data

| Folder | Content | Source | In the repository? |
|---|---|---|---|
| `public/` | `tb3m_weekly.parquet` (3-month T-bill, DTB3) and `macro_regimes_weekly.parquet` (VIX, NBER, term and credit spreads) | FRED, public | **Yes** (rebuild with `build_data/fred_series.py`) |
| `yahoo/` | `prices_daily.parquet`: daily adjusted closes of 3,645 U.S.-listed ETFs, 2001-2026 | Yahoo Finance | **No** (Yahoo's terms do not allow redistribution). Rebuild with `build_data/yahoo/01_download_data.py` |
| `yahoo/` | `etf_categories.parquet`, `etf_categories_raw.csv` (11 super-categories, leveraged/inverse flag), `tickers_etfdb.json` (ticker universe) | Public ETF metadata (yfinance, ETF DB) | **Yes** |
| `lseg/` | `prices_daily.parquet` (total-return index), `etf_categories.parquet`, `liquidity_coverage_weekly.parquet` (liquidity and coverage filters), `fund_status.parquet` (active/delisted, last trading day): 7,299 ETFs, 6,298 with data, 1,332 of them delisted before 2026 | LSEG Workspace, licensed | **No**. Rebuild with `build_data/lseg/` and an LSEG licence |
| `eikon/` | `portfolio_bid_weekly.csv`, `portfolio_ask_weekly.csv`: weekly bid and ask quotes, 128 ETFs, 2020-09 to 2026-05 | LSEG Eikon, licensed | **No** |

**Provenance of the original panel.** The article states that prices come from Refinitiv
DataStream. The panel it used (`yahoo/`) was in fact downloaded from Yahoo Finance by
`build_data/yahoo/01_download_data.py`. Against the LSEG panel, 3,514 of its 3,645 tickers match,
with a median daily return correlation of 0.9999. It contains only ETFs that still traded in
2026, so it is affected by survivorship bias. The LSEG panel corrects this.

## Rebuilding the panels

**Yahoo panel** (a fresh download will differ slightly from the 2026 snapshot used here, since Yahoo
revises adjusted prices and drops delisted tickers):
```bash
python build_data/yahoo/01_download_data.py          # → data/yahoo/prices_daily.parquet
python -m src.precompute yahoo env
python build_data/yahoo/30_etf_categories.py 0       # categories via yfinance (~40 min)
python build_data/yahoo/30_etf_categories.py 1
```

**LSEG panel** (requires LSEG Workspace running and `export LSEG_APP_KEY=...`):
```bash
python build_data/lseg/eikon/01_universe.py          # U.S. ETF universe, including inactive funds
python build_data/lseg/eikon/01b_build_universe.py
python build_data/lseg/eikon/02_history.py           # daily price, volume, total return (resumable)
python build_data/lseg/panel/03a_consolidate.py
python build_data/lseg/panel/03b_daily_panel.py
python build_data/lseg/panel/03c_weekly_panel.py    # filters (i)-(iii) with past data only
python build_data/lseg/eikon/04_metadata.py
python build_data/lseg/panel/03d_classification.py    # super-categories, leveraged/inverse
python build_data/lseg/panel/03e_validate_yahoo.py    # optional: comparison with the Yahoo panel
python build_data/lseg/to_engine_format.py           # → data/lseg/
```
