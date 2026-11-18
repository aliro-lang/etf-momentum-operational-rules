"""STEP 3a — Joins the 8,760 downloaded windows into one daily table. Does NOT use LSEG.

One row per (requested RIC, day) with closing price, volume and daily total return (%).
Rows without price or total return are dropped: LSEG fills Volume=0 before the fund's launch.
Windows overlap only if LSEG returns dates outside the requested range; deduplicated by (ric, date).

Input:  build_data/lseg/data/eikon/history/lote_*.parquet  (raw long format: date, ric, campo = field, valor = value)
Output: build_data/lseg/data/panel/daily_raw.parquet  (date, ric, price, volume, tr_pct)
        build_data/lseg/data/eikon/03a_consolidate.log
"""
import sys, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eikon"))
from common import get_logger, ROOT

PANEL = ROOT / "data" / "panel"; PANEL.mkdir(exist_ok=True)
log = get_logger("03a_consolidate")
FIELDS = {"Price Close": "price", "Volume": "volume", "Total Return": "tr_pct"}

parts, n_long, vol_without_price = [], 0, 0
files = sorted(glob.glob(str(ROOT / "data" / "eikon" / "history" / "lote_*.parquet")))
for i, f in enumerate(files):
    d = pd.read_parquet(f)
    if d.empty:
        continue
    d = d.rename(columns={"campo": "field", "valor": "value"})   # raw column names of 02_history
    n_long += len(d)
    unknown = set(d["field"]) - set(FIELDS)
    if unknown:
        raise SystemExit(f"Unexpected field {unknown} in {f}")
    w = d.pivot_table(index=["ric", "date"], columns="field", values="value", aggfunc="first").rename(columns=FIELDS)
    w = w.reindex(columns=list(FIELDS.values()))
    none = w["price"].isna() & w["tr_pct"].isna()
    vol_without_price += int(none.sum())
    parts.append(w[~none].reset_index())
    if (i + 1) % 1000 == 0:
        log.info(f"  {i + 1}/{len(files)} files")

x = pd.concat(parts, ignore_index=True)
x["date"] = pd.to_datetime(x["date"])
dup = x.duplicated(["ric", "date"], keep=False)
if dup.any():
    # same date in two windows: they must agree; if not, report it and keep the first
    g = x[dup].groupby(["ric", "date"])[["price", "tr_pct"]].nunique(dropna=True)
    log.info(f"Dates repeated across windows: {int(dup.sum())} rows; with different values: {int((g > 1).any(axis=1).sum())}")
    x = x.drop_duplicates(["ric", "date"], keep="first")
x["ric"] = x["ric"].astype("category")
x = x.sort_values(["ric", "date"]).reset_index(drop=True)
x.to_parquet(PANEL / "daily_raw.parquet", index=False)
log.info(f"Rows in long format: {n_long:,} | volume-only rows (before launch or other) dropped: {vol_without_price:,}")
log.info(f"daily_raw: {len(x):,} rows, {x['ric'].nunique()} RICs, {x['date'].min().date()} → {x['date'].max().date()}")
log.info(f"With price: {x['price'].notna().sum():,} | with TR: {x['tr_pct'].notna().sum():,} | with volume: {x['volume'].notna().sum():,}")
