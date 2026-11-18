"""Converts the LSEG panel (build_data/lseg/data/panel) into the engine input format:
  data/lseg/prices_daily.parquet               (wide, adj_price = total-return index)
  data/lseg/etf_categories.parquet             (ticker, yf_category, super_category, is_leveraged_or_inverse)
  data/lseg/liquidity_coverage_weekly.parquet  (ok_liq & ok_cov by week, filters (ii)-(iii))
  data/lseg/fund_status.parquet                (ticker, status = active/delisted, last_date = last day with a price)
Columns: LSEG ticker; if the ticker repeats across funds (symbol reuse), ticker|ISIN.

Usage: python build_data/lseg/to_engine_format.py [--out-dir DIR]    (default: data/lseg/)"""
import argparse
import numpy as np, pandas as pd
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'lseg')
args = ap.parse_args()
L = Path(__file__).resolve().parent / 'data' / 'panel'            # output of panel/03a…03d
O = args.out_dir; O.mkdir(parents=True, exist_ok=True)
F = pd.read_csv(L/'funds.csv', index_col=0); F = F[F.has_data]
tk = F.TickerSymbol.astype(str); dup = tk.duplicated(keep=False)
name = tk.where(~dup, tk + '|' + F.index.astype(str))
d = pd.read_parquet(L/'daily.parquet', columns=['fund','date','adj_price'])
d['fund'] = d['fund'].astype(str)
w = d.pivot(index='date', columns='fund', values='adj_price').astype('float32')
w.index = pd.DatetimeIndex(w.index).as_unit('ns'); w.index.name = 'date'
w.columns = [name.get(c, c) for c in w.columns]
w.to_parquet(O/'prices_daily.parquet'); print('prices', w.shape, w.index.min().date(), w.index.max().date())
C = pd.read_csv(L/'classification.csv'); C['ticker'] = C['fund'].map(name)
C = C.dropna(subset=['ticker'])
pd.DataFrame({'ticker': C.ticker, 'yf_category': C['Fund Type'], 'super_category': C.super_cat,
              'is_leveraged_or_inverse': C.lev_inv.astype(bool)}).to_parquet(O/'etf_categories.parquet', index=False)
print('categories', len(C), 'leveraged/inverse', int(C.lev_inv.sum()))
s = pd.read_parquet(L/'weekly.parquet', columns=['week','fund','ok_liq','ok_cov'])
s['fund'] = s['fund'].astype(str).map(name); s['ok'] = s.ok_liq & s.ok_cov
m = s.pivot_table(index='week', columns='fund', values='ok', aggfunc='max').fillna(False).astype(bool)
m.index = pd.DatetimeIndex(m.index).as_unit('ns'); m.to_parquet(O/'liquidity_coverage_weekly.parquet'); print('filters', m.shape)
e = pd.DataFrame({'ticker': name, 'status': F.fund_status, 'last_date': pd.to_datetime(F.last_date)})
e.to_parquet(O/'fund_status.parquet'); print(e.status.value_counts().to_dict())
