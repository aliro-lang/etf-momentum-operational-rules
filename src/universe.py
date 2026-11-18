"""Investable universe: categories, leveraged/inverse exclusions and entry filters.

Applied on both panels:
  - ex-ante volatility cap: an ETF can be bought on week f only if its trailing 52-week
    annualised volatility (at least 26 weeks of data) is at most 40%;
  - leveraged and inverse ETFs are never bought;
  - at most CAPS[category] positions per super-category.
Only on the LSEG panel, which has the required volume and coverage data:
  - the article's liquidity and coverage filters (ii)-(iii), computed with past data only;
  - positions in funds that stop trading are closed after their last trading day.
"""
import pickle

import numpy as np
import pandas as pd

from .config import VOL_MAX_EXANTE, cache_dir, data_dir


def load_env(base, which='env'):
    with open(cache_dir(base) / f'{which}.pkl', 'rb') as f:
        return pickle.load(f)


def load_categories(base):
    c = pd.read_parquet(data_dir(base) / 'etf_categories.parquet')
    return dict(zip(c['ticker'], c['super_category'])), set(c[c['is_leveraged_or_inverse']]['ticker'])


def vol_exante(weekly, cap=VOL_MAX_EXANTE):
    vol = weekly.pct_change(fill_method=None).rolling(52, min_periods=26).std() * np.sqrt(52)
    return vol <= cap


def entry_filter(base, weekly):
    """Boolean DataFrame (week × ETF) intersected with eligibility at every entry."""
    m = vol_exante(weekly)
    if base == 'lseg':
        f = pd.read_parquet(data_dir('lseg') / 'liquidity_coverage_weekly.parquet')
        m = m & f.reindex(index=weekly.index, columns=weekly.columns).fillna(False).astype(bool)
    return m


def fund_status(base):
    """(last trading day by ticker, set of funds still trading) — LSEG panel only."""
    if base != 'lseg':
        return None, None
    e = pd.read_parquet(data_dir('lseg') / 'fund_status.parquet').set_index('ticker')
    return e['last_date'].to_dict(), set(e.index[e['status'] == 'active'])


def survivors_only(weekly, alive):
    return pd.DataFrame(np.broadcast_to(weekly.columns.isin(list(alive)), weekly.shape),
                        index=weekly.index, columns=weekly.columns)


class Setup:
    """Everything a script needs to run the engine on one panel."""

    def __init__(self, base):
        self.base = base
        self.env = load_env(base)
        self.cat_map, self.leveraged = load_categories(base)
        self.mask = entry_filter(base, self.env['weekly'])
        self.delisted_after, self.alive = fund_status(base)
        f = self.env['dates']
        self.rf = self.env['r_cash'].reindex(f).fillna(0)
        self.spy = self.env['weekly']['SPY'].reindex(f).ffill().pct_change()

    def kwargs(self):
        return dict(mask_extra=self.mask, delisted_after=self.delisted_after)
