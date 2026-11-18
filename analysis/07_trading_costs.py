"""Quoted bid-ask spreads from LSEG compared with the 20 bp per-side trading cost.

    python analysis/07_trading_costs.py <yahoo|lseg>

Weekly bid and ask quotes (LSEG Eikon, licensed, data/eikon/) cover 128 ETFs from
2020-09-11 to 2026-05-15. Compared with the 20 bp per-side cost:
  (a) every ETF-week with a quote;
  (b) the trades of the TSL07 rule on dates with a quote.
The relevant cost per side is the HALF spread.
Output: results/<base>/trading_costs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import engine as E                       # noqa: E402
from src.config import DATA, out_dir              # noqa: E402
from src.universe import Setup                    # noqa: E402


def describe(half_bp):
    x = pd.Series(half_bp).dropna()
    return dict(n=len(x), mean=x.mean(), p25=x.quantile(.25), median=x.median(), p75=x.quantile(.75),
                p90=x.quantile(.90), p95=x.quantile(.95), share_at_or_below_20bp=(x <= 20).mean() * 100)


if __name__ == '__main__':
    base = sys.argv[1]; out = out_dir(base)
    ask = pd.read_csv(DATA / 'eikon' / 'portfolio_ask_weekly.csv', index_col=0, parse_dates=True)
    bid = pd.read_csv(DATA / 'eikon' / 'portfolio_bid_weekly.csv', index_col=0, parse_dates=True)
    half = ((ask - bid) / ((ask + bid) / 2) * 1e4 / 2)
    half = half.where(half > 0)
    S = Setup(base)
    bt = E.run(S.env, *E.RULE_BY_CODE['TSL07'], S.cat_map, S.leveraged, **S.kwargs())
    tr = bt.trades
    tr = tr[tr['date'] >= half.index.min()]
    pairs = [half.loc[f, t] for f, t in zip(tr['date'], tr['ticker'])
             if t in half.columns and f in half.index and pd.notna(half.loc[f, t])]
    rows = [dict(sample='all ETF-weeks with a quote', **describe(half.stack())),
            dict(sample='TSL07 trades with a quote', **describe(pairs),
                 trades_in_quote_period=len(tr), etfs_traded=tr['ticker'].nunique(),
                 etfs_traded_with_quotes=len(set(tr['ticker']) & set(half.columns)))]
    df = pd.DataFrame(rows); df.to_csv(out / 'trading_costs.csv', index=False)
    print(f"Quotes: {half.index.min().date()} → {half.index.max().date()}, {half.shape[1]} ETFs")
    print(df.round(2).T.to_string())
