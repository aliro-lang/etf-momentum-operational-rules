"""Precompute weekly signals from the daily price panel.

    python -m src.precompute <yahoo|lseg> [env] [signals]

Writes cache/<panel>/env.pkl (inputs of the 144 exit rules and of pure momentum) and
cache/<panel>/signals.pkl (the six entry conditions C1..C6 used in the entry-signal ablation).

Signals: Thursday closes stamped on the following Friday (`sig_w`); a composite momentum
score (26-, 13- and 4-week returns and distance to the 30-week average, ranked across ETFs);
conditions C1..C6; the technical half-exit (SP: price below its 50-day average) and full exit
(ST: 50-day average crosses below the 200-day average); moving-average, ATR and LOW-N stop
levels. No ETF is dropped here: the 40% volatility cap is applied at each entry with trailing
volatility (src/universe.py).
"""
import gc
import warnings
import pickle
import sys
import time

import numpy as np
import pandas as pd

from .config import PUBLIC, cache_dir, data_dir

warnings.filterwarnings("ignore", category=FutureWarning)

MIN_HIST = 210  # weeks of history before the first trading date


def _load(base):
    prices = pd.read_parquet(data_dir(base) / 'prices_daily.parquet')
    prices.index = pd.to_datetime(prices.index)
    weekly = prices.resample('W-FRI').last().dropna(how='all')
    # keep every ETF with a defined volatility (no full-sample volatility filter)
    vol = weekly.pct_change().std() * np.sqrt(52)
    tickers = vol[vol <= np.inf].index.tolist()
    return prices[tickers], weekly[tickers], tickers


def _dates(weekly):
    t0 = weekly.index[MIN_HIST]
    for tk in ['VTI', 'EFA']:
        if tk in weekly.columns:
            f = weekly[tk].first_valid_index()
            if f and f > t0:
                t0 = f
    dates = weekly.index[weekly.index >= t0]
    return dates, (dates[-1] - dates[0]).days / 365.25


def _cash(dates):
    tb = pd.read_parquet(PUBLIC / 'tb3m_weekly.parquet')
    return tb['r_cash_weekly'].reindex(dates).ffill().fillna(0.0)


def _av(df_d, weekly):
    """Daily → weekly: Thursday close, stamped on the following Friday."""
    t = df_d.resample('W-THU').last().dropna(how='all')
    t.index = t.index + pd.Timedelta(days=1)
    return t.reindex(weekly.index).ffill().reindex(columns=weekly.columns)


def _score(sig_w):
    ma30 = sig_w.rolling(30, min_periods=21).mean()
    return (sig_w.pct_change(26).rank(axis=1, pct=True).fillna(0) * 0.35 +
            sig_w.pct_change(13).rank(axis=1, pct=True).fillna(0) * 0.35 +
            ((sig_w - ma30) / ma30.clip(lower=1e-6)).rank(axis=1, pct=True).fillna(0) * 0.20 +
            sig_w.pct_change(4).rank(axis=1, pct=True).fillna(0) * 0.10)


def build_env(base):
    t0 = time.time()
    prices, weekly, tickers = _load(base)
    pd_f = prices.astype('float32'); del prices; gc.collect()
    weekly = weekly.astype('float32')
    returns = weekly.pct_change().fillna(0).astype('float32')
    eligible = (weekly.notna().cumsum() >= 52) & weekly.notna()
    sig_w = _av(pd_f, weekly).astype('float32')
    score_mom = _score(sig_w).astype('float32')
    sma50 = pd_f.rolling(50, min_periods=35).mean().astype('float32')
    sma200 = pd_f.rolling(200, min_periods=140).mean().astype('float32')
    max52 = pd_f.rolling(252, min_periods=200).max().shift(1).astype('float32')
    C2 = _av(sma50 > sma200, weekly).fillna(False)
    C6 = _av(pd_f > max52, weekly).fillna(False)
    MASK = C2 & C6
    SP = ~_av(pd_f > sma50, weekly).fillna(False)
    ST = (~C2) & (C2.shift(1).fillna(False))
    del max52, C6; gc.collect()
    MA_W = {'sma50': _av(sma50, weekly), 'sma200': _av(sma200, weekly),
            'ema5w': sig_w.ewm(span=5, min_periods=4).mean().astype('float32')}
    del sma50, sma200; gc.collect()
    MA_W['sma100'] = _av(pd_f.rolling(100, min_periods=70).mean().astype('float32'), weekly)
    del pd_f; gc.collect()
    ATR_W = sig_w.diff().abs().astype('float32').rolling(14, min_periods=10).mean().astype('float32')
    LOW_N = {n: sig_w.rolling(n, min_periods=int(n * 0.7)).min().shift(1).astype('float32')
             for n in [10, 20, 40]}
    dates, n_years = _dates(weekly)
    env = dict(weekly=weekly, returns=returns, eligible=eligible, score_mom=score_mom,
               MASK=MASK, SP=SP, ST=ST, sig_w=sig_w, MA_W=MA_W, ATR_W=ATR_W, LOW_N=LOW_N,
               dates=dates, n_years=n_years, tickers=tickers, r_cash=_cash(dates))
    p = cache_dir(base) / 'env.pkl'
    with open(p, 'wb') as f:
        pickle.dump(env, f)
    print(f'[{base}] env: {len(tickers)} ETFs, {dates[0].date()} → {dates[-1].date()}, '
          f'{p.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)


def build_signals(base):
    """Six entry conditions. float64 on the Yahoo panel (as in the article); float32 on the
    larger LSEG panel to fit in 8 GB of RAM."""
    t0 = time.time()
    dt = 'float32' if base == 'lseg' else 'float64'
    prices, weekly, tickers = _load(base)
    pd_f = prices.astype(dt); del prices; gc.collect()
    weekly = weekly.astype(dt)
    returns = weekly.pct_change().fillna(0)
    eligible = (weekly.notna().cumsum() >= 52) & weekly.notna()
    sig_w = _av(pd_f, weekly)
    score_mom = _score(sig_w)
    del sig_w; gc.collect()
    ma50 = pd_f.rolling(50, min_periods=35).mean()
    ma200 = pd_f.rolling(200, min_periods=140).mean()
    C = {'C1': _av(pd_f > ma50, weekly).fillna(False),
         'C2': _av(ma50 > ma200, weekly).fillna(False),
         'C3': _av(ma50 - ma50.shift(5) > 0, weekly).fillna(False),
         'C4': _av(ma200 - ma200.shift(5) > 0, weekly).fillna(False)}
    del ma50, ma200; gc.collect()
    hh = ((pd_f.rolling(10, min_periods=7).max() > pd_f.shift(10).rolling(10, min_periods=7).max()) &
          (pd_f.rolling(10, min_periods=7).min() > pd_f.shift(10).rolling(10, min_periods=7).min()))
    C['C5'] = _av(hh, weekly).fillna(False); del hh; gc.collect()
    C['C6'] = _av(pd_f > pd_f.rolling(252, min_periods=200).max().shift(1), weekly).fillna(False)
    del pd_f; gc.collect()
    SP = ~C['C1']
    ST = (~C['C2']) & (C['C2'].shift(1).fillna(False))
    dates, n_years = _dates(weekly)
    sig = dict(weekly=weekly, returns=returns, eligible=eligible, score_mom=score_mom, C=C,
               SP=SP, ST=ST, dates=dates, n_years=n_years, tickers=tickers,
               r_cash=_cash(dates))
    p = cache_dir(base) / 'signals.pkl'
    with open(p, 'wb') as f:
        pickle.dump(sig, f)
    print(f'[{base}] signals: {p.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    base = sys.argv[1]
    what = sys.argv[2:] or ['env', 'signals']
    if 'env' in what:
        build_env(base)
    if 'signals' in what:
        build_signals(base)
