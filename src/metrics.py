"""Performance metrics and inference.

Two Sharpe ratios are reported for every series:
  sharpe      : the article's definition, (CAGR - 3%) / annualised volatility
  sharpe_std  : standard definition, mean weekly excess return over the 3-month T-bill
                divided by its standard deviation, × √52, with the Lo (2002) 95% CI
                computed on the WEEKLY ratio and the weekly sample size.
"""
import numpy as np
import pandas as pd

from .config import BLOCK, N_SLOTS, RF_PAPER


def max_drawdown(eq):
    eq = pd.Series(eq)
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def paper_metrics(eq, trades_per_year=None, slots=None, n_years=None, rf=RF_PAPER, N=N_SLOTS):
    r = eq.pct_change().dropna(); n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (52 / n) - 1
    vol = r.std() * np.sqrt(52)
    pos, neg = r[r > 0], -r[r < 0]
    d = dict(sharpe=(cagr - rf) / vol if vol > 0 else 0.0, cagr=cagr * 100, vol=vol * 100,
             max_dd=max_drawdown(eq) * 100,
             CVaR_95=-r[r <= np.quantile(r, 0.05)].mean() * 100,
             Omega=pos.sum() / neg.sum(), skew=r.skew())
    if trades_per_year is not None and n_years:
        d['trades_yr'] = sum(trades_per_year.values()) / n_years
    if slots is not None:
        d['cash_avg_pct'] = (1 - slots.mean() / N) * 100
    return d


def standard_sharpe(eq, r_cash):
    r = eq.pct_change().dropna()
    ex = r - r_cash.reindex(r.index).fillna(0.0)
    n = len(ex); sr_w = ex.mean() / ex.std()
    se_w = np.sqrt((1 + 0.5 * sr_w ** 2) / n)
    return dict(sharpe_std=sr_w * np.sqrt(52),
                sharpe_std_lo=(sr_w - 1.96 * se_w) * np.sqrt(52),
                sharpe_std_hi=(sr_w + 1.96 * se_w) * np.sqrt(52))


def all_metrics(eq, r_cash, **kw):
    d = paper_metrics(eq, **kw)
    d.update(standard_sharpe(eq, r_cash))
    return d


def stats_from_returns(r, rf):
    """Summary of a weekly return series (used for benchmarks and blends)."""
    r = r.dropna(); eq = (1 + r).cumprod(); ex = r - rf.reindex(r.index).fillna(0)
    return dict(sharpe_std=np.sqrt(52) * ex.mean() / ex.std(),
                cagr=(eq.iloc[-1] ** (52 / len(r)) - 1) * 100,
                vol=r.std() * np.sqrt(52) * 100, max_dd=max_drawdown(eq) * 100,
                CVaR_95=-r[r <= r.quantile(.05)].mean() * 100)


def paper_sharpe_from_returns(r, rf=RF_PAPER):
    r = r.dropna(); eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (52 / (len(r) + 1)) - 1
    return (cagr - rf) / (r.std() * np.sqrt(52))


def stationary_bootstrap_index(n, rng, L=BLOCK):
    """Politis-Romano stationary bootstrap, geometric blocks with mean length L."""
    out = np.empty(n, dtype=np.int64); i = 0
    while i < n:
        s = rng.integers(0, n); l = rng.geometric(1 / L); k = min(l, n - i)
        out[i:i + k] = (s + np.arange(k)) % n; i += k
    return out


def reality_check(EX, es, rng, B):
    """White (2000) Reality Check: is the best of K rules better than the benchmark?

    EX : (n × K) weekly excess returns of the rules over the T-bill
    es : (n,)   weekly excess return of the benchmark
    Returns the bootstrap p-values for the difference in annualised Sharpe and in mean.
    """
    n = len(es)
    shs = lambda A: np.sqrt(52) * A.mean(0) / A.std(0, ddof=1)
    sh1 = lambda a: np.sqrt(52) * a.mean() / a.std(ddof=1)
    d_sh = shs(EX) - sh1(es); d_mu = (EX - es[:, None]).mean(0)
    V1 = np.empty(B); V2 = np.empty(B)
    for b in range(B):
        ix = stationary_bootstrap_index(n, rng)
        V1[b] = np.max(np.sqrt(n) * ((shs(EX[ix]) - sh1(es[ix])) - d_sh))
        V2[b] = np.max(np.sqrt(n) * ((EX[ix] - es[ix, None]).mean(0) - d_mu))
    return dict(best=int(np.argmax(d_sh)), dSharpe_best=float(d_sh.max()),
                p_RC_sharpe=float(np.mean(V1 >= np.sqrt(n) * d_sh.max())),
                p_RC_mean=float(np.mean(V2 >= np.sqrt(n) * d_mu.max())),
                n_rules_above_benchmark=int((d_sh > 0).sum()), benchmark_sharpe_std=float(sh1(es)))


def paired_bootstrap(r1, r2, fn, rng, B):
    """Difference fn(r1) - fn(r2) and its two-sided stationary-bootstrap p-value."""
    r1 = np.asarray(r1); r2 = np.asarray(r2)
    d0 = fn(r1) - fn(r2)
    ds = np.empty(B)
    for b in range(B):
        ix = stationary_bootstrap_index(len(r1), rng)
        ds[b] = fn(r1[ix]) - fn(r2[ix])
    return float(d0), float(np.mean(np.abs(ds - ds.mean()) >= abs(d0))), \
        (float(np.quantile(ds, .025)), float(np.quantile(ds, .975)))


def mdd_from_returns(r):
    eq = np.cumprod(1 + np.asarray(r))
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak).min()
