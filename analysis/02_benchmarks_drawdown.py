"""Key rules against passive benchmarks, with formal tests on the drawdown.

    python analysis/02_benchmarks_drawdown.py <yahoo|lseg>      (after 01_rules_144.py)

For BASELINE (no exit rule), the trailing-stop rules TSL07 and TSL10, ATR4,
MA_SMA200_OV_TP100, TP100 and the best of the 144 rules:
  - SPY (buy and hold) and a 60% SPY / 40% AGG blend (from AGG's first week, 2003-10);
  - a passive blend of SPY and T-bills with the SAME volatility as the rule (w = σ_rule/σ_SPY);
  - CAPM alpha against SPY (HAC Newey-West, 5 lags);
  - stationary bootstrap (mean block 12 weeks, B = 2000) of the MaxDD difference
    rule − SPY and rule − volatility-matched blend, and of the Sharpe difference
    rule − BASELINE.
Output: results/<base>/benchmarks_drawdown.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M                      # noqa: E402
from src.config import N_BOOT, SEED, out_dir      # noqa: E402
from src.universe import Setup                    # noqa: E402

KEY = ['BASELINE', 'TSL07', 'TSL10', 'ATR4', 'MA_SMA200_OV_TP100', 'TP100']

if __name__ == '__main__':
    base = sys.argv[1]; out = out_dir(base); S = Setup(base)
    eq = pd.read_csv(out / 'rules_144_equity.csv', index_col=0, parse_dates=True)
    best = pd.read_csv(out / 'rules_144_metrics.csv').sort_values('sharpe_std').iloc[-1]['code']
    rules = KEY + ([best] if best not in KEY else [])
    rf = S.rf; spy = S.spy
    agg = S.env['weekly']['AGG'].reindex(S.env['dates']).pct_change()
    b6040 = (0.6 * spy + 0.4 * agg).dropna()
    rng = np.random.default_rng(SEED)
    mdd = M.mdd_from_returns
    sharpe_w = lambda x: np.sqrt(52) * x.mean() / x.std(ddof=1)  # noqa: E731
    rows = []
    for c in rules:
        r = eq[c].pct_change().iloc[1:]
        rs = spy.reindex(r.index); f = rf.reindex(r.index)
        w = r.std() / rs.std(); mix = w * rs + (1 - w) * f
        s = M.stats_from_returns(r, rf); sm_ = M.stats_from_returns(mix, rf)
        o = sm.OLS(r - f, sm.add_constant(rs - f), missing='drop').fit(
            cov_type='HAC', cov_kwds={'maxlags': 5})
        c6 = r.loc[b6040.index[0]:]
        s6 = M.stats_from_returns(c6, rf); b6 = M.stats_from_returns(b6040.loc[c6.index], rf)
        d_spy, p_spy, ci_spy = M.paired_bootstrap(r.values, rs.values, mdd, rng, N_BOOT)
        d_mix, p_mix, ci_mix = M.paired_bootstrap(r.values, mix.values, mdd, rng, N_BOOT)
        rb = eq['BASELINE'].pct_change().iloc[1:]
        d_b, p_b, _ = M.paired_bootstrap((r - f).values, (rb - f).values, sharpe_w, rng, N_BOOT)
        rows.append(dict(
            rule=c, is_best_of_144=(c == best),
            sharpe_std=s['sharpe_std'], cagr=s['cagr'], vol=s['vol'], max_dd=s['max_dd'], CVaR_95=s['CVaR_95'],
            SPY_sharpe_std=M.stats_from_returns(rs, rf)['sharpe_std'], SPY_max_dd=mdd(rs.values) * 100,
            alpha_annual_pct=o.params['const'] * 52 * 100, t_alpha=o.tvalues['const'], beta=o.params.iloc[1],
            mix_weight_spy=w, mix_cagr=sm_['cagr'], mix_max_dd=sm_['max_dd'], mix_CVaR_95=sm_['CVaR_95'],
            dMaxDD_vs_SPY_pp=d_spy * 100, p_dMaxDD_vs_SPY=p_spy,
            dMaxDD_vs_mix_pp=d_mix * 100, p_dMaxDD_vs_mix=p_mix,
            dMaxDD_vs_mix_ci95_pp=f'[{ci_mix[0] * 100:.1f}; {ci_mix[1] * 100:.1f}]',
            dSharpe_std_vs_BASELINE=d_b, p_dSharpe_vs_BASELINE=p_b,
            sharpe_std_since_2003_10=s6['sharpe_std'], sharpe_std_6040=b6['sharpe_std'],
            max_dd_6040=b6['max_dd'], cagr_6040=b6['cagr']))
    df = pd.DataFrame(rows); df.to_csv(out / 'benchmarks_drawdown.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.round(3).T.to_string())
