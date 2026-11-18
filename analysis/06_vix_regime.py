"""Macro regimes: same-week association vs. predictability.

    python analysis/06_vix_regime.py <yahoo|lseg>     (after 01_rules_144.py)

Weekly return (pp) of each key rule regressed on four regime dummies (VIX > 25, NBER
recession, term spread below 1 pp, credit spread above its sample 75th percentile), HAC
Newey-West with 5 lags, with the regimes measured in the SAME week (lag 0) and in the
PREVIOUS week (lag 1). Only the lag-1 regression is a predictive test.
Output: results/<base>/vix_regime.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PUBLIC, out_dir            # noqa: E402

REG = ['VIX_HIGH', 'NBER', 'TERM_LOW', 'CREDIT_HIGH']
RULES = ['BASELINE', 'TSL07', 'TSL10', 'ATR4', 'MA_SMA200_OV_TP100', 'TP100']

if __name__ == '__main__':
    base = sys.argv[1]; out = out_dir(base)
    macro = pd.read_parquet(PUBLIC / 'macro_regimes_weekly.parquet')
    eq = pd.read_csv(out / 'rules_144_equity.csv', index_col=0, parse_dates=True)
    rows = []
    for lag in [0, 1]:
        X0 = macro[REG].shift(lag)
        for c in RULES:
            r = eq[c].pct_change().dropna()
            X = X0.reindex(r.index).dropna(); y = r.reindex(X.index) * 100
            o = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            on = X['VIX_HIGH'].astype(bool)
            d = dict(rule=c, lag_weeks=lag, n=int(o.nobs), R2=o.rsquared, weeks_vix_high=int(on.sum()),
                     sharpe_vix_high=np.sqrt(52) * y[on].mean() / y[on].std(),
                     sharpe_vix_normal=np.sqrt(52) * y[~on].mean() / y[~on].std())
            for k in ['const'] + REG:
                d[f'{k}_b'] = o.params[k]; d[f'{k}_t'] = o.tvalues[k]
            rows.append(d)
    df = pd.DataFrame(rows); df.to_csv(out / 'vix_regime.csv', index=False)
    print(df[['rule', 'lag_weeks', 'VIX_HIGH_b', 'VIX_HIGH_t', 'sharpe_vix_high', 'sharpe_vix_normal']]
          .round(3).to_string(index=False))
