"""Pure momentum (replication of Tse, 2015) on the broad ETF universe.

    python analysis/03_pure_momentum.py <yahoo|lseg>

Cross-sectional ('cross') and time-series ('ts') momentum with 1, 3, 6 and 12-month
lookbacks, 8 ETFs, rebalanced every 2 weeks, 20 bp per side on the traded weight, no
technical filters and no exit rules. Same universe filters as the rest of the repository.
Output: results/<base>/pure_momentum.csv, pure_momentum_equity.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import engine as E                       # noqa: E402
from src import metrics as M                      # noqa: E402
from src.config import out_dir                    # noqa: E402
from src.universe import Setup                    # noqa: E402

LOOKBACK = {'1m': 4, '3m': 13, '6m': 26, '12m': 52}

if __name__ == '__main__':
    base = sys.argv[1]; out = out_dir(base); S = Setup(base)
    rows = []; eqs = {}
    for lb, k in LOOKBACK.items():
        for st in ['cross', 'ts']:
            eq, slots, n_tr = E.run_momentum(S.env, k, S.leveraged, strategy=st, mask_extra=S.mask)
            d = M.all_metrics(eq, S.env['r_cash'], trades_per_year={0: n_tr}, slots=slots,
                              n_years=S.env['n_years'])
            d.update(lookback=lb, strategy=st); rows.append(d); eqs[f'{st}_{lb}'] = eq
            print(f"{st:<5} {lb:<3} Sharpe={d['sharpe']:.3f} Sharpe_std={d['sharpe_std']:.3f} "
                  f"CAGR={d['cagr']:.2f} MaxDD={d['max_dd']:.1f}", flush=True)
    spy = S.spy.iloc[1:]
    rows.append(dict(lookback='-', strategy='SPY', sharpe=M.paper_sharpe_from_returns(spy),
                     **{k: v for k, v in M.stats_from_returns(spy, S.rf).items()}))
    pd.DataFrame(rows).to_csv(out / 'pure_momentum.csv', index=False)
    pd.DataFrame(eqs).to_csv(out / 'pure_momentum_equity.csv')
