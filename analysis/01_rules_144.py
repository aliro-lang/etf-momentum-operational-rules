"""The 144 exit rules with the engine in src/engine.py, and White's Reality Check against SPY.

    python analysis/01_rules_144.py <yahoo|lseg>

Output (results/<base>/):
  rules_144_metrics.csv   one row per rule (both Sharpe definitions, CAGR, MaxDD, CVaR, ...)
  rules_144_equity.csv    weekly equity curve of every rule
  reality_check.csv       best rule vs SPY after correcting for the choice among 144
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import engine as E                       # noqa: E402
from src import metrics as M                      # noqa: E402
from src.config import N_BOOT, SEED, out_dir      # noqa: E402
from src.universe import Setup                    # noqa: E402

S = None


def _init(base):
    global S
    S = Setup(base)


def _one(rule):
    bt = E.run(S.env, *rule, S.cat_map, S.leveraged, **S.kwargs())
    d = M.all_metrics(bt.equity, S.env['r_cash'], trades_per_year=bt.trades_per_year,
                      slots=bt.slots, n_years=S.env['n_years'])
    d['code'] = E.code(*rule)
    d['n_delisted_closed'] = int((bt.exits['reason'] == 'DELISTED').sum()) if len(bt.exits) else 0
    return d, bt.equity


if __name__ == '__main__':
    base = sys.argv[1]; out = out_dir(base)
    with Pool(5, initializer=_init, initargs=(base,)) as p:
        res = p.map(_one, E.ALL_RULES)
    df = pd.DataFrame([r[0] for r in res]).sort_values('sharpe', ascending=False)
    eq = pd.DataFrame({r[0]['code']: r[1] for r in res})
    df.to_csv(out / 'rules_144_metrics.csv', index=False)
    eq.to_csv(out / 'rules_144_equity.csv')
    print(df.head(8)[['code', 'sharpe', 'sharpe_std', 'cagr', 'max_dd']].round(3).to_string(index=False))

    _init(base)
    R = eq.pct_change().iloc[1:]
    rf = S.rf.reindex(R.index); spy = S.spy.reindex(R.index)
    rc = M.reality_check(R.sub(rf, axis=0).values, (spy - rf).values,
                         np.random.default_rng(SEED), N_BOOT)
    rc['best_rule'] = eq.columns[rc.pop('best')]
    rc['SPY_sharpe_paper'] = M.paper_sharpe_from_returns(spy)
    pd.DataFrame([rc]).to_csv(out / 'reality_check.csv', index=False)
    print(rc)
