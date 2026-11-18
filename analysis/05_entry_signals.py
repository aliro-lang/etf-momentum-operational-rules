"""Entry-signal ablation: the 63 combinations of the six technical conditions C1..C6.

    python analysis/05_entry_signals.py <yahoo|lseg>

  C1 price > 50-day MA            C4 200-day MA slope > 0
  C2 50-day MA > 200-day MA       C5 higher highs and higher lows over 10 days
  C3 50-day MA slope > 0          C6 price > previous 52-week high

Each combination is used as the ONLY entry filter (no stop, no take-profit; technical
half-exit and exit as in the article), with the engine in src/engine.py.
Output: results/<base>/entry_signals_63.csv, entry_signals_63_equity.csv
"""
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import engine as E                       # noqa: E402
from src import metrics as M                      # noqa: E402
from src.config import N_BOOT, SEED, out_dir      # noqa: E402
from src.universe import Setup, load_env          # noqa: E402

SUBSETS = [list(c) for k in range(1, 7) for c in combinations(['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], k)]
S = None; SIG = None


def _init(base):
    global S, SIG
    S = Setup(base); SIG = load_env(base, 'signals')
    # signals, returns and technical exits of the ablation come from the six-condition panel
    S.env = dict(S.env, SP=SIG['SP'], ST=SIG['ST'], returns=SIG['returns'], score_mom=SIG['score_mom'])


def _one(conds):
    C = SIG['C']; el = C[conds[0]].copy()
    for c in conds[1:]:
        el = el & C[c]
    env = dict(S.env, eligible=el)
    allow = pd.DataFrame(True, index=el.index, columns=el.columns)
    bt = E.run(env, None, None, S.cat_map, S.leveraged, entry_mask=allow, **S.kwargs())
    d = M.all_metrics(bt.equity, S.env['r_cash'], trades_per_year=bt.trades_per_year,
                      slots=bt.slots, n_years=S.env['n_years'])
    d.update(code='+'.join(conds), k=len(conds))
    return d, bt.equity


if __name__ == '__main__':
    base = sys.argv[1]; out = out_dir(base)
    with Pool(4 if base == 'lseg' else 5, initializer=_init, initargs=(base,)) as p:
        res = p.map(_one, SUBSETS)
    df = pd.DataFrame([r[0] for r in res]).sort_values('sharpe', ascending=False)
    df.to_csv(out / 'entry_signals_63.csv', index=False)
    eq = pd.DataFrame({r[0]['code']: r[1] for r in res}); eq.to_csv(out / 'entry_signals_63_equity.csv')
    # Reality Check: best of the 63 entry filters against SPY
    _init(base)
    R = eq.pct_change().iloc[1:]; rf = S.rf.reindex(R.index); spy = S.spy.reindex(R.index)
    rc = M.reality_check(R.sub(rf, axis=0).values, (spy - rf).values, np.random.default_rng(SEED), N_BOOT)
    rc['best_rule'] = eq.columns[rc.pop('best')]
    pd.DataFrame([rc]).to_csv(out / 'entry_signals_63_reality_check.csv', index=False); print(rc)
    print(f"Sharpe (paper) {df.sharpe.min():.3f} … {df.sharpe.max():.3f} | "
          f"Sharpe (std) {df.sharpe_std.min():.3f} … {df.sharpe_std.max():.3f} | "
          f"MaxDD {df.max_dd.min():.1f} … {df.max_dd.max():.1f} | distinct Sharpe {df.sharpe.round(3).nunique()}")
