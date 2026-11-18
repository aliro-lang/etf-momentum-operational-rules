"""Survivorship bias: the LSEG panel with and without the ETFs that were delisted.

    python analysis/04_survivorship.py          (LSEG panel only)

Runs the key rules twice with the engine in src/engine.py: on the full LSEG universe (funds that
close are liquidated after their last trading day) and restricted to the funds that still
trade in 2026. Also reports how many positions were closed by a delisting.
Output: results/lseg/survivorship.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import engine as E                       # noqa: E402
from src import metrics as M                      # noqa: E402
from src.config import out_dir                    # noqa: E402
from src.universe import Setup, survivors_only    # noqa: E402

RULES = ['BASELINE', 'TSL07', 'TSL10', 'ATR4', 'MA_SMA200_OV_TP100', 'TP100']

if __name__ == '__main__':
    S = Setup('lseg'); out = out_dir('lseg')
    surv = survivors_only(S.env['weekly'], S.alive)
    n_funds = len(S.env['tickers']); n_alive = len(set(S.env['tickers']) & S.alive)
    rows = []
    for c in RULES:
        for universe, mask in [('all_funds', S.mask), ('survivors_only', S.mask & surv)]:
            bt = E.run(S.env, *E.RULE_BY_CODE[c], S.cat_map, S.leveraged, mask_extra=mask,
                       delisted_after=S.delisted_after)
            d = M.all_metrics(bt.equity, S.env['r_cash'], trades_per_year=bt.trades_per_year,
                              slots=bt.slots, n_years=S.env['n_years'])
            d.update(rule=c, universe=universe,
                     n_delisted_closed=int((bt.exits['reason'] == 'DELISTED').sum()) if len(bt.exits) else 0,
                     n_funds=n_funds if universe == 'all_funds' else n_alive)
            rows.append(d)
            print(f"{c:<20} {universe:<15} Sharpe={d['sharpe']:.3f} Sharpe_std={d['sharpe_std']:.3f} "
                  f"MaxDD={d['max_dd']:.1f} delisted_closed={d['n_delisted_closed']}", flush=True)
    df = pd.DataFrame(rows)
    w = df.pivot(index='rule', columns='universe', values='sharpe')
    print('\nSharpe change from dropping delisted funds:\n', (w['survivors_only'] - w['all_funds']).round(3))
    df.to_csv(out / 'survivorship.csv', index=False)
