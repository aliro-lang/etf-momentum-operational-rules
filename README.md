# Momentum in ETFs under simple operational rules: updated code

> **Notice.** After publication, we found an error in the code that changed the results of the
> article. For that reason we set up this repository with the updated tests. The error and its
> correction are described in [CORRECTION.md](CORRECTION.md).

**Article:** Magner, N. and Sanhueza, A. *Momentum Strategies in ETFs under Simple Operational
Rules: Economic Value and Conditional Predictability.* North American Journal of Economics and
Finance. DOI: _to be added_.

## What the code does

The code backtests a momentum strategy on U.S.-listed ETFs, weekly, from August 2001 to May 2026:

- **Portfolio:** 8 positions of 1/8 of the capital each. Every two weeks, free slots are filled with
  the ETFs that have the highest composite momentum score (26-, 13- and 4-week returns and the
  distance to the 30-week average).
- **Entry filter:** a combination of six technical conditions (C1–C6). The default is C2 ∧ C6:
  50-day average above the 200-day average, and price above the previous 52-week high.
- **Exit rules:** 144 rules, made of 23 stop-losses (fixed, trailing, LOW-N, ATR and moving-average
  stops) crossed with 5 take-profit levels, plus a technical half-exit and full exit.
- **Constraints:** caps per asset class, no leveraged or inverse ETFs, and a 40% trailing-volatility
  cap at entry.
- **Costs and cash:** 20 bp per side on the traded weight; uninvested cash earns the 3-month T-bill
  rate.

Every test can be run on either of two price panels:

| Panel | ETFs | Delisted funds |
|---|---|---|
| `yahoo`: the panel used in the article (Yahoo Finance) | 3,645 | Not included |
| `lseg`: LSEG Workspace | 6,204 with enough history | 1,332 included; open positions are closed after the last trading day |

## Tests

| Script | What it computes | Output in `results/<panel>/` |
|---|---|---|
| `analysis/01_rules_144.py` | The 144 exit rules: Sharpe (two definitions), CAGR, volatility, MaxDD, CVaR, Omega, trades per year, average cash. White (2000) Reality Check of the best rule against SPY | `rules_144_metrics.csv`, `rules_144_equity.csv`, `reality_check.csv` |
| `analysis/02_benchmarks_drawdown.py` | Key rules against SPY, a 60/40 SPY–AGG portfolio and a SPY + T-bill blend with the same volatility. CAPM alpha (Newey-West). Bootstrap tests of the MaxDD difference against SPY and against the blend, and of the Sharpe difference against the rule without exit rules | `benchmarks_drawdown.csv` |
| `analysis/03_pure_momentum.py` | Cross-sectional and time-series momentum with 1-, 3-, 6- and 12-month lookbacks, without technical filters or exit rules (as in Tse, 2015) | `pure_momentum.csv`, `pure_momentum_equity.csv` |
| `analysis/04_survivorship.py` | LSEG panel only: key rules on all funds and on surviving funds only | `survivorship.csv` |
| `analysis/05_entry_signals.py` | The 63 combinations of the six entry conditions, each used as the only entry filter, plus a Reality Check against SPY | `entry_signals_63.csv`, `entry_signals_63_equity.csv`, `entry_signals_63_reality_check.csv` |
| `analysis/06_vix_regime.py` | Weekly returns regressed on four macro regimes (VIX > 25, NBER recession, term spread, credit spread), measured in the same week and in the previous week | `vix_regime.csv` |
| `analysis/07_trading_costs.py` | Half bid-ask spreads from LSEG quotes, for all ETF-weeks and for the trades of one rule (TSL07), compared with the 20 bp cost | `trading_costs.csv` |
| `analysis/08_summary.py` | A table of selected statistics from the files above, for both panels | `results/SUMMARY.md` |

Key rules are BASELINE (no exit rule), TSL07, TSL10, ATR4, MA_SMA200_OV_TP100, TP100 and the best of
the 144. Bootstrap tests use a stationary bootstrap with a mean block of 12 weeks and 2,000
replications (`src/config.py`).

The outputs of a full run on both panels are included in `results/`. They can be inspected without
the licensed price data.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pandas 2.2.x
# place the price panels in data/yahoo and data/lseg (see data/README.md)
PYTHON=python ./run_all.sh               # both panels, about 30 minutes with 8 GB of RAM
PYTHON=python ./run_all.sh yahoo         # one panel
```

Each script can also be run on its own, after `python -m src.precompute <panel>`, for example
`python analysis/03_pure_momentum.py lseg`.

## Using the engine directly

```python
from src import engine, metrics
from src.universe import Setup

s = Setup('yahoo')                                   # loads cache/yahoo/env.pkl and the filters
bt = engine.run(s.env, 'TSL10', 'TP50', s.cat_map, s.leveraged, **s.kwargs())
print(metrics.all_metrics(bt.equity, s.env['r_cash'], trades_per_year=bt.trades_per_year,
                          slots=bt.slots, n_years=s.env['n_years']))
bt.trades.head()                                     # every trade: date, ticker, side, weight
```

- **Change the parameters** (number of positions, rebalancing frequency, cost, volatility cap,
  bootstrap settings) in `src/config.py`, or pass `N`, `freq` and `cost` to `engine.run`.
- **Change the entry condition** with `entry_mask` (any boolean week × ETF table, for example one
  of the six conditions in `cache/<panel>/signals.pkl`).
- **Add a universe filter** with `mask_extra`.
- **Add a stop or take-profit** in `SL_CONFIGS` and `TP_LEVELS`.

## Repository structure

```
src/            backtest engine, signal construction, universe filters, metrics and tests
analysis/       one script per test
results/        outputs for both panels
data/public/    FRED series (T-bill, VIX, NBER, term and credit spreads)
data/yahoo/     ETF categories and ticker universe (prices not redistributed)
build_data/     scripts that rebuild the Yahoo and LSEG panels from the primary sources
```

## Data

The ETF price panels and the bid-ask quotes are not included: Yahoo Finance's terms do not allow
redistribution, and the LSEG data are licensed. The FRED series and the ETF classification are
included. How to rebuild each panel: [data/README.md](data/README.md).

## License

Code under the MIT license ([LICENSE](LICENSE)). The data keep the terms of their providers.
