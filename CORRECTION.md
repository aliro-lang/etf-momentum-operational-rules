# What was wrong in the original code, and what changed

This note describes the errors found after publication in the code of *Momentum Strategies
in ETFs under Simple Operational Rules: Economic Value and Conditional Predictability*, and how
this repository corrects them. All outputs in `results/` come from the corrected code.

## Errors that changed the results

| # | Error in the original code | Correction in this repository | Where |
|---|---|---|---|
| 1 | **Look-ahead in the backtest engine.** Each Friday the engine took exit and entry decisions with that week's Thursday close and only then credited the Friday-to-Friday return, so the decisions had already seen 4 of the 5 sessions of the return they earned. Example: on 10-Oct-2008 the trailing stop on SPY already saw a −17.8% fall and avoided the −19.8% week. | The return of the week ending on Friday *f* is credited to the portfolio held during that week; decisions taken on *f* earn returns from *f*+1 onward. | `src/engine.py` |
| 2 | Trading cost charged on the whole portfolio (20 bp × capital) instead of on the traded weight (20 bp × 1/8), i.e. 8 times the cost stated in the text. | Cost on the traded weight. | `src/engine.py` |
| 3 | Restoring a half-exited position to 100% was free. | It pays the cost. | `src/engine.py` |
| 4 | ETFs with full-sample volatility above 40% were dropped from the universe (uses future data). | No ETF is dropped; the 40% cap is applied at each entry with trailing 52-week volatility. | `src/precompute.py`, `src/universe.py` |
| 5 | The 63-rule entry-signal ablation and the replication of pure momentum (Tse, 2015) were both run inside the C2 ∧ C6 mask. This is why the ablation produced "only two Sharpe values". | No mask: each combination is the only entry filter, and pure momentum has no technical filter. | `analysis/05_entry_signals.py`, `src/engine.py::run_momentum` |
| 6 | Sharpe-ratio confidence intervals mixed the weekly sample size with the annual ratio, which made them √52 times too narrow. | Lo (2002) interval computed on the weekly ratio. | `src/metrics.py` |
| 7 | No correction for choosing the best of 144 rules. | White (2000) Reality Check with a stationary bootstrap. | `analysis/01_rules_144.py` |
| 8 | The price panel was described as Refinitiv DataStream; it was downloaded from Yahoo Finance and contains only surviving ETFs. The liquidity and coverage filters (ii)-(iii) described in the text were not implemented. | Provenance stated correctly; every test is repeated on an LSEG panel that includes 1,332 delisted ETFs and implements filters (ii)-(iii). | `data/README.md`, `build_data/lseg/` |
| 9 | The "conditional predictability" of the VIX regime was estimated with the regime measured in the same week as the return. | Same-week and previous-week regressions are reported separately; only the latter is a predictive test. | `analysis/06_vix_regime.py` |

## Effect on the headline numbers (Yahoo panel)

| | Original article | Corrected |
|---|---|---|
| TSL07 (trailing stop 7%) Sharpe, article's definition | 1.34 | 0.09 |
| TSL07 maximum drawdown | −7.0% | −26.9% |
| Rules with a higher Sharpe ratio than SPY | 144 of 144 | 4 of 144 (Reality Check p = 0.69) |
| Best rule | ATR2_TP60, Sharpe 1.42 | TP100, Sharpe 0.45 (SPY 0.39) |

