#!/usr/bin/env bash
# Reproduce every result in results/. Needs the price panels in data/yahoo and data/lseg
# (see data/README.md). Usage: ./run_all.sh [yahoo] [lseg]   (default: both)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
BASES=${*:-yahoo lseg}
for b in $BASES; do
  echo "=== $b ==="
  $PY -m src.precompute "$b"
  $PY analysis/01_rules_144.py "$b"
  $PY analysis/02_benchmarks_drawdown.py "$b"
  $PY analysis/03_pure_momentum.py "$b"
  [ "$b" = lseg ] && $PY analysis/04_survivorship.py
  $PY analysis/05_entry_signals.py "$b"
  $PY analysis/06_vix_regime.py "$b"
  if [ -f data/eikon/portfolio_ask_weekly.csv ]; then $PY analysis/07_trading_costs.py "$b"; fi
done
$PY analysis/08_summary.py
