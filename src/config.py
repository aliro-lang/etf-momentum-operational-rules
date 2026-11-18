"""Paths and global settings.

Two price panels are supported (see data/README.md):
  yahoo : the panel used in the original article (Yahoo Finance, survivors only)
  lseg  : LSEG Workspace panel that includes delisted ETFs (licensed, not redistributed)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PUBLIC = DATA / 'public'          # FRED series (public, included)
CACHE = ROOT / 'cache'            # precomputed signals (generated, git-ignored)
RESULTS = ROOT / 'results'        # every number reported in README.md

BASES = ('yahoo', 'lseg')

# Strategy parameters (as in the article)
N_SLOTS = 8          # positions, each 1/8 of capital
FREQ = 2             # entries every 2 weeks
COST = 0.0020        # 20 bp per side, charged on the traded weight
VOL_MAX_EXANTE = 0.40  # trailing 52-week annualised volatility cap at entry
RF_PAPER = 0.03      # fixed rate in the article's Sharpe definition (CAGR - 3%) / vol
N_BOOT = 2000        # stationary-bootstrap replications
BLOCK = 12           # mean block length (weeks)
SEED = 2026


def data_dir(base: str) -> Path:
    if base not in BASES:
        raise SystemExit(f'base must be one of {BASES}, got {base!r}')
    return DATA / base


def out_dir(base: str) -> Path:
    p = RESULTS / base
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(base: str) -> Path:
    p = CACHE / base
    p.mkdir(parents=True, exist_ok=True)
    return p
