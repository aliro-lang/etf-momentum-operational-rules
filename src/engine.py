"""Backtest engine for the ETF momentum strategy with operational exit rules.

Each Friday f, in this order:
  1. VALUATION. The portfolio held during the week that ends on f earns that week's return
     (Friday f-1 → Friday f), plus the T-bill rate on the uninvested share.
  2. EXITS. Stop-loss, take-profit and technical exits are evaluated with the signal of f
     (Thursday close of that week). The technical half-exit (SP) cuts a position to 50%;
     when SP clears, the position is restored to 100%. Positions in funds that stop
     trading are closed.
  3. ENTRIES (every FREQ weeks). Free slots are filled with the highest momentum scores
     among eligible ETFs, subject to category caps.
Decisions taken on f earn returns from week f+1 onward. Trading costs are charged on the
traded weight (cost × s / N), including the restoration of a half-exited position.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COST, FREQ, N_SLOTS

SL_CONFIGS = {
    'SL05': ('pct', 0.05), 'SL07': ('pct', 0.07), 'SL10': ('pct', 0.10),
    'SL15': ('pct', 0.15), 'SL20': ('pct', 0.20),
    'TSL07': ('tpct', 0.07), 'TSL10': ('tpct', 0.10), 'TSL15': ('tpct', 0.15),
    'TSL20': ('tpct', 0.20),
    'LOW10': ('low', 10), 'LOW20': ('low', 20), 'LOW40': ('low', 40),
    'ATR2': ('atr', 2), 'ATR3': ('atr', 3), 'ATR4': ('atr', 4),
    'MA_EMA5W_OV': ('ma_overlay', 'ema5w'), 'MA_SMA50_OV': ('ma_overlay', 'sma50'),
    'MA_SMA100_OV': ('ma_overlay', 'sma100'), 'MA_SMA200_OV': ('ma_overlay', 'sma200'),
    'MA_EMA5W_RP': ('ma_replace', 'ema5w'), 'MA_SMA50_RP': ('ma_replace', 'sma50'),
    'MA_SMA100_RP': ('ma_replace', 'sma100'), 'MA_SMA200_RP': ('ma_replace', 'sma200'),
}
TP_LEVELS = {'TP30': 0.30, 'TP50': 0.50, 'TP60': 0.60, 'TP100': 1.00, 'TP200': 2.00}

# 23 stops × (no TP + 5 TP) + 5 TP alone + BASELINE (no stop, no TP) = 144 rules
ALL_RULES = []
for _sl in SL_CONFIGS:
    ALL_RULES.append((_sl, None))
    for _tp in TP_LEVELS:
        ALL_RULES.append((_sl, _tp))
for _tp in TP_LEVELS:
    ALL_RULES.append((None, _tp))
ALL_RULES.append((None, None))

CAPS = {'US_LARGE_CAP': 2, 'US_MID_SMALL': 1, 'US_SECTORAL': 2, 'INTL_DM_EQUITY': 1,
        'EM_EQUITY': 1, 'GOV_BONDS': 1, 'CORP_BONDS': 1, 'MONEY_MARKET': 1,
        'COMMODITIES': 1, 'REITS': 1, 'OTHER': 1}


def code(sl, tp):
    if sl and tp:
        return f'{sl}_{tp}'
    return sl or tp or 'BASELINE'


RULE_BY_CODE = {code(a, b): (a, b) for a, b in ALL_RULES}


@dataclass
class Backtest:
    equity: pd.Series        # portfolio value, starts at 100
    trades_per_year: dict    # year → number of trades
    slots: pd.Series         # occupied slots each week
    exits: pd.DataFrame      # one row per full exit (date, ticker, reason, weeks held)
    cash: pd.Series          # uninvested share each week
    trades: pd.DataFrame     # every trade (date, ticker, side, traded weight)


def run(env, sl_name, tp_name, cat_map, leveraged, *, mask_extra=None, entry_mask=None,
        delisted_after=None, N=N_SLOTS, freq=FREQ, cost=COST, diversify=True, caps=CAPS):
    """Run one exit rule.

    env            : dict from src.precompute (cache/<panel>/env.pkl)
    mask_extra     : boolean DataFrame (week × ETF) intersected with eligibility at entry
                     (ex-ante volatility cap, LSEG liquidity/coverage filters, ...)
    entry_mask     : entry condition; defaults to env['MASK'] = C2 ∧ C6
    delisted_after : dict ticker → last trading day; open positions are closed after it
    """
    returns = env['returns']; eligible = env['eligible']
    score = env['score_mom']; MASK = env['MASK'] if entry_mask is None else entry_mask
    SP = env['SP']; ST = env['ST']; sig_w = env['sig_w']
    MA_W = env.get('MA_W'); ATR_W = env.get('ATR_W'); LOW_N = env.get('LOW_N')
    dates = env['dates']; r_cash = env['r_cash']

    sl_type, sl_param = (None, None); use_tec = True
    ma_series = low_series = None
    if sl_name is not None:
        sl_type, sl_param = SL_CONFIGS[sl_name]
        if sl_type == 'low':
            low_series = LOW_N[sl_param]
        elif sl_type in ('ma_overlay', 'ma_replace'):
            ma_series = MA_W[sl_param]
            use_tec = sl_type == 'ma_overlay'
    tp_pct = TP_LEVELS[tp_name] if tp_name else None

    def fee(s):  # cost on the traded weight s/N
        return cost * s / N

    capital = 100.0
    holdings = {}; entry_price = {}; entry_date = {}; peak_price = {}
    values = []; trades_yr = defaultdict(int); slots_h = []; cash_h = []
    exits = []; trades = []

    def trade(f, t, side, s):
        nonlocal capital
        capital *= (1 - fee(s)); trades_yr[f.year] += 1
        trades.append((f, t, side, s / N))

    def close(t):
        holdings[t] = 0.0
        entry_date.pop(t, None); entry_price.pop(t, None); peak_price.pop(t, None)

    for i, f in enumerate(dates):
        # 1) valuation of the portfolio held during the week ending on f
        active = {t: s for t, s in holdings.items() if s > 0}
        invested = sum(s / N for s in active.values())
        port_ret = sum((s / N) * returns.loc[f, t] for t, s in active.items()
                       if t in returns.columns)
        cash_ret = max(0.0, 1.0 - invested) * float(r_cash.loc[f]) if f in r_cash.index else 0
        capital *= (1 + port_ret + cash_ret)

        # 2) exits
        for t in list(holdings.keys()):
            s = holdings[t]
            if s <= 0:
                continue
            if delisted_after is not None and t in delisted_after and f > delisted_after[t]:
                trade(f, t, 'sell', s)
                exits.append({'date': f, 'ticker': t, 'reason': 'DELISTED', 'weeks': 0})
                close(t)
                continue
            price = sig_w.loc[f, t] if t in sig_w.columns else np.nan
            if pd.isna(price):
                continue
            peak_price[t] = max(peak_price.get(t, entry_price[t]), price)
            sl_hit = False
            if sl_type == 'pct':
                sl_hit = price < entry_price[t] * (1 - sl_param)
            elif sl_type == 'tpct':
                sl_hit = price < peak_price[t] * (1 - sl_param)
            elif sl_type == 'low':
                lv = low_series.loc[f, t] if t in low_series.columns else np.nan
                if not pd.isna(lv):
                    sl_hit = price < lv
            elif sl_type == 'atr':
                atr = ATR_W.loc[f, t] if t in ATR_W.columns else np.nan
                if not pd.isna(atr):
                    sl_hit = price < peak_price[t] - sl_param * atr
            elif sl_type in ('ma_overlay', 'ma_replace'):
                ma_v = ma_series.loc[f, t] if t in ma_series.columns else np.nan
                if not pd.isna(ma_v):
                    sl_hit = price < ma_v
            tp_hit = tp_pct is not None and price >= entry_price[t] * (1 + tp_pct)
            st = bool(ST.loc[f, t]) if (use_tec and t in ST.columns) else False
            sp = bool(SP.loc[f, t]) if (use_tec and t in SP.columns) else False
            if sl_hit or tp_hit or st:
                trade(f, t, 'sell', s)
                exits.append({'date': f, 'ticker': t,
                              'reason': 'SL' if sl_hit else ('TP' if tp_hit else 'ST'),
                              'weeks': round((f - entry_date.get(t, f)).days / 7, 1)})
                close(t)
            elif sp and s == 1.0:
                trade(f, t, 'sell', 0.5); holdings[t] = 0.5
            elif not sp and s == 0.5:
                trade(f, t, 'buy', 0.5); holdings[t] = 1.0

        # 3) entries
        if i % freq == 0:
            holdings = {t: s for t, s in holdings.items() if s > 0}
            free = N - len(holdings)
            if free > 0:
                ok = eligible.loc[f]
                if mask_extra is not None and f in mask_extra.index:
                    ok = ok & mask_extra.loc[f].reindex(ok.index, fill_value=False)
                candidates = list(ok[ok].index)
                sc = score.loc[f, candidates]; passes = MASK.loc[f, candidates]
                ranked = sc[passes].sort_values(ascending=False)
                held = set(holdings.keys()); cat_count = defaultdict(int)
                for t in holdings:
                    cat_count[cat_map.get(t, 'OTHER')] += 1
                added = 0
                for t in ranked.index:
                    if t in held or added >= free or t in leveraged:
                        continue
                    cat = cat_map.get(t, 'OTHER')
                    if diversify and cat_count[cat] >= caps.get(cat, 1):
                        continue
                    trade(f, t, 'buy', 1.0); holdings[t] = 1.0
                    entry_date[t] = f
                    entry_price[t] = sig_w.loc[f, t] if t in sig_w.columns else np.nan
                    peak_price[t] = entry_price[t]
                    cat_count[cat] += 1; added += 1

        active = {t: s for t, s in holdings.items() if s > 0}
        slots_h.append(len(active))
        cash_h.append(max(0.0, 1.0 - sum(s / N for s in active.values())))
        values.append(capital)

    return Backtest(pd.Series(values, index=dates), dict(trades_yr),
                    pd.Series(slots_h, index=dates), pd.DataFrame(exits),
                    pd.Series(cash_h, index=dates),
                    pd.DataFrame(trades, columns=['date', 'ticker', 'side', 'weight']))


def run_momentum(env, lookback_weeks, leveraged, *, strategy='cross', mask_extra=None,
                 N=N_SLOTS, freq=FREQ, cost=COST):
    """Pure momentum, no technical filters and no exit rules (as in Tse, 2015).

    Every FREQ weeks hold the N ETFs with the highest past return over `lookback_weeks`
    ('cross'), or only those among them with a positive past return ('ts').
    Same timing and cost convention as `run`.
    """
    dates, returns, r_cash = env['dates'], env['returns'], env['r_cash']
    mom = env['weekly'].pct_change(lookback_weeks)
    mask = env['eligible']
    if mask_extra is not None:
        mask = mask & mask_extra.reindex_like(mask).fillna(False).astype(bool)
    capital = 100.0; holdings = {}; values = []; slots_h = []; n_trades = 0
    for i, f in enumerate(dates):
        active = {t: s for t, s in holdings.items() if s > 0}
        invested = sum(s / N for s in active.values())
        port_ret = sum((s / N) * returns.loc[f, t] for t, s in active.items()
                       if pd.notna(returns.loc[f, t]))
        capital *= (1 + port_ret + max(0.0, 1 - invested) * float(r_cash.loc[f]))
        if i % freq == 0:
            cand = mom.loc[f][mask.loc[f]].dropna()
            cand = cand[~cand.index.isin(leveraged)]
            if strategy == 'ts':
                cand = cand[cand > 0]
            keep = list(cand.sort_values(ascending=False).head(N).index)
            for t in list(holdings):
                if t not in keep:
                    capital *= (1 - cost * holdings[t] / N); n_trades += 1; del holdings[t]
            for t in keep:
                if t not in holdings:
                    capital *= (1 - cost / N); holdings[t] = 1.0; n_trades += 1
        slots_h.append(len(holdings)); values.append(capital)
    return pd.Series(values, index=dates), pd.Series(slots_h, index=dates), n_trades
