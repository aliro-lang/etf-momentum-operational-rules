"""
30_etf_categories.py
====================
Downloads the Morningstar/yfinance category of the ETFs in the eligible universe
(every ETF that passes the eligibility mask in some week).

Output: data/yahoo/etf_categories_raw.csv (block 0) and data/yahoo/etf_categories.parquet (block 1) with columns:
    ticker, yf_category, super_category, is_leveraged_or_inverse

Super-categories (12) used for the diversification caps:
    US_LARGE_CAP     — Large Blend/Growth/Value
    US_MID_SMALL     — Mid/Small Cap Blend/Growth/Value
    US_SECTORAL      — Technology, Financial, Health, Energy, etc.
    INTL_DM_EQUITY   — Foreign Large, Europe, Japan, Pacific, Asia DM
    EM_EQUITY        — Diversified EM, China, India, LatAm, regional EM
    GOV_BONDS        — Long/Short/Intermediate Government, TIPS
    CORP_BONDS       — Corporate Bond, High Yield, EM Bond, Multisector
    MONEY_MARKET     — Ultrashort, Cash, BIL
    COMMODITIES      — Commodities Focused/Broad Basket
    REITS            — Real Estate, Global Real Estate
    DIVIDEND         — Dividend (sometimes categorised separately)
    OTHER            — catch-all

BLOCKS:
    0  Download the yfinance categories (with caching)
    1  Build the super_category mapping + report the distribution

Usage:
    python build_data/yahoo/30_etf_categories.py 0
    python build_data/yahoo/30_etf_categories.py 1 [output.parquet]   (default: data/yahoo/etf_categories.parquet)
"""

import sys, time, json
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else -1
BASE = Path(__file__).resolve().parents[2]   # repository root
OUT_RAW = BASE / 'data' / 'yahoo' / 'etf_categories_raw.csv'
OUT     = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / 'data' / 'yahoo' / 'etf_categories.parquet'


def get_universe_tickers():
    """Full historical eligible universe (MASK=True in some week).
    Uses the precompute pickle as the single source."""
    import pickle as pkl
    PICKLE = BASE / 'cache' / 'yahoo' / 'env.pkl'   # python -m src.precompute yahoo env
    with open(PICKLE, 'rb') as fh:
        env = pkl.load(fh)
    mask_total = env['MASK'].any(axis=0)
    return sorted(mask_total[mask_total].index.tolist())


# ════════════════════════════════════════════════════════════════
# BLOCK 0 — Download yfinance categories with caching
# ════════════════════════════════════════════════════════════════
if BLOCK == 0:
    import yfinance as yf
    print('═'*60); print(' BLOCK 0 — download yfinance categories'); print('═'*60)
    tickers = get_universe_tickers()
    print(f'  Universe: {len(tickers)} unique ETFs')

    # Load the cache if it exists
    if OUT_RAW.exists():
        cache = pd.read_csv(OUT_RAW).set_index('ticker')['yf_category'].to_dict()
        print(f'  Previous cache: {len(cache)} ETFs already mapped')
    else:
        cache = {}

    pending = [t for t in tickers if t not in cache]
    print(f'  Pending: {len(pending)} ETFs to download')
    if not pending:
        print('  All already mapped. Skip.')
        sys.exit(0)

    t0 = time.time()
    for i, t in enumerate(pending):
        try:
            info = yf.Ticker(t).info
            cat = info.get('category')
            if cat is None:
                cat = info.get('quoteType', '<NONE>')
            cache[t] = cat
        except Exception as e:
            cache[t] = f'ERROR:{str(e)[:30]}'
        if (i+1) % 25 == 0 or i == len(pending)-1:
            print(f'  [{i+1:>4}/{len(pending)}] {t:<6} → {cache[t][:40]:<40}  '
                  f'({time.time()-t0:.0f}s)')
            # Save the cache every 25 ETFs in case of interruption
            pd.DataFrame([{'ticker':k, 'yf_category':v} for k,v in cache.items()]
                         ).to_csv(OUT_RAW, index=False)

    pd.DataFrame([{'ticker':k, 'yf_category':v} for k,v in cache.items()]
                 ).to_csv(OUT_RAW, index=False)
    print(f'\n  Saved: {OUT_RAW}')
    print(f'  Total time: {time.time()-t0:.0f}s')
    sys.exit(0)


# ════════════════════════════════════════════════════════════════
# yfinance categories of LEVERAGED / INVERSE products
# (excluded from the investable universe; they are trading products,
# not rebalanced-portfolio holdings)
# ════════════════════════════════════════════════════════════════
LEVERAGED_INVERSE_CATEGORIES = {
    'Trading--Leveraged Equity',
    'Trading--Leveraged Debt',
    'Trading--Leveraged Commodities',
    'Trading--Inverse Equity',
    'Trading--Inverse Debt',
    'Trading--Inverse Commodities',
    'Trading--Miscellaneous',
}


# ════════════════════════════════════════════════════════════════
# Mapping yfinance category → super_category
# ════════════════════════════════════════════════════════════════
SUPER_CATEGORY_MAP = {
    # US Large-Cap Equity
    'Large Blend':                     'US_LARGE_CAP',
    'Large Growth':                    'US_LARGE_CAP',
    'Large Value':                     'US_LARGE_CAP',
    'US Fund Large Blend':             'US_LARGE_CAP',
    'US Fund Large Growth':            'US_LARGE_CAP',
    'US Fund Large Value':             'US_LARGE_CAP',

    # US Mid/Small Cap
    'Mid-Cap Blend':                   'US_MID_SMALL',
    'Mid-Cap Growth':                  'US_MID_SMALL',
    'Mid-Cap Value':                   'US_MID_SMALL',
    'Small Blend':                     'US_MID_SMALL',
    'Small Growth':                    'US_MID_SMALL',
    'Small Value':                     'US_MID_SMALL',

    # US Sectoral Equity
    'Technology':                      'US_SECTORAL',
    'Financial':                       'US_SECTORAL',
    'Health':                          'US_SECTORAL',
    'Equity Energy':                   'US_SECTORAL',
    'Industrials':                     'US_SECTORAL',
    'Consumer Defensive':              'US_SECTORAL',
    'Consumer Cyclical':               'US_SECTORAL',
    'Utilities':                       'US_SECTORAL',
    'Communications':                  'US_SECTORAL',
    'Natural Resources':               'US_SECTORAL',
    'Equity Precious Metals':          'US_SECTORAL',
    'Miscellaneous Sector':            'US_SECTORAL',
    'Energy Limited Partnership':      'US_SECTORAL',
    'Infrastructure':                  'US_SECTORAL',
    'Derivative Income':               'US_SECTORAL',

    # International DM Equity
    'Foreign Large Blend':             'INTL_DM_EQUITY',
    'Foreign Large Growth':            'INTL_DM_EQUITY',
    'Foreign Large Value':             'INTL_DM_EQUITY',
    'Foreign Small/Mid Blend':         'INTL_DM_EQUITY',
    'Foreign Small/Mid Growth':        'INTL_DM_EQUITY',
    'Foreign Small/Mid Value':         'INTL_DM_EQUITY',
    'Europe Stock':                    'INTL_DM_EQUITY',
    'Japan Stock':                     'INTL_DM_EQUITY',
    'Pacific/Asia ex-Japan Stk':       'INTL_DM_EQUITY',
    'Focused Region':                  'INTL_DM_EQUITY',
    'Global Large-Stock Blend':        'INTL_DM_EQUITY',
    'World Stock':                     'INTL_DM_EQUITY',
    'World Large Stock':               'INTL_DM_EQUITY',
    'World Small/Mid Stock':           'INTL_DM_EQUITY',
    'Global Small/Mid Stock':          'INTL_DM_EQUITY',
    'Global Large-Stock Value':        'INTL_DM_EQUITY',

    # EM Equity
    'Diversified Emerging Mkts':       'EM_EQUITY',
    'Greater China Region':            'EM_EQUITY',
    'China Region':                    'EM_EQUITY',
    'India Equity':                    'EM_EQUITY',
    'Latin America Stock':             'EM_EQUITY',
    'Emerging Markets':                'EM_EQUITY',

    # Government Bonds
    'Long Government':                 'GOV_BONDS',
    'Short Government':                'GOV_BONDS',
    'Intermediate Government':         'GOV_BONDS',
    'Inflation-Protected Bond':        'GOV_BONDS',
    'Short-Term Inflation-Protected Bond': 'GOV_BONDS',
    'Government Mortgage-Backed Bond': 'GOV_BONDS',
    'Target Maturity':                 'GOV_BONDS',

    # Corporate / Credit Bonds
    'Corporate Bond':                  'CORP_BONDS',
    'High Yield Bond':                 'CORP_BONDS',
    'Emerging Markets Bond':           'CORP_BONDS',
    'World Bond':                      'CORP_BONDS',
    'Multisector Bond':                'CORP_BONDS',
    'Bank Loan':                       'CORP_BONDS',
    'Preferred Stock':                 'CORP_BONDS',
    'Convertibles':                    'CORP_BONDS',
    'Long-Term Bond':                  'CORP_BONDS',
    'Intermediate Core Bond':          'CORP_BONDS',
    'Intermediate Core-Plus Bond':     'CORP_BONDS',
    'Short-Term Bond':                 'CORP_BONDS',
    'Muni National Long':              'CORP_BONDS',
    'Muni National Short':             'CORP_BONDS',
    'Muni National Intermediate':      'CORP_BONDS',
    'Muni National Interm':            'CORP_BONDS',
    'High Yield Muni':                 'CORP_BONDS',
    'Muni California Long':            'CORP_BONDS',
    'Muni New York Long':              'CORP_BONDS',
    'Nontraditional Bond':             'CORP_BONDS',
    'Tactical Allocation':             'CORP_BONDS',
    'Securitized Bond - Focused':      'CORP_BONDS',
    'Global Bond-USD Hedged':          'CORP_BONDS',

    # Money Market / Ultrashort
    'Ultrashort Bond':                 'MONEY_MARKET',
    'Money Market':                    'MONEY_MARKET',
    'Stable Value':                    'MONEY_MARKET',

    # Commodities
    'Commodities Focused':             'COMMODITIES',
    'Commodities Broad Basket':        'COMMODITIES',
    'Trading--Inverse Commodities':    'COMMODITIES',
    'Trading--Leveraged Commodities':  'COMMODITIES',
    'Single Currency':                 'COMMODITIES',

    # REITs
    'Real Estate':                     'REITS',
    'Global Real Estate':              'REITS',

    # Dividend / Alternative
    'Equity Hedged':                   'OTHER',
    'Long-Short Equity':               'OTHER',
    'Market Neutral':                  'OTHER',
    'Trading--Inverse Equity':         'OTHER',
    'Trading--Inverse Debt':           'OTHER',
    'Multistrategy':                   'OTHER',
    'Allocation--50% to 70% Equity':   'OTHER',
    'Allocation--70% to 85% Equity':   'OTHER',
    'Allocation--85%+ Equity':         'OTHER',
    'Allocation--15% to 30% Equity':   'OTHER',
    'Allocation--30% to 50% Equity':   'OTHER',
    'Target-Date 2020':                'OTHER',
    'Target-Date 2025':                'OTHER',
    'Target-Date 2030':                'OTHER',
}


# ════════════════════════════════════════════════════════════════
# BLOCK 1 — Build the super_category mapping + report
# ════════════════════════════════════════════════════════════════
if BLOCK == 1:
    print('═'*60); print(' BLOCK 1 — build super_category + report'); print('═'*60)
    raw = pd.read_csv(OUT_RAW)
    raw['super_category'] = raw['yf_category'].map(SUPER_CATEGORY_MAP).fillna('OTHER')
    # Flag leveraged / inverse
    raw['is_leveraged_or_inverse'] = raw['yf_category'].isin(LEVERAGED_INVERSE_CATEGORIES)

    print(f'\n  ETFs flagged LEVERAGED/INVERSE (to exclude): {raw["is_leveraged_or_inverse"].sum()}')
    if raw['is_leveraged_or_inverse'].sum() > 0:
        print('  Sample:')
        print(raw[raw['is_leveraged_or_inverse']][['ticker','yf_category']].head(20).to_string(index=False))

    print('\n  Distribution of yfinance category vs super_category:\n')
    print('  yfinance categories without a mapping (go to OTHER):')
    unmapped = raw[raw['yf_category'].apply(
        lambda c: SUPER_CATEGORY_MAP.get(c) is None)]['yf_category'].value_counts()
    print(unmapped.head(30).to_string())

    print('\n  super_category distribution (ETFs per category):')
    print(raw['super_category'].value_counts().to_string())

    raw[['ticker', 'yf_category', 'super_category', 'is_leveraged_or_inverse']
        ].to_parquet(OUT, compression='snappy')
    print(f'\n  Saved: {OUT}')
    sys.exit(0)


print('Usage: python build_data/yahoo/30_etf_categories.py {0|1} [output.parquet]')
print('  0  Download yfinance categories (~40 min)')
print('  1  Build super_category + report the distribution')
