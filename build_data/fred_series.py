"""Rebuild the public FRED inputs in data/public/ (already included in the repository).

    python build_data/fred_series.py

data/public/tb3m_weekly.parquet
    dtb3_pct       3-month T-bill, secondary market (DTB3), last value of each week (Friday)
    r_cash_weekly  (1 + dtb3_pct/100)^(1/52) - 1, the weekly return on uninvested cash
data/public/macro_regimes_weekly.parquet
    VIX (VIXCLS), TERM (T10Y3M), CREDIT (BAA10Y), NBER (USREC), weekly (Friday), and
    VIX_HIGH = VIX > 25, TERM_LOW = TERM < 1 pp, CREDIT_HIGH = CREDIT above its sample 75th
    percentile, NBER = recession month. CREDIT_HIGH and NBER use information that is not
    available in real time; only VIX_HIGH and TERM_LOW are real-time regimes.
"""
import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / 'data' / 'public'


def fred(code):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}'
    s = pd.read_csv(io.StringIO(urlopen(url, timeout=60).read().decode()), index_col=0,
                    parse_dates=True, na_values='.').iloc[:, 0].astype(float)
    return s.dropna()


if __name__ == '__main__':
    tb = fred('DTB3').resample('W-FRI').last().ffill()
    t = pd.DataFrame({'dtb3_pct': tb, 'r_cash_weekly': (1 + tb / 100) ** (1 / 52) - 1})
    t.index.name = 'date'
    t.to_parquet(OUT / 'tb3m_weekly.parquet')

    w = pd.date_range('1995-01-06', '2026-05-31', freq='W-FRI')
    df = pd.DataFrame(index=w)
    for k, c in [('VIX', 'VIXCLS'), ('TERM', 'T10Y3M'), ('CREDIT', 'BAA10Y')]:
        df[k] = fred(c).resample('W-FRI').last().reindex(w)
    df['NBER'] = fred('USREC').resample('W-FRI').last().ffill().reindex(w)
    df = df.ffill().dropna()
    df['VIX_HIGH'] = (df['VIX'] > 25).astype(int)
    df['TERM_LOW'] = (df['TERM'] < 1.0).astype(int)
    df['CREDIT_HIGH'] = (df['CREDIT'] > df['CREDIT'].quantile(0.75)).astype(int)
    df['NBER'] = df['NBER'].astype(int)
    df.to_parquet(OUT / 'macro_regimes_weekly.parquet')
    print(t.tail(2), df.tail(2), sep='\n')
