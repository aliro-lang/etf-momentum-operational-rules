"""STEP 3e — Validation of the daily and weekly panels against the Yahoo panel of the original paper. Does NOT use LSEG.

For each common ticker: correlation of daily returns, median absolute difference (bp), % of days with a
difference > 1 pp; the same for weekly (Friday) returns. Yahoo is not the truth: it serves to detect gross
errors (crossed series, badly adjusted splits). Cases with low correlation are listed for review.

Output: build_data/lseg/data/panel/03e_validate_yahoo.csv and log
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eikon"))
from common import get_logger, ROOT, YAHOO

log = get_logger("03e_validate_yahoo")
P = ROOT / "data" / "panel"
F = pd.read_csv(P / "funds.csv", index_col=0)
Y = pd.read_parquet(YAHOO / "prices_daily.parquet"); Y.index = pd.to_datetime(Y.index)
tk2f = F.reset_index().drop_duplicates("TickerSymbol").set_index("TickerSymbol")["fund"]
in_common = [t for t in Y.columns if t in tk2f.index]
log.info(f"Yahoo tickers: {Y.shape[1]} | in common with the LSEG panel: {len(in_common)}")


def summary(a, b, level):
    """a, b: wide DataFrames (date × ticker) of returns; statistics by column over common dates."""
    m = a.notna() & b.notna()
    a, b = a.where(m), b.where(m)
    n = m.sum()
    am, bm = a.mean(), b.mean()
    cov = ((a - am) * (b - bm)).mean(); corr = cov / (a.std(ddof=0) * b.std(ddof=0))
    e = (a - b).abs()
    return pd.DataFrame({f"n_{level}": n, f"corr_{level}": corr, f"med_diff_bp_{level}": e.median() * 1e4,
                         f"pct_diff1pp_{level}": (e > 0.01).sum() / n})


d = pd.read_parquet(P / "daily.parquet", columns=["fund", "date", "ret", "adj_price"])
d = d[d["fund"].isin(set(tk2f[in_common]))]
d["fund"] = d["fund"].astype(str)
f2t = pd.Series(tk2f.index, index=tk2f.values)
L = d.pivot(index="date", columns="fund", values="ret").rename(columns=f2t)
Yr = Y[[t for t in in_common if t in L.columns]].pct_change(fill_method=None).reindex(L.index)
c1 = [t for t in in_common if t in L.columns]
rd = summary(L[c1], Yr[c1], "daily")

s = pd.read_parquet(P / "weekly.parquet", columns=["fund", "week", "ret_week"])
s = s[s["fund"].isin(set(tk2f[in_common]))]; s["fund"] = s["fund"].astype(str)
Ls = s.pivot(index="week", columns="fund", values="ret_week").rename(columns=f2t)
Yw = Y[[t for t in in_common if t in Ls.columns]].resample("W-FRI").last().pct_change(fill_method=None).reindex(Ls.index)
c2 = [t for t in in_common if t in Ls.columns]
rw = summary(Ls[c2], Yw[c2], "weekly")
r = rd.join(rw)
r.index.name = "ticker"
r.to_csv(P / "03e_validate_yahoo.csv")
pd.set_option("display.width", 220)
log.info("Distribution by fund:\n" + r.describe(percentiles=[.01, .05, .25, .5]).round(4).to_string())
log.info(f"Funds with daily corr < 0.95: {int((r.corr_daily < .95).sum())} | weekly corr < 0.95: {int((r.corr_weekly < .95).sum())}")
log.info("Worst 20 (weekly corr):\n" + r.sort_values("corr_weekly").head(20).round(4).to_string())
