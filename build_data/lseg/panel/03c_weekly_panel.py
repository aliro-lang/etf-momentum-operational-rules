"""STEP 3c — Weekly panel (Friday closes) and investable-universe filters. Does NOT use LSEG.

Everything reported for week t uses only data up to the close of the last trading day of that week
(Friday, or the previous trading day if Friday is a holiday). Nothing looks ahead.

Filters of the paper (footnote in section 3, version R1), evaluated every week t:
  (i)   age: at least 52 weeks since the fund's first price (age_weeks >= 52)
  (ii)  liquidity: 13-week (63 trading days) rolling median of daily dollar volume > USD 100,000.
        Trading days without trades within the fund's life count as volume 0.
  (iii) coverage: at least 90% non-missing closes over the last 252 trading days
  (iv)  leveraged and inverse funds: flagged in a separate step (03d), not here.
Calendar: trading days = days with an SPY price.

Input:  build_data/lseg/data/panel/daily.parquet
Output: build_data/lseg/data/panel/weekly.parquet (fund, week, close_date, price, adj_price, ret_week, trading_days,
                                       dollar_vol_med63, coverage252, age_weeks, ok_age, ok_liq, ok_cov)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eikon"))
from common import get_logger, ROOT

PANEL = ROOT / "data" / "panel"
log = get_logger("03c_weekly_panel")
d = pd.read_parquet(PANEL / "daily.parquet")
d["fund"] = d["fund"].astype(str)
F = pd.read_csv(PANEL / "funds.csv", index_col=0)
spy = F.index[F["TickerSymbol"] == "SPY"][0]
cal = pd.DatetimeIndex(sorted(d.loc[d["fund"] == spy, "date"].unique()))
outside = ~d["date"].isin(cal)
log.info(f"Rows outside the SPY calendar (dropped): {int(outside.sum())}")
d = d[~outside]

W = lambda c: d.pivot(index="date", columns="fund", values=c).reindex(cal).astype("float32")
ret, pa, pr, dv = W("ret"), W("adj_price"), W("price"), W("dollar_vol")
funds = ret.columns
has = pr.notna()
first = has.idxmax()                                   # first day with a price for each fund
last = has[::-1].idxmax()
alive = (cal.values[:, None] >= first.values[None, :]) & (cal.values[:, None] <= last.values[None, :])
alive = pd.DataFrame(alive, index=cal, columns=funds)
log.info(f"Daily matrix: {len(cal)} days × {len(funds)} funds")

# (ii) liquidity: days without trades within the fund's life count as 0
dv0 = dv.where(~(alive & dv.isna()), 0.0).where(alive)
med63 = dv0.rolling(63, min_periods=63).median()
# (iii) coverage of closes over the last 252 trading days (before the first price counts as missing)
cov = has.astype(float).rolling(252, min_periods=252).mean()

# Weeks: label = Friday of the week; close = last trading day of the week
wk = pd.Series(cal.to_period("W-FRI").end_time.normalize(), index=cal)
last_day = wk.groupby(wk).apply(lambda s: s.index.max())
close_idx = pd.DatetimeIndex(last_day.values)
log.info(f"Weeks: {len(last_day)} ({last_day.index.min().date()} → {last_day.index.max().date()}); "
         f"weeks whose close is not a Friday (holiday): {int((close_idx.dayofweek != 4).sum())}")

# Weekly return = product of the daily returns of the week (days without trades have no row;
# the TR of the next day with trades covers from the last price, so compounding what is available is exact).
ret_week = (1 + ret.fillna(0)).groupby(wk.values).prod() - 1
trading_days = has.groupby(wk.values).sum()
# Prices at the weekly close: last value available up to that day (within the fund's life)
pa_c = pa.ffill().where(alive).loc[close_idx]; pa_c.index = last_day.index
pr_c = pr.ffill().where(alive).loc[close_idx]; pr_c.index = last_day.index
med_c = med63.loc[close_idx]; med_c.index = last_day.index
cov_c = cov.loc[close_idx]; cov_c.index = last_day.index
alive_c = alive.loc[close_idx]; alive_c.index = last_day.index
ret_week = ret_week.where(alive_c)
# the week in which the fund appears has no complete weekly return
first_wk = wk.reindex(first.values).values
for j, f in enumerate(funds):
    ret_week.loc[first_wk[j], f] = np.nan
age = pd.DataFrame((last_day.index.values[:, None] - first.values[None, :]) / np.timedelta64(7, "D"),
                   index=last_day.index, columns=funds)

L = lambda m, n: m.stack(future_stack=True).rename(n)
s = pd.concat([L(pr_c, "price"), L(pa_c, "adj_price"), L(ret_week, "ret_week"), L(trading_days.where(alive_c), "trading_days"),
               L(med_c, "dollar_vol_med63"), L(cov_c, "coverage252"), L(age.where(alive_c), "age_weeks")], axis=1)
s.index.names = ["week", "fund"]
s = s[s["price"].notna()].reset_index()
s["close_date"] = s["week"].map(last_day)
s["ok_age"] = s["age_weeks"] >= 52
s["ok_liq"] = s["dollar_vol_med63"] > 100_000
s["ok_cov"] = s["coverage252"] >= 0.90
s["fund"] = s["fund"].astype("category")
s.to_parquet(PANEL / "weekly.parquet", index=False)

allok = s["ok_age"] & s["ok_liq"] & s["ok_cov"]
n = s[allok].groupby("week")["fund"].nunique()
log.info(f"weekly.parquet: {len(s):,} fund-week rows, {s['fund'].nunique()} funds")
log.info(f"Funds that pass (i)-(iii) in some week: {s.loc[allok, 'fund'].nunique()}")
for dte in ["1995-12-29", "2001-08-31", "2005-12-30", "2010-12-31", "2015-12-31", "2020-12-31", "2026-05-15"]:
    t = pd.Timestamp(dte)
    log.info(f"  {dte}: investable (i)-(iii) = {int(n.get(t, 0))}")
