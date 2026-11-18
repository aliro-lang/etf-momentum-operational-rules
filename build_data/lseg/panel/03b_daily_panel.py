"""STEP 3b — Clean daily panel, one fund (ISIN) per series. Does NOT use LSEG.

Rules (each backed by a diagnostic; see 03b_daily_panel.log):
  R1 Source. The consolidated (primary) RIC is used. The request by ISIN ONLY fills dates outside the span
     in which the primary trades (before its first observation or after its last: delisted funds or funds with
     an older history on another exchange). Inside the span, a day without a primary observation is a day without
     trades: the ISIN observation of that day comes from a secondary venue (when it differs from the primary,
     Yahoo agrees with the primary in 99.4% of cases).
  R2 Daily return = LSEG TR.TotalReturn (includes dividends). If missing, the price return.
     (When TR and the price return differ and it is not a dividend, TR agrees with Yahoo in 95% of cases.)
  R3 Gaps > 183 calendar days: everything before the last gap is dropped (usually another
     instrument with the same code, e.g. DIG before 2007, MLN 1998-2004).
  R4 Reverting spikes: |r_t| > 20% and (1+r_t)(1+r_t+1)-1 within ±5% → wrong price on day t:
     r_t = compounded return of the two days and r_t+1 = 0.
  R5 Dollar volume = price × volume (consolidated volume if the source is the primary).
     LSEG fills Volume=0 before launch; those rows were already removed in 03a (they have no price).
  Calendar: trading days = days with an SPY (primary) price.

Input:  build_data/lseg/data/panel/daily_raw.parquet, data/eikon/universe_rics_download.csv, universe_etf_us.csv
Output: build_data/lseg/data/panel/daily.parquet  (fund, date, price, adj_price, volume, dollar_vol, ret, source, adjustment)
        build_data/lseg/data/panel/funds.csv      (one record per fund: ticker, status, dates, number of days, source)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eikon"))
from common import get_logger, ROOT

PANEL = ROOT / "data" / "panel"
E = ROOT / "data" / "eikon"
log = get_logger("03b_daily_panel")
GAP, SPIKE, REVERTS = 183, 0.20, 0.05

x = pd.read_parquet(PANEL / "daily_raw.parquet")
x["ric"] = x["ric"].astype(str)
m = pd.read_csv(E / "universe_rics_download.csv").drop_duplicates("RIC").set_index("RIC")[["fund", "role"]]
x = x.join(m, on="ric")
if x["fund"].isna().any():
    raise SystemExit(f"RIC without a fund in the universe: {x.loc[x.fund.isna(), 'ric'].unique()[:10]}")
x = x[x["price"].notna()]

# ── R1: primary + ISIN only outside the primary's span ──
p = x[x["role"] == "primary"]
i = x[x["role"] == "isin"]
span = p.groupby("fund")["date"].agg(["min", "max"])
i = i.join(span, on="fund")
outside = i["min"].isna() | (i["date"] < i["min"]) | (i["date"] > i["max"])
log.info(f"R1 primary rows: {len(p):,} | ISIN rows used (outside the primary's span): {int(outside.sum()):,} "
         f"| ISIN rows dropped (inside the span, secondary venue): {int((~outside).sum()):,}")
d = pd.concat([p.assign(source="primary"), i[outside].drop(columns=["min", "max"]).assign(source="isin")])
d = d.sort_values(["fund", "date"]).drop_duplicates(["fund", "date"]).reset_index(drop=True)

# ── R3: cut before the last long gap ──
gap = d.groupby("fund")["date"].diff().dt.days
last_gap = d.loc[gap > GAP].groupby("fund")["date"].max()
cut = d["fund"].map(last_gap)
drop = cut.notna() & (d["date"] < cut)
log.info(f"R3 gaps > {GAP} days: {int((gap > GAP).sum())} in {last_gap.size} funds; rows dropped: {int(drop.sum()):,}")
d = d[~drop].reset_index(drop=True)

# ── R2: return ──
prev_p = d.groupby("fund")["price"].shift(1)
d["ret"] = d["tr_pct"] / 100
no_tr = d["ret"].isna() & prev_p.notna()
d.loc[no_tr, "ret"] = d.loc[no_tr, "price"] / prev_p[no_tr] - 1
first = ~d["fund"].duplicated()
d.loc[first, "ret"] = np.nan                  # the first day of each fund has no return
d["adjustment"] = ""
d.loc[no_tr, "adjustment"] = "price return (no TR)"
log.info(f"R2 days without TR using the price return: {int(no_tr.sum()):,}")
# When the source changes (primary ↔ ISIN), LSEG's TR was computed against the previous price of its own
# series, which may be from another date: it is recomputed with the price of the previous row of the panel.
switch = (d["source"] != d.groupby("fund")["source"].shift(1)) & ~first
d.loc[switch, "ret"] = d.loc[switch, "price"] / prev_p[switch] - 1
d.loc[switch, "adjustment"] = "source switch: price return"
log.info(f"   source-switch days (price return): {int(switch.sum()):,}")

# ── R4: reverting spikes ──
r1 = d.groupby("fund")["ret"].shift(-1)
same = d["fund"].eq(d["fund"].shift(-1))
spike = same & (d["ret"].abs() > SPIKE) & (((1 + d["ret"]) * (1 + r1) - 1).abs() < REVERTS)
idx = np.flatnonzero(spike.values)
d.loc[idx, "ret"] = ((1 + d["ret"].values[idx]) * (1 + r1.values[idx]) - 1)
d.loc[idx + 1, "ret"] = 0.0
d.loc[idx, "adjustment"] = "reverted spike: 2-day compounded return"
d.loc[idx + 1, "adjustment"] = "reverted spike: return 0"
log.info(f"R4 reverting spikes (|r|>{SPIKE:.0%}): {len(idx)} ({d.loc[idx, 'source'].value_counts().to_dict()})")

# ── R5 ──
d["dollar_vol"] = d["price"] * d["volume"]
# Price adjusted for dividends and splits (total-return index), anchored to the price of the fund's first day
d["adj_price"] = d.groupby("fund")["price"].transform("first") * (1 + d["ret"].fillna(0)).groupby(d["fund"]).cumprod()

# Calendar and summary
spy = pd.read_csv(E / "universe_etf_us.csv").query("TickerSymbol == 'SPY'")["fund"].iloc[0]
cal = d.loc[d["fund"] == spy, "date"]
log.info(f"Calendar (days with an SPY price): {len(cal)} days, {cal.min().date()} → {cal.max().date()}")
off_cal = ~d["date"].isin(set(cal))
log.info(f"Rows on dates that are not SPY trading days: {int(off_cal.sum()):,} (kept; to review)")

r = d["ret"]
log.info(f"Remaining extreme returns: |r|>20%: {int((r.abs() > .2).sum())}, |r|>50%: {int((r.abs() > .5).sum())}")
u = pd.read_csv(E / "universe_etf_us.csv").set_index("fund")
F = d.groupby("fund").agg(first_date=("date", "min"), last_date=("date", "max"), n_days=("date", "size"),
                          n_days_isin=("source", lambda s: int((s == "isin").sum())),
                          median_dollar_vol=("dollar_vol", "median"))
F = u[["TickerSymbol", "DocumentTitle", "fund_status", "RCSAssetCategoryLeaf"]].join(F, how="left")
F["has_data"] = F["n_days"].notna()
F.to_csv(PANEL / "funds.csv")
log.info(f"Funds with data: {int(F.has_data.sum())} of {len(F)} → {F[F.has_data].fund_status.value_counts().to_dict()}")
log.info(f"Funds without data: {F[~F.has_data].fund_status.value_counts().to_dict()} "
         "(active: launched after 15-May-2026; delisted: mostly Nasdaq ????X symbols)")

d["fund"] = d["fund"].astype("category")
d[["fund", "date", "price", "adj_price", "volume", "dollar_vol", "ret", "source", "adjustment"]].to_parquet(PANEL / "daily.parquet", index=False)
log.info(f"daily.parquet: {len(d):,} rows, {d['fund'].nunique()} funds")
