"""STEP 1b — Builds the ETF universe from the raw quotes. Does NOT use LSEG: it can be repeated.

Rules (from diagnostics 0, 0b and 0c; see their logs):
  1. It is an ETF if AssetCategory contains 'ETF' or if the detailed category (RCSAssetCategoryLeaf) is an ETF one
     (e.g. UNG, GLDM and FBTC have AssetCategory 'ECL' but category 'Commodity/Alternative ETF').
     Always excluded: ETNs, closed-end funds, open-end funds.
  2. Excluded: series that are not prices (RIC root with lowercase letters: NAV, IIV...) and OTC quotes.
  3. One fund = one ISIN (if missing: ticker + retirement code).
  4. Primary RIC: active > delisted; then consolidated quote; then IsPrimaryRIC; then listing exchange.
  5. Historical RICs: already retired consolidated quotes of the same fund (change of exchange or ticker),
     one per retirement, to append their earlier history.
  6. Every fund with an ISIN is also requested by ISIN (role 'isin'): this returns the history of the listing
     exchange, including delisted funds and segments before a change of exchange. Volume from a single exchange.

Outputs (build_data/lseg/data/eikon/): universe_etf_us.csv, universe_rics_download.csv, universe_discarded.csv,
         universe_yahoo_not_found.csv, 01b_build_universe.log
Usage:   python build_data/lseg/eikon/01b_build_universe.py
"""
import pandas as pd
from common import get_logger, DATA, ROOT, YAHOO

log = get_logger("01b_build_universe")
q = pd.read_parquet(DATA / "universe_quotes_raw.parquet")
for c in q.columns:
    q[c] = q[c].astype(object).where(q[c].notna(), None)
log.info(f"Raw quotes: {len(q)}")

ETF_LEAF = {"Equity ETF", "Bond ETF", "Alternative ETF", "Commodity ETF", "Mixed Asset ETF", "Other ETF",
            "Money Market ETF", "Currency ETF", "Exchange-Traded Fund",
            "Exchange-Traded Product", "Exchange-Traded Commodity"}
NON_ETF_LEAF = {"Exchange-Traded Note", "Closed-End Fund", "Open-End Fund", "Exchange-Traded Managed Fund"}
LISTING_EXCHANGES = ["NYSE Arca", "NYSE American", "NASDAQ Stock Exchange Global Market", "New York Stock Exchange",
                     "Cboe BZX Exchange", "NASDAQ Stock Exchange Capital Market", "NASDAQ Stock Exchange Global Select Market"]

leaf = q["RCSAssetCategoryLeaf"].fillna("")
cat = q["AssetCategory"].fillna("").astype(str)
exch = q["ExchangeName"].fillna("")
root = q["RIC"].astype(str).str.split(r"[.^]", n=1, regex=True).str[0]
is_etf = (cat.str.contains("ETF") | leaf.isin(ETF_LEAF)) & ~leaf.isin(NON_ETF_LEAF)

reason = pd.Series("", index=q.index)
reason[~is_etf] = "not an ETF: " + leaf.where(leaf != "", cat)
reason[(reason == "") & ~root.str.fullmatch(r"[A-Z0-9]+")] = "series that is not a price (NAV/IIV/other)"
# NAV symbols that Nasdaq disseminates in ticker form: X + 3 letters + X on the Capital Market (e.g. XHAOX.O)
tick = q["TickerSymbol"].fillna("").astype(str)
nav_x = tick.str.fullmatch(r"X[A-Z]{3}X") & exch.str.contains("Capital Market")
nav_title = q["DocumentTitle"].fillna("").str.contains("Indicative|Net Asset Value|Index Value", case=False)
reason[(reason == "") & (nav_x | nav_title)] = "series that is not a price (NAV/IIV/other)"
reason[(reason == "") & exch.str.contains("OTC|Grey|Pink", case=False)] = "OTC quote"
disc = q[reason != ""].assign(reason=reason[reason != ""])
disc.to_csv(DATA / "universe_discarded.csv", index=False)
e = q[reason == ""].copy()
log.info(f"Valid ETF quotes: {len(e)} | discarded: {len(disc)}")
log.info(f"  reasons: {disc['reason'].str.replace(r': .*', '', regex=True).value_counts().to_dict()}")

e["retire_code"] = e["RIC"].astype(str).str.extract(r"\^(\w+)$", expand=False)
# Fund identifier = ISIN; without ISIN, "SIN_ISIN|ticker|retirement code". The "SIN_ISIN" prefix ("no ISIN") is
# kept as is: it is part of the fund identifiers stored in the raw metadata file (04) and in the output tickers.
e["fund"] = e["IssueISIN"].where(e["IssueISIN"].notna(),
                                 "SIN_ISIN|" + e["TickerSymbol"].fillna("").astype(str) + "|" + e["retire_code"].fillna(""))
e["consol"] = e["ExchangeName"].fillna("").str.contains("Consolidated")
e["prim"] = e["IsPrimaryRIC"].astype(str).str.lower().eq("true")
e["active"] = e["AssetState"].eq("AC")
e["listed"] = e["ExchangeName"].isin(LISTING_EXCHANGES)
# Data feeds (BATS One, PITCH, Imbalance...): not exchanges; they aggregate volume from others and would duplicate it
FEEDS = r"BATS One|PITCH|Multicast|Depth|Imbalance|Total View|BASIC|Best Quotes|ArcaBook|Blue Ocean|IMBALANCE"
e["feed"] = e["ExchangeName"].fillna("").str.contains(FEEDS, case=False)
e["pref"] = e["active"] * 16 + (~e["feed"]) * 8 + e["consol"] * 4 + e["prim"] * 2 + e["listed"] * 1

e = e.sort_values(["fund", "pref"], ascending=[True, False])
prim = e.drop_duplicates(subset="fund", keep="first").copy()
g = e.groupby("fund")
prim["fund_status"] = prim["fund"].map(g["active"].any()).map({True: "active", False: "delisted"})
prim["has_consolidated"] = prim["fund"].map(g["consol"].any())

hist = e[~e["active"] & e["consol"] & ~e["RIC"].isin(prim["RIC"])]
hist = hist.sort_values(["fund", "retire_code", "prim"], ascending=[True, True, False]).drop_duplicates(["fund", "retire_code"])
no_consol = set(prim.loc[~prim["has_consolidated"], "fund"])
exchanges = e[e["fund"].isin(no_consol) & ~e["RIC"].isin(prim["RIC"]) & ~e["feed"]]
log.info(f"Data feeds excluded from the exchanges: {int((e['fund'].isin(no_consol) & e['feed']).sum())}")

cols = ["fund", "RIC", "TickerSymbol", "DocumentTitle", "AssetState", "ExchangeName", "RCSAssetCategoryLeaf", "IssueISIN"]
prim[cols + ["fund_status", "has_consolidated"]].to_csv(DATA / "universe_etf_us.csv", index=False)
# Besides the primary RIC, every fund is requested by its ISIN: LSEG resolves it to the listing exchange and returns
# the full history even if the fund is delisted or changed exchange (diagnostic 0d: QQQ since 1999,
# HAO 2008-2020, delisted funds whose secondary-exchange RICs return no data). Volume by ISIN is from a single exchange.
by_isin = prim[prim["IssueISIN"].notna()][cols].copy()
by_isin["RIC"] = by_isin["IssueISIN"]
dl = pd.concat([prim[cols].assign(role="primary"), hist[cols].assign(role="historical"),
                by_isin.assign(role="isin")], ignore_index=True)
dl.to_csv(DATA / "universe_rics_download.csv", index=False)

log.info(f"Funds whose primary RIC is still a feed (they have no other quote): {int(prim['feed'].sum())}")
log.info("─" * 60)
log.info(f"ETF funds: {len(prim)} → {prim['fund_status'].value_counts().to_dict()}")
log.info(f"Funds without any consolidated quote: {len(no_consol)} "
         f"({prim.loc[~prim.has_consolidated, 'fund_status'].value_counts().to_dict()})")
log.info(f"RICs to download: {len(dl)} → {dl['role'].value_counts().to_dict()}")
log.info(f"Categories (active): {prim[prim.fund_status == 'active']['RCSAssetCategoryLeaf'].value_counts().to_dict()}")
for t in ("SPY", "BUG", "UNG", "GLDM", "FBTC", "GLD", "VXX", "AADR"):
    r = prim[prim["TickerSymbol"] == t][["RIC", "fund_status", "RCSAssetCategoryLeaf"]].values.tolist()
    log.info(f"  check {t}: {r if r else 'missing (correct only if it is not an ETF, e.g. VXX is an ETN)'}")

yp = YAHOO / "prices_daily.parquet"
if yp.exists():
    import pyarrow.parquet as pq
    ytk = set(pq.read_schema(yp).names) - {"date", "Date", "__index_level_0__"}
    tk_etf = set(e["TickerSymbol"].dropna()); tk_no = set(disc.loc[disc.reason.str.startswith("not an ETF"), "TickerSymbol"].dropna()) - tk_etf
    missing = sorted(ytk - tk_etf - tk_no)
    log.info(f"Yahoo panel of the paper: {len(ytk)}. As ETF in LSEG: {len(ytk & tk_etf)}. "
             f"As non-ETF (ETN, closed-end fund): {len(ytk & tk_no)}. Not found: {len(missing)}")
    pd.Series(missing, name="ticker").to_csv(DATA / "universe_yahoo_not_found.csv", index=False)
