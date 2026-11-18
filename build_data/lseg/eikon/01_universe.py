"""STEP 1 — Search download: ALL U.S. fund quotes (active and delisted).

Only downloads and stores the raw result. The rules that decide what is an ETF and which RIC to use
are in 01b_build_universe.py, which runs offline and can be repeated without querying LSEG.

The search returns at most 10,000 rows: it pages by ticker prefix and subdivides when it hits the cap.

Output: build_data/lseg/data/eikon/universe_quotes_raw.parquet  and  build_data/lseg/data/eikon/01_universe.log
Usage:  python build_data/lseg/eikon/01_universe.py
"""
import string, time
import pandas as pd
from common import get_logger, open_session, with_retry, DATA

log = get_logger("01_universe")
ld = open_session(log)
VIEW = ld.discovery.Views.FUND_QUOTES
SEL = ("RIC,TickerSymbol,DocumentTitle,AssetState,IsPrimaryRIC,ExchangeName,ExchangeCode,"
       "RCSAssetCategoryLeaf,ExchangeCountry,AssetCategory,IssueISIN")
TOP = 10000
BASES = {"AC": "ExchangeCountry eq 'USA' and AssetState eq 'AC'",
         "DC": "ExchangeCountry eq 'USA' and AssetState eq 'DC'"}
problems = []


def search(flt):
    return with_retry(lambda: ld.discovery.search(view=VIEW, filter=flt, select=SEL, top=TOP), log, flt[-45:])


def paginate(base, prefix, out):
    r = search(f"{base} and startswith(TickerSymbol,'{prefix}')")
    time.sleep(0.3)
    if r is None:
        problems.append(f"{prefix}: query failed"); return
    if len(r) >= TOP:
        if len(prefix) >= 3:
            problems.append(f"{prefix}: still at the cap with 3 letters"); out.append(r); return
        log.info(f"    {prefix}: {len(r)} (cap) → subdividing")
        for c in string.ascii_uppercase + string.digits:
            paginate(base, prefix + c, out)
        return
    out.append(r)


for pref, ric in (("SP", "SPY"), ("BU", "BUG.O"), ("UN", "UNG"), ("FB", "FBTC.K")):
    t = search(f"{BASES['AC']} and startswith(TickerSymbol,'{pref}')")
    if t is None or ric not in set(t.get("RIC", [])):
        raise SystemExit(f"The test with prefix '{pref}' did not return {ric}. Stopping: check "
                         "build_data/lseg/data/eikon/01_universe.log")
log.info("Preliminary tests OK (SPY, BUG.O, UNG and FBTC.K are found)")

parts = []
for state, base in BASES.items():
    log.info(f"Searching {state} quotes ...")
    out = []
    for c in string.ascii_uppercase + string.digits:
        paginate(base, c, out)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=SEL.split(","))
    log.info(f"  {state}: {len(df)} quotes")
    parts.append(df)
ld.close_session()

q = pd.concat(parts, ignore_index=True).drop_duplicates(subset="RIC")
q["AssetCategory"] = q["AssetCategory"].astype(str)
for c in q.columns:
    if q[c].dtype == object:
        q[c] = q[c].astype("string")
q.to_parquet(DATA / "universe_quotes_raw.parquet", index=False)
log.info(f"Saved {len(q)} quotes in universe_quotes_raw.parquet")
log.info("Download COMPLETE." if not problems else f"WARNING, download INCOMPLETE: {problems}")
