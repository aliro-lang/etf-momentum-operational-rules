"""
01_download_data.py
====================
Downloads the FULL universe of U.S.-listed ETFs (~4,500+) and builds an
UNBALANCED price panel: each ETF has data from its actual launch date,
not from a common date.

Unbalanced vs square panel:
  - Square: every ETF from the same date → survivorship bias,
    drops the ETFs listed late.
  - Unbalanced: each ETF enters when it has real history available →
    methodologically correct for empirical research.

Ticker sources:
  1. Yahoo Finance Screener API (primary) — returns up to ~4,000-5,000 tickers
  2. Curated supplementary list (~600 key ETFs) — as a backup and to make sure
     the most important ones are included even if the screener fails.

Note on survivorship bias:
  Yahoo Finance has no history for delisted ETFs. For a rigorous paper, this
  bias is stated in the limitations section. Correcting it requires a commercial
  data provider (CRSP Mutual Funds, Bloomberg).

Usage:
    python build_data/yahoo/01_download_data.py

Output:
    data/yahoo/prices_daily.parquet     — unbalanced panel, adjusted prices
    data/yahoo/tickers_universe.csv     — every ticker found
    data/yahoo/coverage_report.csv      — statistics by ETF
    data/yahoo/cohorts_by_year.csv      — how many ETFs enter each year
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]   # repository root
DATA_RAW   = BASE / "data" / "yahoo" / "raw"
DATA_CLEAN = BASE / "data" / "yahoo"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
START_DATE   = "1990-01-01"    # yfinance returns data from the ETF's actual IPO
MIN_WEEKS    = 52              # minimum number of weeks to include in the panel
BATCH_SIZE   = 50              # tickers per yfinance call
SLEEP_BATCH  = 1.5             # seconds between batches
MAX_SCREENER = 6000            # max tickers to request from the Yahoo screener


# ═════════════════════════════════════════════════════════════
# MODULE 1 — TICKERS
# ═════════════════════════════════════════════════════════════

def fetch_tickers_yahoo_screener(max_total: int = MAX_SCREENER) -> list:
    """
    Calls the internal Yahoo Finance screener to get every U.S.-listed ETF,
    paging 250 at a time.

    Returns a list of tickers (strings). If it fails, returns an empty list.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/screener"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://finance.yahoo.com/",
    }

    all_tickers = []
    offset = 0
    page_size = 250

    print("  Querying the Yahoo Finance Screener...")
    while offset < max_total:
        payload = {
            "offset": offset,
            "size": page_size,
            "sortField": "fundnetassets",
            "sortType": "DESC",
            "quoteType": "ETF",
            "query": {
                "operator": "AND",
                "operands": [
                    {"operator": "EQ",  "operands": ["region", "us"]},
                    {"operator": "EQ",  "operands": ["exchange", "NMS"]},
                ]
            },
            "userId": "",
            "userIdType": "guid",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            if resp.status_code != 200:
                # The other exchanges are queried below
                break

            data  = resp.json()
            result = (data.get("finance", {})
                          .get("result", [{}])[0])
            quotes = result.get("quotes", [])
            total  = result.get("total", 0)

            if not quotes:
                break

            batch_tickers = [q["symbol"] for q in quotes
                             if "^" not in q["symbol"]    # exclude indices
                             and "=" not in q["symbol"]   # exclude FX
                             and len(q["symbol"]) <= 6]   # exclude unusual tickers

            all_tickers.extend(batch_tickers)
            offset += page_size

            pct = min(offset, total) / max(total, 1) * 100
            print(f"  Screener: {len(all_tickers):,} tickers retrieved "
                  f"({pct:.0f}% of ~{total:,}) ...", end="\r")

            if offset >= total:
                break
            time.sleep(0.3)

        except Exception as e:
            print(f"\n  Screener error (offset={offset}): {e}")
            break

    # Also query other exchanges (NAS, ASE)
    for exchange in ["NAS", "ASE"]:
        payload_ex = {
            "offset": 0, "size": 500,
            "sortField": "fundnetassets", "sortType": "DESC",
            "quoteType": "ETF",
            "query": {
                "operator": "AND",
                "operands": [
                    {"operator": "EQ", "operands": ["region", "us"]},
                    {"operator": "EQ", "operands": ["exchange", exchange]},
                ]
            },
            "userId": "", "userIdType": "guid",
        }
        try:
            r = requests.post(url, json=payload_ex, headers=headers, timeout=10)
            if r.status_code == 200:
                quotes_ex = (r.json().get("finance", {})
                                     .get("result", [{}])[0]
                                     .get("quotes", []))
                for q in quotes_ex:
                    s = q.get("symbol", "")
                    if "^" not in s and "=" not in s and len(s) <= 6:
                        all_tickers.append(s)
            time.sleep(0.3)
        except Exception:
            pass

    unique = list(dict.fromkeys(all_tickers))  # deduplicate keeping the order
    print(f"\n  Screener done: {len(unique):,} unique tickers found.")
    return unique


def supplementary_tickers() -> list:
    """
    Curated list of ~600 important ETFs as a backup.
    Covers every category relevant to the paper.
    They guarantee that the ETFs with the longest history (and most useful
    for the paper) are always included, whatever the screener returns.
    """
    return [
        # ── Broad U.S. indices ──────────────────────────────
        "SPY","IVV","VOO","QQQ","QQQM","IWM","MDY","VTI","DIA","IJR","IJH",
        "ITOT","SCHB","SPTM","ONEQ","OEF","IWB","IWC",
        # ── SPDR sectors ────────────────────────────────
        "XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLB","XLRE","XLU","XLC",
        # ── Vanguard sectors ─────────────────────────────
        "VGT","VFH","VDE","VHT","VIS","VCR","VDC","VAW","VNQ","VPU","VOX",
        # ── iShares sectors ──────────────────────────────
        "IYW","IYF","IYE","IYH","IYJ","IYC","IYK","IYM","IYR","IDU",
        # ── Sub-sectors / Thematic ──────────────────────
        "SMH","SOXX","XBI","IBB","XHB","XRT","KRE","KBE","KIE","KCE",
        "XOP","OIH","GDX","GDXJ","SIL","RING","GEX",
        "ITB","PKB","XES","JETS","AIRR",
        "HACK","CIBR","BUG","IHAK",
        "ROBO","IRBO","BOTZ","ARKQ",
        "ARKK","ARKG","ARKW","ARKF","PRNT",
        "FINX","KBWB","KBWR","IAI","IAK",
        "PBW","ICLN","TAN","FAN","GRID",
        "MOO","WOOD","SOIL","CROP",
        "IHI","IHF","IHE","IBB","XPH",
        "PSJ","IGV","SKYY","WCLD","CLOU",
        "ESGU","ESGV","DSI","SUSA",
        "METV","UFO","MAGS","AIEQ",
        # ── U.S. style / factor ──────────────────────────────
        "IWD","IWF","IWN","IWO","IWP","IWS",
        "VTV","VUG","VBR","VBK","VOE","VOT",
        "MTUM","QUAL","VLUE","USMV","SIZE","DGRW",
        "VIG","DVY","HDV","SDY","NOBL","DGRO","SCHD",
        "SPLV","SPHQ","SPHB","SPMD","SPYD",
        "RPV","RPG","RFV","RFG","RZV","RZG",
        "FVAL","FQAL","FLRN","FLDM",
        "COWZ","VALQ","DIVO","DIVI",
        "IVE","IVW","IJJ","IJK","IJS","IJT",
        # ── International developed ────────────────────────
        "EFA","VEA","IEFA","SCHF","SPDW",
        "EWJ","EWG","EWU","EWC","EWA","EWL","EWQ",
        "EWP","EWI","EWD","EWN","EWO","EWS","EWH","EWT",
        "EWK","EIRL","EWM","EWY","EWT","EWX",
        "VGK","FEZ","EZU","HEDJ","HEZU",
        "DXJ","DBJP","EWZ","DBEF","DBEU",
        "SCZ","GWX","VSS","FNDC","FNDF",
        "ACWX","CWI","VXUS","IXUS",
        # ── International emerging ───────────────────────────
        "EEM","VWO","IEMG","SCHE","SPEM",
        "EWZ","EWY","MCHI","FXI","ASHR","CNYA",
        "INDA","INDY","PIN","INDL",
        "EWW","EWC","EWZS","BRZU",
        "TUR","EPOL","ERUS","RSX","GREK","ENZL",
        "ARGT","ECH","EPU","GXC","HAO",
        "VNM","THD","EPHE","IDX","EWM","EWA",
        "FM","FLQM","FLQE","DEM","DGS",
        # ── U.S. government bonds ───────────────────────────
        "SHY","IEI","IEF","TLH","TLT","EDV","VGLT","VGIT","VGSH",
        "SCHO","SCHR","SCHQ","GOVT","FNBGX",
        "TIP","VTIP","SCHP","STIP","RINF",
        "ZROZ","STRIPS","PFIX",
        # ── U.S. corporate bonds ────────────────────────
        "AGG","BND","SCHZ","IUSB",
        "LQD","VCIT","IGIB","USIG","QLTA",
        "HYG","JNK","USHY","FALN","HYD","HYEM",
        "MBB","MBS","VMBS","SPMB",
        "VCSH","IGSB","SLQD","FLOT","SRLN",
        "BKLN","PPLT","SNLN",
        # ── International bonds ──────────────────────────
        "EMB","VWOB","PCY","LEMB","CEMB",
        "BNDX","IAGG","BWX","IGOV",
        "HYXU","HYXE","EMHY",
        # ── Commodities / Metals / Energy ───────────────────
        "GLD","IAU","GLDM","BAR","SGOL",
        "SLV","SIVR","PSLV",
        "PPLT","PALL","GLTR",
        "GDX","GDXJ","SIL","SILJ","RING","SGDM",
        "USO","BNO","UCO","DBO","OILK",
        "UNG","BOIL","KOLD","UNL",
        "DBA","CORN","WEAT","SOYB","CANE","VEGI",
        "DBC","PDBC","GSG","CMDT","BCI",
        "CPER","COPX","ICOP",
        "REMX","LIT","PICK","SLX","XME",
        # ── Real Assets / REITs ───────────────────────────────
        "VNQ","IYR","SCHH","REM","MORT",
        "USRT","INDS","SRVR","HOMZ","NURE",
        "PSA","AMT","PLD",
        # ── Strategies / Multi-asset ────────────────────────
        "ACWI","VT","VXUS","URTH",
        "AOA","AOM","AOR","AOK","AOR",
        "NTSX","GDE","RPAR","TRTY",
        "SWAN","BTAL","TAIL","VIXM",
        "PHDG","IVOL","CTA","KMLM","DBMF",
        "LQDH","HYGH","IGBH",
        # ── Leveraged (for comparison) ───────────
        "SSO","UPRO","QLD","TQQQ","SDS","SPXU",
        "USD","DDM","DXD","UDOW","SDOW",
        "TBT","TBF","TTT","PST",
        "UVXY","SVXY","VIXY",
        # ── Inverse ──────────────────────────────────────────
        "SH","DOG","RWM","PSQ",
        "MYY","SBB","SKF","FAZ","FAS",
        # ── Currencies / FX ─────────────────────────────────────
        "UUP","FXE","FXB","FXY","FXF","FXC","FXA",
        "CEW","DBV","ICI",
    ]


# ═════════════════════════════════════════════════════════════
# MODULE 2 — PRICE DOWNLOAD
# ═════════════════════════════════════════════════════════════

def download_batch(tickers: list, start: str = START_DATE) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for a batch of tickers.
    Returns a DataFrame with columns = tickers that had data.
    """
    try:
        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
        if raw.empty:
            return pd.DataFrame()

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        # With a single ticker, yfinance returns a Series
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        return close

    except Exception:
        return pd.DataFrame()


def download_universe(
    tickers: list,
    batch_size: int = BATCH_SIZE,
    sleep: float = SLEEP_BATCH,
) -> pd.DataFrame:
    """
    Downloads every ticker in batches and consolidates them into one DataFrame.
    The result is an UNBALANCED panel: NaN where the ETF did not exist yet
    or was already delisted.
    """
    batches     = [tickers[i:i+batch_size]
                   for i in range(0, len(tickers), batch_size)]
    total       = len(batches)
    all_frames  = []
    ok_count    = 0
    fail_count  = 0

    for idx, batch in enumerate(batches):
        elapsed = f"[{idx+1}/{total}]"
        frame   = download_batch(batch)

        if frame.empty:
            fail_count += len(batch)
        else:
            # Drop columns with less than the minimum amount of data
            frame = frame.loc[:, frame.notna().sum() >= MIN_WEEKS * 5]
            ok_count  += frame.shape[1]
            fail_count += len(batch) - frame.shape[1]
            all_frames.append(frame)

        print(f"  {elapsed} batch downloaded | "
              f"OK={ok_count:,}  No data={fail_count:,}", end="\r")

        if idx < total - 1:
            time.sleep(sleep)

    print()  # new line

    if not all_frames:
        raise RuntimeError("No data was downloaded.")

    # Concatenate horizontally, aligned by date
    combined = pd.concat(all_frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    combined = combined.sort_index()

    return combined


# ═════════════════════════════════════════════════════════════
# MODULE 3 — STATISTICS OF THE UNBALANCED PANEL
# ═════════════════════════════════════════════════════════════

def coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each ETF: actual start date, end date, years of history,
    % completeness over its active period.
    """
    rows = []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < MIN_WEEKS * 5:
            continue
        start = col.index[0]
        end   = col.index[-1]
        n_active = len(prices.loc[start:end])
        pct_comp = round(len(col) / max(n_active, 1) * 100, 1)
        years    = round((end - start).days / 365.25, 1)

        rows.append({
            "ticker":           ticker,
            "start_date":       start.date(),
            "end_date":         end.date(),
            "years_history":    years,
            "n_obs":            len(col),
            "completeness_pct": pct_comp,
        })

    return pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)


def cohorts_by_year(coverage: pd.DataFrame) -> pd.DataFrame:
    """
    How many ETFs enter the universe each year (useful for the paper).
    Shows how the available universe evolves over time.
    """
    coverage["start_year"] = pd.to_datetime(
        coverage["start_date"]).dt.year

    cohorts = coverage.groupby("start_year").agg(
        new_etfs = ("ticker", "count"),
    ).reset_index()

    cohorts["cumulative_etfs"] = cohorts["new_etfs"].cumsum()
    return cohorts


def universe_on_date(prices: pd.DataFrame,
                     date: pd.Timestamp,
                     min_weeks: int = 52) -> list:
    """
    Key function for the strategy: returns the ETFs available on a given
    date that already have at least min_weeks of history.

    This is the correct implementation of the unbalanced panel:
    - Avoids look-ahead bias (does not use ETFs not yet launched)
    - Avoids forward survivorship bias (uses only the active ones)
    - Lets the universe grow organically year by year
    """
    result = []
    prices_to_date = prices.loc[:date]

    for ticker in prices.columns:
        series = prices_to_date[ticker].dropna()
        # Condition 1: the ETF has the minimum history
        if len(series) < min_weeks:
            continue
        # Condition 2: the ETF is still active (it has a price on the current date)
        price_today = prices.loc[date, ticker] if date in prices.index else np.nan
        if pd.isna(price_today):
            continue
        result.append(ticker)

    return result


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("DATABASE CONSTRUCTION — Unbalanced ETF panel")
    print("=" * 65)

    # ── Step 1: full list of tickers ──────────────────────────
    print("\n[1/5] Getting the ticker universe...")

    tickers_screener = fetch_tickers_yahoo_screener(max_total=MAX_SCREENER)
    tickers_suppl    = supplementary_tickers()

    # Union of both lists, deduplicated
    all_tickers = list(dict.fromkeys(tickers_screener + tickers_suppl))
    all_tickers = [t.strip().upper() for t in all_tickers
                   if t and len(t) <= 6 and t.isalpha()]

    print(f"\n  Screener:      {len(tickers_screener):,} tickers")
    print(f"  Supplementary: {len(tickers_suppl):,} tickers")
    print(f"  Total universe: {len(all_tickers):,} unique tickers")

    # Save the full list
    pd.DataFrame({"ticker": all_tickers}).to_csv(
        DATA_CLEAN / "tickers_universe.csv", index=False)
    print(f"  List saved → data/yahoo/tickers_universe.csv")

    # ── Step 2: download prices ───────────────────────────────
    print(f"\n[2/5] Downloading prices (batches of {BATCH_SIZE})...")
    print(f"  This may take {len(all_tickers)//BATCH_SIZE * SLEEP_BATCH / 60:.0f}–"
          f"{len(all_tickers)//BATCH_SIZE * SLEEP_BATCH / 60 * 2:.0f} minutes.")
    print(f"  Period: {START_DATE} → maximum available per ETF\n")

    prices = download_universe(all_tickers, batch_size=BATCH_SIZE, sleep=SLEEP_BATCH)

    print(f"\n  ETFs with enough data:   {prices.shape[1]:,}")
    print(f"  Date range of the panel: {prices.index[0].date()} → "
          f"{prices.index[-1].date()}")
    print(f"  Total rows × columns:    {prices.shape[0]:,} × {prices.shape[1]:,}")

    # ── Step 3: save the unbalanced panel ─────────────────────
    print("\n[3/5] Saving the price panel...")
    out_prices = DATA_CLEAN / "prices_daily.parquet"
    prices.to_parquet(out_prices)
    size_mb = out_prices.stat().st_size / 1e6
    print(f"  Panel saved → {out_prices}  ({size_mb:.1f} MB)")

    # ── Step 4: coverage report ───────────────────────────────
    print("\n[4/5] Building the coverage report...")
    coverage = coverage_report(prices)
    coverage.to_csv(DATA_CLEAN / "coverage_report.csv", index=False)
    print(f"  Report saved → data/yahoo/coverage_report.csv")

    cohorts = cohorts_by_year(coverage)
    cohorts.to_csv(DATA_CLEAN / "cohorts_by_year.csv", index=False)

    # ── Step 5: summary ───────────────────────────────────────
    print("\n[5/5] SUMMARY OF THE UNBALANCED PANEL")
    print("=" * 65)

    print(f"\nTotal ETFs in the panel:  {prices.shape[1]:,}")
    print(f"ETFs before 2000 (most useful for the paper):  "
          f"{(coverage['start_date'] < '2000-01-01').sum():,}")
    print(f"ETFs before 2005:  "
          f"{(coverage['start_date'] < '2005-01-01').sum():,}")
    print(f"ETFs before 2010:  "
          f"{(coverage['start_date'] < '2010-01-01').sum():,}")

    print(f"\nEvolution of the universe (ETFs available by year):")
    print(cohorts.to_string(index=False))

    print(f"\nETFs with the longest history (top 20):")
    top20 = coverage.nsmallest(20, "start_date")[
        ["ticker", "start_date", "years_history"]]
    print(top20.to_string(index=False))

    # Illustration of the unbalanced panel
    print(f"\n{'='*65}")
    print("ILLUSTRATION OF THE UNBALANCED PANEL")
    print("(universe available on key dates)")
    print(f"{'='*65}")
    weekly = prices.resample("W-FRI").last()
    for date_str in ["1995-12-31", "2000-12-31", "2005-12-31",
                     "2010-12-31", "2015-12-31", "2020-12-31",
                     str(prices.index[-1].date())]:
        try:
            date = pd.Timestamp(date_str)
            if date > weekly.index[-1]:
                date = weekly.index[-1]
            n = len(universe_on_date(weekly, date, min_weeks=52))
            print(f"  {date_str}: {n:>5,} ETFs available "
                  f"(with ≥52 weeks of history)")
        except Exception:
            pass

    print(f"\n✓ Database ready.")
    print(f"  Next step: python build_data/yahoo/30_etf_categories.py 0")
    print(f"\n  METHODOLOGICAL NOTE: this panel does NOT include delisted ETFs")
    print(f"  (survivorship bias). For the paper, state it in the")
    print(f"  'Limitations' section and cite Elton, Gruber & Blake (1996).")
