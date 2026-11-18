"""STEP 2 — Daily history of every ETF in the universe: closing price, volume and total return.

Downloaded in batches of ETFs and windows of years. Each (batch, window) is saved to its own
file as soon as it arrives, so if the connection drops or the daily limit is reached, just run
the same command again: it resumes where it stopped and does not repeat what is already downloaded.

Desktop API limits (approx.): 5 requests per second and ~10,000 per day.
By default the script stops by itself after 9,000 requests in a run.

Raw files (their names and columns are kept from the original download so that existing downloads stay
readable and resumable):
  build_data/lseg/data/eikon/history/lote_XXXX_YYYY-YYYY.parquet  long format: date, ric, campo (field), valor (value)
  build_data/lseg/data/eikon/history/_lotes.csv       batch manifest: lote (batch), ric, batch, years
  build_data/lseg/data/eikon/history/_fallidos.csv    what could not be downloaded: lote, desde (from), hasta (to), rics
  build_data/lseg/data/eikon/history/_sin_datos.csv   RICs with no data in a failed window: lote, desde, ric
  build_data/lseg/data/eikon/02_history.log
Usage:  python build_data/lseg/eikon/02_history.py
        python build_data/lseg/eikon/02_history.py --fields TR.PriceClose TR.Volume TR.TotalReturn --batch 20 --years 3
        python build_data/lseg/eikon/02_history.py --test-sample   # targeted sample (~60 funds) in data/eikon/history_prueba/
"""
import argparse, time, datetime as dt
import pandas as pd
from common import get_logger, open_session, with_retry, DATA

ap = argparse.ArgumentParser()
ap.add_argument("--fields", nargs="+", default=["TR.PriceClose", "TR.Volume", "TR.TotalReturn"])
ap.add_argument("--batch", type=int, default=20, help="ETFs per request")
ap.add_argument("--years", type=int, default=3, help="years per window")
ap.add_argument("--start", default="1993-01-01")
ap.add_argument("--end", default="2026-05-15")
ap.add_argument("--max-req", type=int, default=9000, help="cap on requests per run")
ap.add_argument("--sleep", type=float, default=0.35)
ap.add_argument("--test-sample", action="store_true")
args = ap.parse_args()

# Raw-file layout (kept from the original download; see the docstring)
BATCH_FILE = "lote_{:04d}_{}-{}.parquet"               # batch index, first year, last year
LONG_COLS = ["date", "ric", "campo", "valor"]          # date, ric, field, value
MANIFEST_COLS = ["lote", "ric", "batch", "years"]      # batch index, ric, batch size, years per window
FAILED_COLS = ["lote", "desde", "hasta", "rics"]       # batch index, window start, window end, rics
NO_DATA_COLS = ["lote", "desde", "ric"]                # batch index, window start, ric

log = get_logger("02_history")
OUT = DATA / ("history_prueba" if args.test_sample else "history"); OUT.mkdir(exist_ok=True)
FAIL = OUT / "_fallidos.csv"

uni = pd.read_csv(DATA / "universe_rics_download.csv")
if args.test_sample:
    # Targeted sample: known hard cases + 20 active and 20 delisted at random, with all their exchanges
    HARD = ["SPY", "AADR", "AGGM", "AIVI", "GLD", "UNG", "BUG", "HAO", "FINF", "EVAL", "QQQ", "SCHD", "IBIT", "KWT"]
    pr = uni[uni["role"] == "primary"]
    f_hard = set(pr.loc[pr["TickerSymbol"].isin(HARD), "fund"])
    status = pd.read_csv(DATA / "universe_etf_us.csv").set_index("fund")["fund_status"]
    pr = pr.assign(status=pr["fund"].map(status))
    f_rand = set(pr[pr.status == "active"].sample(20, random_state=7)["fund"]) | \
             set(pr[pr.status == "delisted"].sample(20, random_state=7)["fund"])
    uni = uni[uni["fund"].isin(f_hard | f_rand)]
rics = uni["RIC"].dropna().astype(str).drop_duplicates().tolist()
batches = [rics[i:i + args.batch] for i in range(0, len(rics), args.batch)]
y0, y1 = int(args.start[:4]), int(args.end[:4])
windows = [(max(f"{y}-01-01", args.start), min(f"{y + args.years - 1}-12-31", args.end))
           for y in range(y0, y1 + 1, args.years)]
# Batch manifest: if the universe changed since the last run, the files already downloaded would not match
man = OUT / "_lotes.csv"
cur = pd.DataFrame([(i, r) for i, l in enumerate(batches) for r in l], columns=MANIFEST_COLS[:2])
cur["batch"] = args.batch; cur["years"] = args.years
if man.exists():
    prev = pd.read_csv(man)
    if not prev[MANIFEST_COLS].reset_index(drop=True).equals(cur.reset_index(drop=True)):
        raise SystemExit(f"The universe or the parameters changed since the last download in {OUT}. "
                         "Not mixing files: rename that folder first.")
else:
    cur.to_csv(man, index=False)
total = len(batches) * len(windows)
log.info(f"{len(rics)} RICs → {len(batches)} batches × {len(windows)} windows = {total} requests  fields={args.fields}")

import sys, traceback
sys.excepthook = lambda t, v, tb: log.info("UNEXPECTED ERROR:\n" + "".join(traceback.format_exception(t, v, tb)))
ld = open_session(log)
done = new = consecutive = 0
# Windows that already failed in an earlier run: not repeated in the main loop, they go straight to the second pass
failed = pd.read_csv(FAIL).drop_duplicates(["lote", "desde"]).to_dict("records") if FAIL.exists() else []
already_failed = {(int(r["lote"]), r["desde"]) for r in failed}


def failed_row(bi, a, b, rics_):
    return dict(zip(FAILED_COLS, (bi, a, b, rics_)))


def to_long(df, batch):
    """Normalises the response (one or several columns) to long format."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=LONG_COLS)
    d = df.copy()
    d.index.name = "date"
    if isinstance(d.columns, pd.MultiIndex):
        d = d.stack(level=[0, 1], future_stack=True).rename("valor").reset_index()
        d.columns = LONG_COLS
    elif len(batch) == 1:
        d = d.reset_index().melt(id_vars="date", var_name="campo", value_name="valor"); d["ric"] = batch[0]
    else:  # several RICs and a single field: columns = RIC
        d = d.reset_index().melt(id_vars="date", var_name="ric", value_name="valor"); d["campo"] = args.fields[0]
    d = d[LONG_COLS].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["ric"] = d["ric"].astype(str); d["campo"] = d["campo"].astype(str)
    d["valor"] = pd.to_numeric(d["valor"].astype(object), errors="coerce").astype("float64")
    return d.dropna(subset=["date", "valor"]).reset_index(drop=True)


for bi, batch in enumerate(batches):
    for (a, b) in windows:
        f = OUT / BATCH_FILE.format(bi, a[:4], b[:4])
        done += 1
        if f.exists() or (bi, a) in already_failed:
            continue
        if new >= args.max_req:
            log.info(f"Cap of {args.max_req} requests reached. Run the same command again tomorrow: it resumes by itself.")
            ld.close_session(); raise SystemExit(0)
        df = with_retry(lambda: ld.get_history(universe=batch, fields=args.fields, start=a, end=b, interval="daily"),
                        log, f"batch {bi} {a[:4]}-{b[:4]}")
        new += 1
        time.sleep(args.sleep)
        consecutive = consecutive + 1 if df is None else 0
        if consecutive >= 12:
            # Windows truly empty, or did Workspace go down? Test with SPY, which always has data.
            spy = with_retry(lambda: ld.get_history(universe=["SPY"], fields=["TR.PriceClose"], start="2024-01-02",
                                                    end="2024-01-10", interval="daily"), log, "connection check", tries=3)
            if spy is None or len(spy) == 0:
                del failed[-(consecutive - 1):]          # the previous consecutive failures may also have been the outage
                pd.DataFrame(failed).to_csv(FAIL, index=False)
                log.info("Workspace is not responding (SPY fails too). Stopping without recording this window as failed. "
                         "Open Workspace and run the same command again: it resumes by itself.")
                ld.close_session(); raise SystemExit(2)
            consecutive = 0
        if df is None:
            failed.append(failed_row(bi, a, b, " ".join(batch)))
            pd.DataFrame(failed).to_csv(FAIL, index=False)
            continue
        try:
            long = to_long(df, batch)
            long.to_parquet(f, index=False)      # saved even if empty: window without data
        except Exception:
            import traceback
            log.info(f"  ERROR processing batch {bi} {a[:4]}-{b[:4]} (marked as failed):\n{traceback.format_exc()}")
            log.info(f"  columns received: {list(df.columns)[:12]}  dtypes: {df.dtypes.astype(str).value_counts().to_dict()}")
            failed.append(failed_row(bi, a, b, " ".join(batch)))
            pd.DataFrame(failed).to_csv(FAIL, index=False)
        if new % 50 == 0:
            log.info(f"  {done}/{total} ({done / total:.0%})  new in this run: {new}")

# ── Second pass over failed windows ───────────────────────────────────────
# A batch of ISINs with no data at all in the window returns an error instead of an empty response (before the
# funds exist). To avoid confusing "no data" with "error": if an instrument of the batch has data in the first
# 10 days of the next window, it may have existed earlier → it is requested alone for the failed window. If none,
# the window is saved empty and recorded in _sin_datos.csv. Whatever still fails when requested alone stays in _fallidos.csv.
if FAIL.exists():
    pend = pd.read_csv(FAIL).drop_duplicates(["lote", "desde"])
    remaining, no_data = [], []
    for _, fr in pend.iterrows():
        bi, a, b = int(fr["lote"]), fr["desde"], fr["hasta"]
        f = OUT / BATCH_FILE.format(bi, a[:4], b[:4])
        if f.exists():
            continue
        k = [i for i, (x, _) in enumerate(windows) if x == a]
        batch = batches[bi]
        # A fund may have had data in the failed window if it was trading at the start of the next one (born earlier)
        # or at the end of the previous one (still alive). If a neighbour is missing, everything is requested: nothing
        # is discarded blindly. If the immediate neighbour also failed (several consecutive windows without data), the
        # nearest window with a file in that direction is used: a fund alive in the failed window is still alive at
        # the start of the next window with data.
        def neighbour(step):
            j = k[0] + step
            while 0 <= j < len(windows):
                p = OUT / BATCH_FILE.format(bi, windows[j][0][:4], windows[j][1][:4])
                if p.exists():
                    return pd.read_parquet(p), j
                j += step
            return "edge", None
        if not k:
            candidates = list(batch)
        else:
            (prv, jp), (nxt, jn) = neighbour(-1), neighbour(+1)
            cand = set()
            if not isinstance(nxt, str):
                start = pd.Timestamp(windows[jn][0])
                nxt = nxt[nxt["campo"] == "Price Close"]   # LSEG fills Volume=0 before launch: only the price counts
                cand |= set(nxt.loc[pd.to_datetime(nxt["date"]) <= start + pd.Timedelta(days=10), "ric"])
            if not isinstance(prv, str):
                end = pd.Timestamp(windows[jp][1])
                prv = prv[prv["campo"] == "Price Close"]
                cand |= set(prv.loc[pd.to_datetime(prv["date"]) >= end - pd.Timedelta(days=10), "ric"])
            if isinstance(nxt, str):          # last window: no next neighbour, request everything that stayed alive
                cand = set(batch) if isinstance(prv, str) else cand
            candidates = sorted(cand)
        parts, still_failing = [], []
        for ric in candidates:
            if new >= args.max_req:
                break
            r1 = with_retry(lambda: ld.get_history(universe=[ric], fields=args.fields, start=a, end=b, interval="daily"),
                            log, f"second pass {ric} {a[:4]}", tries=2)
            new += 1; time.sleep(args.sleep)
            if r1 is None:
                still_failing.append(ric)
            else:
                parts.append(to_long(r1, [ric]))
        if new >= args.max_req:
            remaining.append(fr.to_dict()); continue
        out = pd.concat(parts, ignore_index=True) if parts else to_long(None, batch)
        out.to_parquet(f, index=False)
        no_data += [dict(zip(NO_DATA_COLS, (bi, a, r))) for r in batch if r not in candidates]
        remaining += [failed_row(bi, a, b, r) for r in still_failing]
        log.info(f"  second pass batch {bi} {a[:4]}-{b[:4]}: {len(candidates)} requested alone, {len(still_failing)} "
                 f"still failing, {len(batch) - len(candidates)} without data in that window")
    pd.DataFrame(remaining, columns=FAILED_COLS).to_csv(FAIL, index=False)
    if no_data:
        sd = OUT / "_sin_datos.csv"
        prev = pd.read_csv(sd) if sd.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(no_data)]).drop_duplicates().to_csv(sd, index=False)

if args.test_sample:
    # Truncation check: the first ETF of each batch, requested alone, must return the same rows as inside the batch
    for bi, batch in enumerate(batches):
        for (a, b) in windows:
            f = OUT / BATCH_FILE.format(bi, a[:4], b[:4])
            if not f.exists():
                continue
            in_batch_all = pd.read_parquet(f)
            if in_batch_all.empty:
                continue
            r0 = in_batch_all["ric"].value_counts().index[0]          # the RIC with the most data in that window
            in_batch = in_batch_all[in_batch_all["ric"] == r0]
            alone = to_long(with_retry(lambda: ld.get_history(universe=[r0], fields=args.fields, start=a, end=b,
                                                              interval="daily"), log, "check"), [r0])
            ok = len(alone) == len(in_batch)
            log.info(f"  CHECK {r0} {a[:4]}-{b[:4]}: alone={len(alone)} rows, in batch={len(in_batch)}  "
                     + ("OK" if ok else "DIFFERENT: the batch is truncated, lower --batch"))
            break
ld.close_session()
n_f = len(pd.read_csv(FAIL)) if FAIL.exists() else 0
log.info(f"Finished: {done}/{total} windows. Truly pending in _fallidos.csv: {n_f}")
