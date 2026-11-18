"""Shared utilities for the LSEG (Eikon / Workspace) downloads.

The key is NOT written in the code. It is read from the LSEG_APP_KEY environment variable:
    export LSEG_APP_KEY="your_key"
"""
import os, sys, time, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # build_data/lseg/ (outputs in build_data/lseg/data/)
YAHOO = Path(__file__).resolve().parents[3] / "data" / "yahoo"   # panel of the original article
DATA = ROOT / "data" / "eikon"
DATA.mkdir(parents=True, exist_ok=True)


def get_logger(name):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(DATA / f"{name}.log", encoding="utf-8")):
            h.setFormatter(fmt); log.addHandler(h)
    return log


def open_session(log):
    import lseg.data as ld
    key = os.environ.get("LSEG_APP_KEY", "").strip()
    if not key:
        sys.exit("Set the LSEG_APP_KEY environment variable (LSEG Workspace / Data Library app key).")
    try:
        ses = ld.open_session(app_key=key)
    except Exception as e:
        sys.exit(f"Could not open the session. Is LSEG Workspace running and logged in?\n{e!r}")
    # open_session does not always raise when it fails: check the state and run a real query
    state = str(getattr(ses, "open_state", ""))
    ok = "Opened" in state
    if ok:
        try:
            ok = len(ld.get_history(universe="SPY", fields=["TR.PriceClose"], start="2024-01-02", end="2024-01-05")) > 0
        except Exception:
            ok = False
    if not ok:
        sys.exit(f"The session did NOT open (state: {state or 'unknown'}). "
                 "Open LSEG Workspace, log in and run the command again.")
    log.info("LSEG session open and verified with a test query")
    return ld


def with_retry(fn, log, what, tries=4, base_sleep=2.0):
    """Retries with increasing waits. Returns None if every attempt fails."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = repr(e)[:300]
            if k == tries - 1:
                log.info(f"  FAILED {what}: {msg}")
                return None
            wait = base_sleep * (2 ** k)
            log.info(f"  retry {k+1}/{tries-1} in {wait:.0f}s ({what}): {msg}")
            time.sleep(wait)
