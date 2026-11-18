"""STEP 4 — Classification metadata of every fund with data (for super-categories and leveraged/inverse funds).

Fields (diagnostic 0e): TR.FundType, TR.FundGeographicFocus, TR.FundBenchmarkName, TR.FundObjective, TR.FundLaunchDate.
They are current attributes of the fund (no history); they are used only to classify, never as a signal.
Requested by primary RIC; funds without a response are requested again by ISIN.

Usage:  python build_data/lseg/eikon/04_metadata.py      (after 03b, which writes data/panel/funds.csv)
Output: build_data/lseg/data/eikon/metadata_fondos.csv, build_data/lseg/data/eikon/04_metadata.log
        Raw file: its name and layout are kept from the original download so that existing downloads stay
        readable by 03d (index 'fondo' = fund identifier, 'metadata_vacia' = no metadata at all).
"""
import time
import pandas as pd
from common import get_logger, open_session, with_retry, DATA, ROOT

log = get_logger("04_metadata")
FIELDS = ["TR.FundType", "TR.FundGeographicFocus", "TR.FundBenchmarkName", "TR.FundObjective", "TR.FundLaunchDate"]
F = pd.read_csv(ROOT / "data" / "panel" / "funds.csv", index_col=0)
F = F[F["has_data"]]
u = pd.read_csv(DATA / "universe_etf_us.csv").set_index("fund")
F = F.join(u[["RIC", "IssueISIN"]])
log.info(f"Funds with data: {len(F)}")
ld = open_session(log)


def request(ids, label):
    out = []
    for k in range(0, len(ids), 100):
        chunk = ids[k:k + 100]
        r = with_retry(lambda: ld.get_data(universe=chunk, fields=FIELDS), log, f"{label} {k}")
        time.sleep(0.3)
        if r is not None:
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


a = request(F["RIC"].tolist(), "RIC")
a = a.rename(columns={"Instrument": "RIC"})
vals = [c for c in a.columns if c != "RIC"]
a = a[a[vals].notna().any(axis=1) & (a[vals].astype(str).apply(lambda s: s.str.strip()).ne("").any(axis=1))]
m = F.reset_index().merge(a.drop_duplicates("RIC"), on="RIC", how="left").set_index("fund")
missing = m[m[vals].isna().all(axis=1) & m["IssueISIN"].notna()]
log.info(f"With metadata by RIC: {len(F) - len(missing)}; requested by ISIN: {len(missing)}")
if len(missing):
    b = request(missing["IssueISIN"].tolist(), "ISIN").rename(columns={"Instrument": "IssueISIN"})
    b = b.drop_duplicates("IssueISIN").set_index("IssueISIN")
    for c in vals:
        m.loc[missing.index, c] = missing["IssueISIN"].map(b[c]).values
m["metadata_empty"] = m[vals].isna().all(axis=1)
# Raw-file layout kept from the original download: index 'fondo', column 'metadata_vacia'
(m[["TickerSymbol", "RIC", "IssueISIN"] + vals + ["metadata_empty"]]
 .rename(columns={"metadata_empty": "metadata_vacia"}).rename_axis("fondo")
 .to_csv(DATA / "metadata_fondos.csv"))
log.info(f"Without any metadata: {int(m.metadata_empty.sum())} ({m[m.metadata_empty].fund_status.value_counts().to_dict()})")
for c in vals:
    log.info(f"  {c}: {int(m[c].notna().sum())} with a value")
log.info(f"Fund Type: {m['Fund Type'].value_counts(dropna=False).to_dict()}" if "Fund Type" in m else "")
ld.close_session()
