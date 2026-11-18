"""STEP 3d — Classification of every fund: leveraged/inverse and super-category. Does NOT use LSEG.

Source: LSEG/Lipper metadata (TR.FundType, GeographicFocus, BenchmarkName, Objective) + fund name.
They are current attributes of the fund; used only to classify (filter iv and diversification caps).

Leveraged/inverse (filter iv of the paper): the objective states a multiple or the inverse of the index
("two times", "3x", "-1x", "inverse", "200%"...), or it is a daily-reset product
("daily investment results ... for a single day"), or it tracks volatility (VIX) futures, which the paper
excluded as Trading--Miscellaneous. "Short-Term"/"Short Duration" do not count as inverse.

Super-categories (the 11 of the paper): MONEY_MARKET, GOV_BONDS, CORP_BONDS, COMMODITIES, REITS, EM_EQUITY,
INTL_DM_EQUITY, US_SECTORAL, US_MID_SMALL, US_LARGE_CAP, OTHER. Rules in order; the first one that matches wins.

Input:  build_data/lseg/data/eikon/metadata_fondos.csv (raw, from 04), build_data/lseg/data/panel/funds.csv
Output: build_data/lseg/data/panel/classification.csv
        (fund, TickerSymbol, fund_status, Fund Type, Geographic Focus, lev_inv, lev_inv_reason, super_cat, rule)
        Check: agreement with the classification of the original paper (Yahoo) on the common funds.
"""
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eikon"))
from common import get_logger, ROOT, YAHOO

log = get_logger("03d_classification")
# Raw file of 04_metadata: index 'fondo' (fund identifier), column 'metadata_vacia' (no metadata at all)
m = (pd.read_csv(ROOT / "data" / "eikon" / "metadata_fondos.csv", index_col=0)
     .rename_axis("fund").rename(columns={"metadata_vacia": "metadata_empty"}))
F = pd.read_csv(ROOT / "data" / "panel" / "funds.csv", index_col=0)
m = m.join(F[["DocumentTitle", "fund_status", "RCSAssetCategoryLeaf"]])
txt = lambda c: m[c].fillna("").astype(str)
obj, nam, bm, ftype, geo = txt("Objective"), txt("DocumentTitle"), txt("Benchmark Name"), txt("Fund Type"), txt("Geographic Focus")
alltext = (nam + " | " + bm + " | " + obj)

# ── Leveraged / inverse ──
# The stated objective rules: multiple or inverse of the index + daily reset. The name is only used with
# unambiguous forms (2x, -1x, UltraShort, "ProShares Ultra", Inverse, Bear). "Ultra Short Duration",
# "Ultra Buffer", "Long/Short", "Short-Term", "Interest Rate Hedged" are NOT leveraged or inverse.
MULT = r"\b(?:two|three|four)\s+times\b|\b(?:1\.25|1\.5|1\.75|2|3|4)\s?(?:x|times)\b|\b(?:125|150|175|200|300|400)\s?%|one and one-half times"
INV = r"\binverse\b(?! floater)|\bopposite\b|(?<!long and )(?<!long and  )short exposure|short sales of|short selling securities|\(-[1-4]x\)"
DAILY = r"for a single day|daily investment results|daily leveraged|daily inverse|daily target|daily performance|calendar quarter (?:investment results|performance)|monthly investment results|weekly investment results|(?<!long and )(?<!long and  )short exposure|short sales of|short selling securities"
NAME = (r"\b-?[1-4](?:\.[0-9]+)?x\b|\bUltraShort|\bUltraPro\b|ProShares Ultra\b|ProShares Short\b|\bInverse\b|\bBear\b(?! Stearns)|"
        r"\bLeverag(?:ed|e)\b|Daily (?:Bull|Bear|Target|Inverse)")
VOL = r"\bVIX\b|volatility (?:index )?futures"
flag = pd.DataFrame({
    "multiple": obj.str.contains(MULT, case=False, regex=True) & obj.str.contains(DAILY, case=False, regex=True),
    "inverse": obj.str.contains(INV, case=False, regex=True) & obj.str.contains(DAILY, case=False, regex=True),
    "name": nam.str.contains(NAME, case=True, regex=True),
    "volatility": alltext.str.contains(VOL, case=True, regex=True),
}, index=m.index)
m["lev_inv"] = flag.any(axis=1)
m["lev_inv_reason"] = flag.apply(lambda r: "+".join(k for k, v in r.items() if v), axis=1)

# ── Super-categories ──
EM = {"Global Emerging Markets", "China", "India", "Brazil", "Korea", "Mexico", "Latin America", "Russia", "Taiwan", "South Africa", "Colombia", "Vietnam", "Indonesia", "Greater China",
      "Saudi Arabia", "Argentina", "BRIC", "Poland", "ASEAN", "Greece", "Pakistan", "Eastern Europe", "Thailand", "Chile",
      "Turkey", "Peru", "Philippines", "Malaysia", "United Arab Emirates", "Qatar", "Kuwait", "Egypt", "Africa",
      "Middle East"}
SECTOR = (r"technolog|semiconductor|software|internet|cyber|cloud|artificial intelligence|\bAI\b|robotic|fintech|"
          r"health|biotech|pharma|medical|genomic|financial|bank|insur|regional bank|broker|energy|oil|gas|natural resource|"
          r"\bMLP\b|pipeline|utilit|industrial|aerospace|defense|transport|infrastructure|consumer|retail|homebuild|"
          r"material|metal|mining|miners|gold miners|steel|chemical|communication|telecom|media|gaming|esports|"
          r"clean energy|solar|wind|water|agribusiness|timber|cannabis|lithium|battery|uranium|nuclear|space|"
          r"select sector|sector|thematic|innovation|disruptive|blockchain|crypto|bitcoin|ether")
MIDSMALL = r"mid[- ]?cap|small[- ]?cap|micro[- ]?cap|russell 2000|S&P (?:mid ?cap )?400|S&P (?:small ?cap )?600|russell mid|smid"
GOV = r"treasur|government|\bgovt\b|TIPS|inflation[- ]protected|inflation[- ]linked|mortgage[- ]backed|\bMBS\b|GNMA|agency"
MM = r"money market|treasury bill|t-bill|\b0-3 month|1-3 month|ultra[- ]?short|floating rate treasur|cash management"
COM = r"gold|silver|platinum|palladium|commodit|crude|oil fund|natural gas fund|agricultur|currency|dollar index|"
RE = r"real estate|\bREIT|property"
rules = [
    ("MONEY_MARKET", (ftype == "Money Market") | ((ftype == "Bond") & alltext.str.contains(MM, case=False, regex=True))),
    ("GOV_BONDS", (ftype == "Bond") & (nam + " " + bm).str.contains(GOV, case=False, regex=True)),
    ("CORP_BONDS", ftype == "Bond"),
    ("COMMODITIES", (ftype == "Commodity") | ((ftype == "Alternatives") & (nam + " " + bm).str.contains(COM.rstrip("|"), case=False, regex=True)
                                             & ~m["lev_inv"])),
    ("REITS", (ftype == "Equity") & (nam + " " + bm).str.contains(RE, case=False, regex=True)),
    # Sector and thematic funds with a U.S. or global focus (the paper grouped them in US_SECTORAL)
    ("US_SECTORAL", (ftype == "Equity") & geo.isin({"United States of America", "North America", "Global"})
                    & (nam + " " + bm).str.contains(SECTOR, case=False, regex=True)),
    ("EM_EQUITY", (ftype == "Equity") & geo.isin(EM)),
    ("INTL_DM_EQUITY", (ftype == "Equity") & ~geo.isin(EM | {"United States of America", "North America"})),
    ("US_SECTORAL", (ftype == "Equity") & (nam + " " + bm).str.contains(SECTOR, case=False, regex=True)),
    ("US_MID_SMALL", (ftype == "Equity") & (nam + " " + bm).str.contains(MIDSMALL, case=False, regex=True)),
    ("US_LARGE_CAP", ftype == "Equity"),
]
m["super_cat"] = "OTHER"; m["rule"] = "rest (Alternatives, Mixed Assets, no metadata)"
assigned = pd.Series(False, index=m.index)
for label, cond in rules:
    c = cond & ~assigned
    m.loc[c, "super_cat"] = label; m.loc[c, "rule"] = label
    assigned |= c
out = m[["TickerSymbol", "fund_status", "Fund Type", "Geographic Focus", "lev_inv", "lev_inv_reason", "super_cat", "rule"]]
out.to_csv(ROOT / "data" / "panel" / "classification.csv")
log.info(f"Funds: {len(m)} | leveraged/inverse: {int(m.lev_inv.sum())} → {m.loc[m.lev_inv, 'lev_inv_reason'].value_counts().head(8).to_dict()}")
log.info(f"Super-categories (not leveraged): {m.loc[~m.lev_inv, 'super_cat'].value_counts().to_dict()}")

# ── Check against the classification of the original paper (Yahoo) ──
old = pd.read_parquet(YAHOO / "etf_categories.parquet").set_index("ticker")
c = m.reset_index().set_index("TickerSymbol").join(old, how="inner")
c = c[~c.index.duplicated()]
ct = pd.crosstab(c["is_leveraged_or_inverse"], c["lev_inv"])
log.info(f"Check leveraged vs original paper ({len(c)} common funds):\n{ct}")
new_not_old = c[c.lev_inv & ~c.is_leveraged_or_inverse.astype(bool)]
old_not_new = c[~c.lev_inv & c.is_leveraged_or_inverse.astype(bool)]
log.info(f"  flagged now and not in the paper (sample): {new_not_old.index[:25].tolist()}")
log.info(f"  flagged in the paper and not now: {old_not_new[['yf_category']].to_string()}")
nl = c[~c.lev_inv & ~c.is_leveraged_or_inverse.astype(bool)]
acc = (nl["super_cat"] == nl["super_category"]).mean()
log.info(f"Super-category agreement with the paper (not leveraged): {acc:.1%}")
log.info("\n" + pd.crosstab(nl["super_category"], nl["super_cat"]).to_string())
