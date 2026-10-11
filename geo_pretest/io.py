"""Load a GA4-style CSV export of sales by date by geo (e.g. DMA).

Author: Eric Brinker, Brinker Analytics (https://brinkeranalytics.com).

There is no GA4 API integration and no credentials: export the report from GA4 as CSV and
point the loader at the file.

Handled export quirks:
- comment lines starting with '#' (GA4's export header) are skipped
- multi-section exports (several tables separated by blank lines): the first table that has a
  date column and a geo column is used
- dates as YYYYMMDD (GA4's default) or YYYY-MM-DD
- geo columns named DMA, Designated market area, Region, City, Metro or Country (auto-detected,
  or pass geo_col)
- outcome columns such as Purchase revenue, Total revenue, Revenue, Transactions, Purchases
  (auto-detected, or pass sales_col). Transactions/purchases work too: the checks only need a
  daily outcome series.
- '(not set)' / '(other)' geos and 'Grand total' / 'Totals' rows are dropped and reported
- numbers with thousands separators or a currency symbol

Missing days stay missing (NaN) unless you explicitly ask for zero-fill. Gaps are reported.
"""
from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
import pandas as pd

DATE_CANDIDATES = ["Date", "Day", "date"]
GEO_CANDIDATES = ["DMA", "Designated market area", "Region", "City", "Metro", "Country", "geo"]
SALES_CANDIDATES = ["Purchase revenue", "Total revenue", "Revenue", "Ecommerce revenue",
                    "Transactions", "Purchases", "Ecommerce purchases", "sales"]
DROP_GEOS = {"(not set)", "(other)", "not set", ""}


def _sections(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    blocks, cur = [], []
    for ln in lines:
        if ln.strip() == "" or set(ln.strip()) <= {","}:
            if cur:
                blocks.append("\n".join(cur)); cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _pick(columns, wanted, candidates, kind):
    if wanted:
        if wanted not in columns:
            raise ValueError(f"{kind} column '{wanted}' not found. Columns: {list(columns)}")
        return wanted
    for c in candidates:
        if c in columns:
            return c
    return None


def load_ga4_csv(path, date_col: str | None = None, geo_col: str | None = None,
                 sales_col: str | None = None) -> pd.DataFrame:
    """Return a tidy frame (date, geo, sales). A load report is attached as df.attrs['load_report']."""
    text = Path(path).read_text(encoding="utf-8-sig")
    chosen = None
    for block in _sections(text):
        raw = pd.read_csv(_io.StringIO(block), dtype=str)
        cols = [c.strip() for c in raw.columns]
        raw.columns = cols
        d = _pick(cols, date_col, DATE_CANDIDATES, "Date") if (date_col is None or date_col in cols) else None
        g = _pick(cols, geo_col, GEO_CANDIDATES, "Geo") if (geo_col is None or geo_col in cols) else None
        if d and g:
            s = _pick(cols, sales_col, SALES_CANDIDATES, "Sales")
            if s is None:
                raise ValueError(f"No sales/outcome column found. Columns: {cols}. Pass sales_col.")
            chosen = (raw, d, g, s)
            break
    if chosen is None:
        raise ValueError("No table with a date column and a geo column was found. Pass date_col and geo_col.")
    raw, d, g, s = chosen

    report = {"date_col": d, "geo_col": g, "sales_col": s, "rows_read": len(raw)}
    dates = raw[d].fillna("").str.strip()
    geos = raw[g].fillna("").str.strip()
    # Only whole-cell total labels: a geo whose name merely contains "total" must not be dropped.
    total_re = r"^(grand\s+)?totals?$"
    total_mask = dates.str.lower().str.match(total_re) | geos.str.lower().str.match(total_re)
    notset_mask = geos.str.lower().isin(DROP_GEOS)
    vals = pd.to_numeric(raw[s].fillna("").str.replace(r"[,$£€\s]", "", regex=True), errors="coerce")
    report["total_rows_dropped"] = int(total_mask.sum())
    report["not_set_rows_dropped"] = int((notset_mask & ~total_mask).sum())
    report["not_set_sales_dropped"] = float(vals[notset_mask & ~total_mask].sum())
    keep = ~(total_mask | notset_mask)
    dates, geos, vals = dates[keep], geos[keep], vals[keep]

    # YYYYMMDD only for exactly 8 digits; anything with '-' is Y-M-D (unpadded allowed). Never strip
    # dashes: '2026-1-15' would become '2026115' and misparse as 2026-11-05.
    parsed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    compact = dates.str.fullmatch(r"\d{8}")
    dashed = dates.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}")
    if compact.any():
        parsed[compact] = pd.to_datetime(dates[compact], format="%Y%m%d", errors="coerce")
    if dashed.any():
        parsed[dashed] = pd.to_datetime(dates[dashed], format="%Y-%m-%d", errors="coerce")
    bad = parsed.isna() | vals.isna()
    if bad.any():
        examples = ", ".join(f"{d!r}" for d in dates[bad].head(3))
        raise ValueError(f"{int(bad.sum())} rows have an unparseable date (expected YYYYMMDD or YYYY-M-D) "
                         f"or non-numeric {s}. Examples of dates in those rows: {examples}")
    out = pd.DataFrame({"date": parsed.values, "geo": geos.values, "sales": vals.values.astype(float)})
    out = out.groupby(["date", "geo"], as_index=False)["sales"].sum().sort_values(["date", "geo"])
    report["rows_used"] = len(out)
    out.attrs["load_report"] = report
    return out.reset_index(drop=True)


def to_wide(df: pd.DataFrame, fill_missing: str | None = None) -> pd.DataFrame:
    """Pivot tidy data to a date x geo table over the full date range.

    Missing date/geo cells stay NaN (missing data is not zero sales) unless fill_missing='zero'
    is passed explicitly. Gaps are reported in wide.attrs['gaps'].
    """
    if fill_missing not in (None, "zero"):
        raise ValueError("fill_missing must be None or 'zero'.")
    wide = df.pivot_table(index="date", columns="geo", values="sales", aggfunc="sum")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full).rename_axis("date")
    wide.columns.name = "geo"
    gaps = {str(c): int(n) for c, n in wide.isna().sum().items() if n}
    if fill_missing == "zero":
        wide = wide.fillna(0.0)
    wide.attrs["gaps"] = gaps
    wide.attrs["fill_missing"] = fill_missing
    wide.attrs["load_report"] = df.attrs.get("load_report", {})
    return wide


def missing_days(wide: pd.DataFrame) -> dict:
    """Days with no data per geo (after any fill)."""
    return {str(c): int(n) for c, n in wide.isna().sum().items() if n}
