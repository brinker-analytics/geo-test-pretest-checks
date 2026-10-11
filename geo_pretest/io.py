"""Load a GA4-style CSV export of sales by date by geo (e.g. DMA).

There is no GA4 API integration and no credentials: export the report from GA4 as CSV and
point the loader at the file. GA4 CSV exports start with comment lines beginning with '#';
those are skipped. Dates may be YYYYMMDD (GA4's default) or ISO (YYYY-MM-DD).

If your export uses different column names (for example 'Region' or 'City' instead of 'DMA',
or 'Total revenue' instead of 'Purchase revenue'), pass them via the *_col arguments.
"""
from __future__ import annotations

import pandas as pd


def load_ga4_csv(
    path,
    date_col: str = "Date",
    geo_col: str = "DMA",
    sales_col: str = "Purchase revenue",
) -> pd.DataFrame:
    """Return a tidy frame with columns: date (datetime64), geo (str), sales (float)."""
    raw = pd.read_csv(path, comment="#", dtype={date_col: str})
    missing = [c for c in (date_col, geo_col, sales_col) if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found. Columns in file: {list(raw.columns)}. "
            "Pass date_col / geo_col / sales_col to match your export."
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col].str.replace("-", "", regex=False), format="%Y%m%d"),
            "geo": raw[geo_col].astype(str),
            "sales": pd.to_numeric(
                raw[sales_col].astype(str).str.replace(r"[,$]", "", regex=True), errors="coerce"
            ),
        }
    )
    if out["sales"].isna().any():
        raise ValueError(f"{int(out['sales'].isna().sum())} rows have non-numeric sales values.")
    # GA4 can emit duplicate date/geo rows if other dimensions were included; sum them.
    return out.groupby(["date", "geo"], as_index=False)["sales"].sum().sort_values(["date", "geo"])


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy data to a date x geo table. Missing date/geo cells become 0 (no recorded sales)."""
    wide = df.pivot_table(index="date", columns="geo", values="sales", aggfunc="sum")
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full_index).fillna(0.0).rename_axis("date")
