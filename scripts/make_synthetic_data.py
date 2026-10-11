"""Generate the synthetic sample dataset: daily sales for 20 fictional DMAs, in a GA4-style CSV.

Everything here is invented. No client or real brand data is used, and the numbers say
nothing about real-world performance. The generator is seeded, so the output is reproducible.

Structure:
- 20 fictional geos ("Example DMA 01" ... "Example DMA 20") with different sizes and noise levels
- 2025-01-01 to 2026-03-31 (455 days), so the history covers a year including a holiday peak
- shared weekly pattern, an annual pattern with a Nov-Dec peak, and a shared day-to-day shock
- "Example DMA 17" has a structural break (a level drop from 2025-09-01), so it makes a poor control
- no test has been run in this data (it is pre-test history)

Usage: python scripts/make_synthetic_data.py [output_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261010
N_GEOS = 20
START, END = "2025-01-01", "2026-03-31"


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    geos = [f"Example DMA {i:02d}" for i in range(1, N_GEOS + 1)]

    base = rng.lognormal(mean=np.log(4000), sigma=0.6, size=N_GEOS)  # daily revenue level per geo
    noise_sd = rng.uniform(0.03, 0.10, size=N_GEOS)                   # idiosyncratic daily noise

    weekly = np.array([0.86, 0.92, 0.95, 1.00, 1.12, 1.20, 0.95])[dates.dayofweek]
    doy = dates.dayofyear.to_numpy()
    annual = 1 + 0.10 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    peak = 1 + 0.45 * np.exp(-0.5 * ((doy - 335) / 14.0) ** 2)       # holiday peak around early December
    trend = np.linspace(1.0, 1.08, n)

    shock = np.zeros(n)                                              # shared AR(1) day-to-day shock
    for t in range(1, n):
        shock[t] = 0.6 * shock[t - 1] + rng.normal(0, 0.05)
    common = weekly * annual * peak * trend * np.exp(shock)

    rows = []
    for g, name in enumerate(geos):
        sensitivity = rng.uniform(0.85, 1.15)                        # how strongly the geo follows the shared pattern
        level = base[g] * common ** sensitivity
        idio = np.exp(rng.normal(0, noise_sd[g], size=n))
        sales = level * idio
        if name == "Example DMA 17":
            sales = np.where(dates >= "2025-09-01", sales * 0.70, sales)
        rows.append(pd.DataFrame({"Date": dates.strftime("%Y%m%d"), "DMA": name, "Purchase revenue": sales.round(2)}))
    return pd.concat(rows, ignore_index=True)


def write_ga4_style_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# ----------------------------------------\n"
        "# SYNTHETIC SAMPLE DATA - fictional geos, no client data\n"
        "# Layout mimics a GA4 CSV export: Date x DMA, Purchase revenue\n"
        f"# {START.replace('-', '')}-{END.replace('-', '')}\n"
        "# ----------------------------------------\n"
    )
    with open(path, "w", newline="") as f:
        f.write(header)
        df.to_csv(f, index=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "synthetic_ga4_sales_by_dma.csv"
    write_ga4_style_csv(generate(), out)
    print(f"Wrote {out}")
