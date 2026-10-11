"""Slow calibration test of the noise model on seeded synthetic data (run: pytest -m slow).

For each seed and random 2-vs-2 cell, run back-to-back fake tests (no real effect) after a
140-day history. With the corrected se:
- false-positive rate of the 90% one-sided test should be close to 10% (accept 7-13%)
- power when a true lift equal to the MDE is injected should be close to 80% (accept 72-88%)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from geo_pretest.checks import Z_ALPHA, noise_model  # noqa: E402
from make_synthetic_data import generate  # noqa: E402

N_SEEDS = 20
TEST_DAYS = 28


def simulate(ar: float, seeds: int = N_SEEDS, test_days: int = TEST_DAYS):
    fp = hit = n = 0
    for s in range(seeds):
        df = generate(1000 + s, ar=ar)
        w = df.pivot(index="Date", columns="DMA", values="Purchase revenue")
        w.index = pd.to_datetime(w.index, format="%Y%m%d")
        geos = [g for g in w.columns if g != "Example DMA 17"]
        rng = np.random.default_rng(s)
        for start in range(140, len(w) - test_days + 1, test_days):
            g = rng.choice(geos, 4, replace=False)
            t, c = list(g[:2]), list(g[2:])
            hist, win = w.iloc[:start], w.iloc[start:start + test_days]
            nm = noise_model(hist, t, c, test_days)
            th, ch = hist[t].sum(axis=1), hist[c].sum(axis=1)
            k = th.iloc[-test_days:].sum() / ch.iloc[-test_days:].sum()
            lift = win[t].sum(axis=1).sum() / (win[c].sum(axis=1).sum() * k) - 1
            se = nm.se_corrected
            fp += lift > Z_ALPHA * se
            hit += (1 + lift) * (1 + nm.mde) - 1 > Z_ALPHA * se
            n += 1
    return fp / n, hit / n, n


@pytest.mark.slow
@pytest.mark.parametrize("ar", [0.0, 0.5])
def test_calibration(ar):
    fpr, pw, n = simulate(ar)
    print(f"\nar={ar} n={n} false-positive rate={fpr:.3f} power@MDE={pw:.3f}")
    assert 0.07 <= fpr <= 0.13
    assert 0.72 <= pw <= 0.88
