"""Slow calibration test of the noise model on seeded synthetic data (run: pytest -m slow -s).

For each seed and a random 2-vs-2 cell, run back-to-back fake tests (no real effect) after a
196-day history, for 28- and 84-day tests and geo-noise autocorrelation 0 / 0.5 / 0.8.
- Corrected estimate: false-positive rate of the 90% one-sided test within 7-13%, power at the
  MDE (a true lift equal to the MDE injected) within 72-88%.
- Upper end of the range (what decisions use): no more than 13% false positives, at least 72% power.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from geo_pretest.checks import Z_ALPHA, Z_ONE_SIDED_90_CONF_80_POWER as Z, noise_model  # noqa: E402
from make_synthetic_data import generate  # noqa: E402

SEEDS = {28: 40, 84: 100}   # about 360 and 300 fake tests per setting
HISTORY_START = 196


def simulate(ar: float, test_days: int, seeds: int):
    res = {"corrected": [0, 0], "upper": [0, 0]}
    n = 0
    for s in range(seeds):
        df = generate(5000 + s, ar=ar)
        w = df.pivot(index="Date", columns="DMA", values="Purchase revenue")
        w.index = pd.to_datetime(w.index, format="%Y%m%d")
        geos = [g for g in w.columns if g != "Example DMA 17"]
        rng = np.random.default_rng(s)
        for start in range(HISTORY_START, len(w) - test_days + 1, test_days):
            g = rng.choice(geos, 4, replace=False)
            t, c = list(g[:2]), list(g[2:])
            hist, win = w.iloc[:start], w.iloc[start:start + test_days]
            nm = noise_model(hist, t, c, test_days)
            th, ch = hist[t].sum(axis=1), hist[c].sum(axis=1)
            k = th.iloc[-test_days:].sum() / ch.iloc[-test_days:].sum()
            lift = win[t].sum(axis=1).sum() / (win[c].sum(axis=1).sum() * k) - 1
            for key, se in (("corrected", nm.se_corrected), ("upper", nm.se_high)):
                res[key][0] += lift > Z_ALPHA * se
                res[key][1] += (1 + lift) * (1 + Z * se) - 1 > Z_ALPHA * se
            n += 1
    return {k: (v[0] / n, v[1] / n) for k, v in res.items()}, n


@pytest.mark.slow
@pytest.mark.parametrize("test_days", [28, 84])
@pytest.mark.parametrize("ar", [0.0, 0.5, 0.8])
def test_calibration(ar, test_days):
    r, n = simulate(ar, test_days, SEEDS[test_days])
    (fpr, pw), (fpr_u, pw_u) = r["corrected"], r["upper"]
    print(f"\nT={test_days} ar={ar} n={n} corrected: FPR={fpr:.3f} power={pw:.3f} | upper: FPR={fpr_u:.3f} power={pw_u:.3f}")
    assert 0.07 <= fpr <= 0.13
    assert 0.72 <= pw <= 0.88
    assert fpr_u <= 0.13 and pw_u >= 0.72
