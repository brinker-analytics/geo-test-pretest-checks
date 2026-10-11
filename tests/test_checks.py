"""Basic tests on synthetic data (no client data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geo_pretest import (  # noqa: E402
    DecisionChecklist, comparison_strength, dry_run, evaluate_checklist, load_ga4_csv,
    required_spend, to_wide, verdict, Z_ONE_SIDED_90_CONF_80_POWER,
)
from make_synthetic_data import generate  # noqa: E402

SAMPLE = ROOT / "data" / "synthetic_ga4_sales_by_dma.csv"


@pytest.fixture(scope="module")
def wide():
    return to_wide(load_ga4_csv(SAMPLE))


def test_sample_matches_seeded_generator():
    regenerated = generate()
    loaded = load_ga4_csv(SAMPLE)
    assert len(loaded) == len(regenerated) == 455 * 20
    assert loaded["geo"].nunique() == 20
    assert np.isclose(loaded["sales"].sum(), regenerated["Purchase revenue"].sum())


def test_loader_handles_iso_dates_and_custom_columns(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"day": ["2026-01-01", "2026-01-02"], "Region": ["A", "A"], "Revenue": ["1,000", "$2"]}).to_csv(p, index=False)
    df = load_ga4_csv(p, date_col="day", geo_col="Region", sales_col="Revenue")
    assert df["sales"].tolist() == [1000.0, 2.0]


def test_z_value_is_one_sided_90_conf_80_power():
    assert Z_ONE_SIDED_90_CONF_80_POWER == pytest.approx(1.2816 + 0.8416, abs=1e-3)


def test_identical_series_are_a_perfect_comparison(wide):
    w = wide.copy()
    w["Twin"] = w["Example DMA 05"] * 2.0
    r = comparison_strength(w, ["Twin"], ["Example DMA 05"], test_days=28)
    assert r.corr_daily == pytest.approx(1.0) and r.mape_daily == pytest.approx(0.0, abs=1e-12)
    assert r.scale_factor == pytest.approx(2.0)
    d = dry_run(w, ["Twin"], ["Example DMA 05"], test_days=28, planned_change=0.10)
    assert d.passes and d.placebo_lift == pytest.approx(0.0, abs=1e-12)


def test_broken_control_is_flagged(wide):
    r = comparison_strength(wide, ["Example DMA 03"], ["Example DMA 17"], test_days=84)
    assert not r.passes and r.mape_weekly_band == "weak"


def test_scale_period_must_cover_test(wide):
    with pytest.raises(ValueError):
        comparison_strength(wide, ["Example DMA 03"], ["Example DMA 05"], test_days=84, scale_days=28)


def test_spend_formula(wide):
    s = required_spend(wide, ["Example DMA 03"], ["Example DMA 05", "Example DMA 12"], 84, target_iroas=2.5, gross_margin=0.5)
    assert s.required_spend == pytest.approx(s.mde * s.baseline_treatment_sales / 2.5)
    assert s.breakeven_iroas == pytest.approx(2.0) and s.target_below_breakeven is False


def test_checklist_and_verdict(wide):
    t, c = ["Example DMA 03", "Example DMA 08"], ["Example DMA 05", "Example DMA 12"]
    comp = comparison_strength(wide, t, c, 84)
    dry = dry_run(wide, t, c, 84, 0.15)
    spend = required_spend(wide, t, c, 84, 2.5)
    no = evaluate_checklist(DecisionChecklist(would_roll_out_nationally=False), 2.5)
    assert verdict(comp, dry, spend, no)[0] == "DON'T TEST"
    blank = evaluate_checklist(DecisionChecklist(), 2.5)
    assert verdict(comp, dry, spend, blank)[0] == "FIX"
    assert "not proven" in " ".join(blank.game_plan)
