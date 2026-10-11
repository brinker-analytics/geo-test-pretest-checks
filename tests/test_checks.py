"""Unit tests on synthetic data (no client data). The slow calibration test is in test_calibration.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geo_pretest import (  # noqa: E402
    DecisionChecklist, Z_ALPHA, Z_ONE_SIDED_90_CONF_80_POWER, comparison_strength, days_needed,
    dry_run, evaluate_checklist, load_ga4_csv, noise_model, power, required_spend, to_wide, verdict,
)
from make_synthetic_data import generate  # noqa: E402

SAMPLE = ROOT / "data" / "synthetic_ga4_sales_by_dma.csv"
T, C = ["Example DMA 03", "Example DMA 08"], ["Example DMA 05", "Example DMA 12"]


@pytest.fixture(scope="module")
def wide():
    return to_wide(load_ga4_csv(SAMPLE))


# ---------------------------------------------------------------- io
def test_sample_matches_seeded_generator():
    regenerated = generate()
    loaded = load_ga4_csv(SAMPLE)
    assert len(loaded) == len(regenerated) == 455 * 20
    assert np.isclose(loaded["sales"].sum(), regenerated["Purchase revenue"].sum())


def test_generator_ar_option_changes_noise_only_when_asked():
    a, b = generate(7), generate(7, ar=0.0)
    assert a.equals(b)
    assert not generate(7, ar=0.6).equals(a)


def test_loader_ga4_edge_cases(tmp_path):
    p = tmp_path / "ga4.csv"
    p.write_text(
        "# GA4 export\n# Start date: 20260101\n\n"
        "Date,Region,Transactions\n"
        "20260101,Texas,\"1,000\"\n20260101,(not set),5\n20260102,Texas,$2\n,Grand total,1007\n\n"
        "Date,Totals\n20260101,9\n"
    )
    df = load_ga4_csv(p)
    rep = df.attrs["load_report"]
    assert df["sales"].tolist() == [1000.0, 2.0]
    assert rep["geo_col"] == "Region" and rep["sales_col"] == "Transactions"
    assert rep["not_set_rows_dropped"] == 1 and rep["total_rows_dropped"] == 1


def test_missing_days_stay_nan_and_are_reported(tmp_path):
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-01", "2026-01-02", "2026-01-03"]),
                       "geo": ["A", "A", "B", "B", "B"], "sales": [1.0, 2.0, 3.0, 4.0, 5.0]})
    w = to_wide(df)
    assert np.isnan(w.loc["2026-01-02", "A"]) and w.attrs["gaps"] == {"A": 1}
    assert to_wide(df, fill_missing="zero").loc["2026-01-02", "A"] == 0.0


def test_missing_days_drop_incomplete_weeks_and_are_counted(wide):
    w = wide.copy()
    w.iloc[100:103, w.columns.get_loc("Example DMA 03")] = np.nan
    r = comparison_strength(w, T, C, 28)
    assert r.missing_days == 3 and any("missing data" in n for n in r.notes)
    assert np.isfinite(r.mape_weekly) and np.isfinite(r.corr_weekly)


def test_zero_sales_days_excluded_from_mape_and_reported(wide):
    w = wide.copy()
    w.iloc[50:52, w.columns.get_loc("Example DMA 03")] = 0.0
    w.iloc[50:52, w.columns.get_loc("Example DMA 08")] = 0.0
    r = comparison_strength(w, T, C, 28)
    assert r.zero_days_excluded == 2 and np.isfinite(r.mape_daily) and np.isfinite(r.wape_daily)


# ---------------------------------------------------------------- noise, power, MDE
def test_z_value_is_one_sided_90_conf_80_power():
    assert Z_ONE_SIDED_90_CONF_80_POWER == pytest.approx(1.2816 + 0.8416, abs=1e-3)


def test_test_days_must_be_whole_weeks(wide):
    with pytest.raises(ValueError):
        comparison_strength(wide, T, C, 30)


def test_scale_factor_error_and_mde_range(wide):
    nm = noise_model(wide, T, C, 28)
    assert nm.se_base == pytest.approx(nm.weekly_sd * np.sqrt(7 / 28 + 7 / 28))
    assert nm.se_corrected == pytest.approx(nm.se_base * np.sqrt((1 + nm.rho) / (1 - nm.rho)))
    lo, hi = nm.mde_range
    assert lo <= nm.mde <= hi
    assert nm.se_empirical is not None  # 455 days >= 3 x 28


def test_power_mde_round_trip(wide):
    nm = noise_model(wide, T, C, 28)
    assert power(nm.mde, nm, conservative=False) == pytest.approx(0.80, abs=1e-6)
    assert power(nm.mde_range[1], nm) == pytest.approx(0.80, abs=1e-6)
    lift = 0.6 * nm.mde
    d = days_needed(lift, nm)
    assert d is not None and d % 7 == 0 and d >= 14 and power(lift, nm, d) >= 0.80
    assert d == 14 or power(lift, nm, d - 7) < 0.80


# ---------------------------------------------------------------- check 1
def test_identical_series_are_a_perfect_comparison(wide):
    w = wide.copy()
    w["Twin"] = w["Example DMA 05"] * 2.0
    r = comparison_strength(w, ["Twin"], ["Example DMA 05"], test_days=28)
    assert r.corr_daily == pytest.approx(1.0) and r.mape_daily == pytest.approx(0.0, abs=1e-12)
    assert r.scale_factor == pytest.approx(2.0)
    d = dry_run(w, ["Twin"], ["Example DMA 05"], test_days=28)
    assert d.passes and d.placebo_lift == pytest.approx(0.0, abs=1e-12)


def test_broken_control_is_flagged(wide):
    r = comparison_strength(wide, ["Example DMA 03"], ["Example DMA 17"], test_days=84)
    assert not r.passes and r.mape_weekly_band == "weak"


def test_weekly_only_correlation_does_not_force_fix(wide):
    r = comparison_strength(wide, T, C, 84)
    assert r.corr_daily_band == "weak" and r.corr_weekly_band != "weak"
    assert r.weekly_only and r.passes
    dr = dry_run(wide, T, C, 84)
    sp = required_spend(wide, T, C, 84, 2.5, planned_change=0.15)
    ok = evaluate_checklist(DecisionChecklist("d", True, 1e9, True, True, True), 2.5)
    v = verdict(r, dr, sp, ok)
    assert v.label == "GO" and any("weekly data only" in w for w in v.warnings)


def test_mape_bands():
    from geo_pretest.checks import _band_mape
    assert [_band_mape(x) for x in (0.05, 0.12, 0.18, 0.25)] == ["strong", "acceptable", "caution", "weak"]


def test_dry_run_has_no_leakage(wide):
    a = dry_run(wide, T, C, 28)
    w = wide.copy()
    w.iloc[-a.holdout_days:] *= 3.0          # change only the holdout, in every geo
    b = dry_run(w, T, C, 28)
    assert a.scale_factor == b.scale_factor and a.se_corrected == b.se_corrected
    w.iloc[-a.holdout_days:, w.columns.get_loc("Example DMA 03")] *= 1.5   # fake lift in a test geo
    c = dry_run(w, T, C, 28)
    assert c.placebo_lift > a.placebo_lift and not c.passes


def test_dry_run_rule_requires_both_conditions(wide):
    d = dry_run(wide, T, C, 28)
    assert d.passes == (abs(d.placebo_lift) < d.mde and d.interval_low <= 0 <= d.interval_high)
    assert d.interval_high - d.interval_low == pytest.approx(2 * 1.6449 * d.se_corrected, rel=1e-3)


# ---------------------------------------------------------------- check 2
def test_required_spend_formula_and_holdout_direction(wide):
    r = required_spend(wide, T, C, 28, target_iroas=2.0, gross_margin=0.6, spend_in_test_geos=1.0)
    assert r.direction == "decrease"
    assert r.required_spend == pytest.approx(r.mde * r.baseline_treatment_sales / 2.0)
    assert r.required_spend_low <= r.required_spend <= r.required_spend_high
    assert r.expected_change_from_spend < 0 and r.holdout_spend_sufficient is False
    assert r.breakeven_iroas == pytest.approx(1 / 0.6) and r.target_below_breakeven is False
    h = required_spend(wide, T, C, 28, target_iroas=2.0, design="heavy-up", spend_in_test_geos=1.0,
                       current_spend_in_test_geos=1.0)
    assert h.direction == "increase" and h.expected_change_from_spend > 0
    assert any("sanity check" in n for n in h.notes)


def test_planned_change_power_and_recommended_length(wide):
    r = required_spend(wide, T, C, 28, target_iroas=2.5, planned_change=0.5 * required_spend(wide, T, C, 28, 2.5).mde_high)
    assert r.detectable is False and r.recommended_days > 28
    assert r.power_at_planned_change == pytest.approx(norm.cdf(r.planned_change / r.noise.se_high - Z_ALPHA))


def test_holiday_overlap_warning(wide):
    nm = noise_model(wide, T, C, 28, planned_start=pd.Timestamp("2025-11-20"))
    assert any("November" in n for n in nm.notes)


# ---------------------------------------------------------------- check 3
def test_checklist_dont_test_when_not_precleared():
    assert evaluate_checklist(DecisionChecklist("x", False, 1.0, True, True, True), 2.5).status == "dont_test"


def test_checklist_band():
    c = evaluate_checklist(DecisionChecklist("x", True, 1.0, True, True, True), 2.5)
    assert c.status == "go" and "2.00 and 3.00" in c.game_plan[0]
