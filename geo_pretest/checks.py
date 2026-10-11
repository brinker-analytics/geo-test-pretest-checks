"""The three pre-launch checks.

Thresholds are Eric Brinker's "how I read it" guidance, not universal rules. Every one of
them is a parameter you can change.

Statistical approximation used throughout: the lift estimate's noise comes from weekly
percentage errors between the treatment series and the scaled control in your own
pre-period history, with weeks treated as independent. That is a deliberately simple,
transparent readiness check. It is not a substitute for the design and analysis methods
in GeoLift, Trimmed Match, or Meridian GeoX.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

# 90% confidence, one-sided (lift only) and 80% power.
ALPHA_ONE_SIDED = 0.10
POWER = 0.80
Z_ONE_SIDED_90_CONF_80_POWER = float(norm.ppf(1 - ALPHA_ONE_SIDED) + norm.ppf(POWER))  # ~2.12
Z_TWO_SIDED_90 = float(norm.ppf(0.95))  # ~1.645, for the placebo interval

# "How I read it" bands (guidance, not rules).
CORR_ACCEPTABLE, CORR_STRONG = 0.90, 0.95
MAPE_ACCEPTABLE, MAPE_STRONG = 0.15, 0.10
MIN_HOLDOUT_DAYS = 14  # the dry-run holdout is about the test length, and at least 2 weeks


# ----------------------------------------------------------------------------- helpers
def _series(wide: pd.DataFrame, geos: Sequence[str]) -> pd.Series:
    missing = [g for g in geos if g not in wide.columns]
    if missing:
        raise KeyError(f"Geos not in data: {missing}")
    if not geos:
        raise ValueError("Pass at least one geo.")
    return wide[list(geos)].sum(axis=1)


def _weekly(s: pd.Series) -> pd.Series:
    """Sum to Monday-Sunday weeks and drop partial weeks at either end."""
    grouped = s.resample("W-SUN")
    sums, counts = grouped.sum(), grouped.count()
    return sums[counts == 7]


def _mape(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual != 0
    return float((np.abs(actual[mask] - predicted[mask]) / actual[mask]).mean())


def _band(value: float, acceptable: float, strong: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "strong" if value >= strong else "acceptable" if value >= acceptable else "weak"
    return "strong" if value <= strong else "acceptable" if value <= acceptable else "weak"


def _scale_factor(treat: pd.Series, ctrl: pd.Series, days: int) -> float:
    t, c = treat.iloc[-days:].sum(), ctrl.iloc[-days:].sum()
    if c <= 0:
        raise ValueError("Control has no sales in the scale-factor period.")
    return float(t / c)


def _weekly_pct_errors(treat: pd.Series, scaled_ctrl: pd.Series) -> pd.Series:
    tw, cw = _weekly(treat), _weekly(scaled_ctrl)
    return (tw / cw - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def _check_test_days(test_days: int) -> None:
    if test_days < 7:
        raise ValueError("test_days is your planned test length in days; use at least 7.")


# ----------------------------------------------------------------------------- check 1
@dataclass
class ComparisonResult:
    test_geos: list
    control_geos: list
    test_days: int
    history_days: int
    scale_days: int
    scale_factor: float
    corr_daily: float
    corr_weekly: float
    mape_daily: float
    mape_weekly: float
    corr_daily_band: str
    corr_weekly_band: str
    mape_weekly_band: str
    history_covers_test: bool
    history_covers_year: bool
    passes: bool
    weekly_only: bool
    notes: list = field(default_factory=list)


def comparison_strength(
    wide: pd.DataFrame,
    test_geos: Sequence[str],
    control_geos: Sequence[str],
    test_days: int,
    scale_days: Optional[int] = None,
) -> ComparisonResult:
    """Check 1a: correlation (daily and weekly, over the full history) and MAPE of the scaled control.

    The scale factor comes from the most recent `scale_days` days, which must be at least as
    long as the planned test (default: exactly the test length).
    """
    _check_test_days(test_days)
    scale_days = scale_days or test_days
    if scale_days < test_days:
        raise ValueError("scale_days must be at least as long as the test.")
    treat, ctrl = _series(wide, test_geos), _series(wide, control_geos)
    k = _scale_factor(treat, ctrl, scale_days)
    scaled = ctrl * k

    corr_d = float(treat.corr(ctrl))
    corr_w = float(_weekly(treat).corr(_weekly(ctrl)))
    mape_d = _mape(treat, scaled)
    mape_w = _mape(_weekly(treat), _weekly(scaled))
    days = len(treat)

    cd = _band(corr_d, CORR_ACCEPTABLE, CORR_STRONG, True)
    cw = _band(corr_w, CORR_ACCEPTABLE, CORR_STRONG, True)
    mw = _band(mape_w, MAPE_ACCEPTABLE, MAPE_STRONG, False)
    weekly_only = cd == "weak" and cw != "weak"
    passes = cd != "weak" and cw != "weak" and mw != "weak" and days >= test_days

    notes = []
    if weekly_only:
        notes.append("Correlation passes only on weekly data: plan a longer test and read it weekly.")
    if cd == "weak" and cw == "weak":
        notes.append("Correlation is below 90% on daily and weekly data: choose different controls.")
    if mw == "weak":
        notes.append("Weekly MAPE of the scaled control is above 15%: the control does not track closely enough.")
    elif mw == "acceptable":
        notes.append("Weekly MAPE is between 10% and 15%: acceptable, under 10% is preferred.")
    if days < test_days:
        notes.append("History is shorter than the test: collect more history first.")
    if days < 365:
        notes.append("History is under a year: it may not include a seasonal peak.")
    return ComparisonResult(
        list(test_geos), list(control_geos), test_days, days, scale_days, k,
        corr_d, corr_w, mape_d, mape_w, cd, cw, mw,
        days >= test_days, days >= 365, passes, weekly_only, notes,
    )


# ----------------------------------------------------------------------------- check 1b
@dataclass
class DryRunResult:
    holdout_days: int
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    scale_factor: float
    placebo_lift: float
    interval_low: float
    interval_high: float
    planned_change: float
    interval_includes_zero: bool
    error_below_planned_change: bool
    passes: bool
    notes: list = field(default_factory=list)


def dry_run(
    wide: pd.DataFrame,
    test_geos: Sequence[str],
    control_geos: Sequence[str],
    test_days: int,
    planned_change: float,
    holdout_days: Optional[int] = None,
) -> DryRunResult:
    """Check 1b: short-form out-of-sample dry run on recent weeks when nothing changed.

    Fit the scale factor on the pre-period (its final `test_days` days), hold back a final
    window about as long as the test (at least 2 weeks), and predict it. Nothing changed in
    that window, so the placebo lift should be about zero.

    Pass: the 90% placebo lift interval includes zero AND the placebo error is smaller than the
    planned change. `planned_change` is your own assumption (the lift you plan to create).
    """
    _check_test_days(test_days)
    if planned_change <= 0:
        raise ValueError("planned_change is the lift you plan to create, as a fraction (e.g. 0.15).")
    holdout_days = max(holdout_days or test_days, MIN_HOLDOUT_DAYS)
    treat, ctrl = _series(wide, test_geos), _series(wide, control_geos)
    if len(treat) < holdout_days + test_days + 28:
        raise ValueError("Not enough history for a dry run: need the holdout plus a pre-period.")
    pre_t, pre_c = treat.iloc[:-holdout_days], ctrl.iloc[:-holdout_days]
    hold_t, hold_c = treat.iloc[-holdout_days:], ctrl.iloc[-holdout_days:]

    k = _scale_factor(pre_t, pre_c, test_days)
    lift = float(hold_t.sum() / (hold_c.sum() * k) - 1.0)
    errors = _weekly_pct_errors(pre_t, pre_c * k)
    weeks = holdout_days / 7.0
    se = float(errors.std(ddof=1) / np.sqrt(weeks))
    low, high = lift - Z_TWO_SIDED_90 * se, lift + Z_TWO_SIDED_90 * se
    incl_zero = low <= 0.0 <= high
    below = abs(lift) < planned_change
    notes = []
    if not incl_zero:
        notes.append("The placebo shows a 'lift' where nothing changed: the comparison is drifting or overfit.")
    if not below:
        notes.append("Placebo error is at least as large as the planned change: the test could not tell them apart.")
    return DryRunResult(
        holdout_days, hold_t.index[0], hold_t.index[-1], k, lift, low, high, planned_change,
        incl_zero, below, incl_zero and below, notes,
    )


# ----------------------------------------------------------------------------- check 2
@dataclass
class SpendResult:
    test_days: int
    design: str
    weekly_error_sd: float
    mde: float
    baseline_treatment_sales: float
    required_incremental_sales: float
    target_iroas: float
    required_spend: float
    gross_margin: Optional[float]
    breakeven_iroas: Optional[float]
    target_below_breakeven: Optional[bool]
    planned_change: Optional[float]
    detectable: Optional[bool]
    spend_in_test_geos: Optional[float]
    holdout_spend_sufficient: Optional[bool]
    comfort_budget: Optional[float]
    within_comfort_budget: Optional[bool]
    notes: list = field(default_factory=list)


def required_spend(
    wide: pd.DataFrame,
    test_geos: Sequence[str],
    control_geos: Sequence[str],
    test_days: int,
    target_iroas: float,
    design: str = "holdout",
    gross_margin: Optional[float] = None,
    planned_change: Optional[float] = None,
    spend_in_test_geos: Optional[float] = None,
    comfort_budget: Optional[float] = None,
) -> SpendResult:
    """Check 2: required spend = MDE x baseline treatment sales over the test window / target iROAS.

    - MDE: from your own noise (weekly % error of the scaled control), 90% confidence,
      80% power, one-sided.
    - Baseline treatment sales: the treatment geos' recent run rate x test_days.
    - target_iroas is revenue-based and is YOUR assumption. If gross_margin is given, it is compared
      with the margin-based breakeven iROAS (1 / gross margin).
    - design='holdout': the spend you turn off in the test geos over the window
      (spend_in_test_geos) must be at least the required spend to create the MDE gap at the
      target iROAS. design='heavy-up': the required spend is extra spend on top of business as usual.
    """
    _check_test_days(test_days)
    if target_iroas <= 0:
        raise ValueError("target_iroas must be positive (revenue per $1 of spend).")
    if design not in ("holdout", "heavy-up"):
        raise ValueError("design must be 'holdout' or 'heavy-up'.")
    treat, ctrl = _series(wide, test_geos), _series(wide, control_geos)
    k = _scale_factor(treat, ctrl, test_days)
    errors = _weekly_pct_errors(treat, ctrl * k)
    sd = float(errors.std(ddof=1))
    mde = Z_ONE_SIDED_90_CONF_80_POWER * sd / np.sqrt(test_days / 7.0)
    baseline = float(treat.iloc[-test_days:].mean() * test_days)
    incr = mde * baseline
    spend = incr / target_iroas

    breakeven = (1.0 / gross_margin) if gross_margin else None
    below_be = (target_iroas < breakeven) if breakeven else None
    detectable = (mde <= planned_change) if planned_change else None
    sufficient = (spend_in_test_geos >= spend) if (design == "holdout" and spend_in_test_geos is not None) else None
    within = (spend <= comfort_budget) if comfort_budget is not None else None

    notes = []
    if design == "holdout":
        notes.append("Holdout: the spend you turn off in the test geos must be at least the required spend.")
    else:
        notes.append("Heavy-up: the required spend is extra spend on top of business as usual.")
    if below_be:
        notes.append("Target iROAS is below the margin-based breakeven: a 'win' at target would still lose money.")
    if detectable is False:
        notes.append("The MDE is larger than the planned change: the test could not detect it. Change the cells, length, or hypothesis.")
    if sufficient is False:
        notes.append("The spend you would turn off is too small to create the MDE gap at the target iROAS.")
    if within is False:
        notes.append("Required spend is above leadership's comfort-zone budget.")
    notes.append("Baseline uses the recent run rate; adjust if the test window falls in a seasonal peak or trough.")
    return SpendResult(
        test_days, design, sd, float(mde), baseline, float(incr), target_iroas, float(spend),
        gross_margin, breakeven, below_be, planned_change, detectable, spend_in_test_geos,
        sufficient, comfort_budget, within, notes,
    )


# ----------------------------------------------------------------------------- check 3
@dataclass
class DecisionChecklist:
    """Answer these with the decision maker BEFORE launch. None means 'not answered yet'."""
    decision: str = ""                                   # the budget decision this test informs
    decision_maker_precleared: Optional[bool] = None     # cleared with the decision maker in advance?
    comfort_budget: Optional[float] = None               # leadership's comfort-zone budget for the test
    would_roll_out_nationally: Optional[bool] = None     # would they roll a winner out nationally?
    stepped_plan_agreed: Optional[bool] = None           # the stepped game plan below is agreed
    null_means_not_proven_agreed: Optional[bool] = None  # everyone agrees a null = "not proven"
    band: float = 0.20                                   # act if iROAS lands within +/- band of expectation


@dataclass
class ChecklistResult:
    status: str  # "go", "fix", "dont_test"
    game_plan: list
    notes: list = field(default_factory=list)


def evaluate_checklist(c: DecisionChecklist, target_iroas: float) -> ChecklistResult:
    lo, hi = target_iroas * (1 - c.band), target_iroas * (1 + c.band)
    plan = [
        f"Measured iROAS between {lo:.2f} and {hi:.2f} (target {target_iroas:.2f} +/- {c.band:.0%}): act on the decision.",
        f"Measured iROAS above {hi:.2f}: make a digestible change (e.g. a measured spend increase) and retest.",
        f"Measured iROAS below {lo:.2f}: make a digestible change and retest, or run a different test.",
        "No significant lift: 'not proven' at this spend level, not 'doesn't work'. A geo test proves or "
        "disproves one effect at one spend level; it gives no response curve.",
    ]
    notes = []
    if c.decision_maker_precleared is False:
        notes.append("Not cleared with the decision maker: there is no point testing what you won't act on.")
    if c.would_roll_out_nationally is False:
        notes.append("Leadership would not roll a winner out nationally: test a different channel.")
    if c.decision_maker_precleared is False or c.would_roll_out_nationally is False:
        return ChecklistResult("dont_test", plan, notes)
    unanswered = [n for n in ("decision_maker_precleared", "would_roll_out_nationally",
                              "stepped_plan_agreed", "null_means_not_proven_agreed")
                  if getattr(c, n) is None]
    if not c.decision:
        unanswered.insert(0, "decision")
    if c.comfort_budget is None:
        unanswered.append("comfort_budget")
    if unanswered:
        notes.append(f"Answer before launch: {', '.join(unanswered)}.")
    if c.stepped_plan_agreed is False or c.null_means_not_proven_agreed is False:
        notes.append("Agree the stepped game plan and what a null result means before launch.")
    ok = not unanswered and c.stepped_plan_agreed and c.null_means_not_proven_agreed
    return ChecklistResult("go" if ok else "fix", plan, notes)


# ----------------------------------------------------------------------------- verdict
def verdict(comp: ComparisonResult, dry: DryRunResult, spend: SpendResult, check3: ChecklistResult):
    """Combine the three checks into GO / FIX / DON'T TEST with reasons."""
    if check3.status == "dont_test":
        return "DON'T TEST", check3.notes
    reasons = []
    if not comp.passes:
        reasons += [n for n in comp.notes if "under a year" not in n and "between 10% and 15%" not in n]
    if not dry.passes:
        reasons += dry.notes
    for flag, note_key in (
        (spend.detectable is False, "MDE is larger"),
        (spend.holdout_spend_sufficient is False, "spend you would turn off"),
        (spend.within_comfort_budget is False, "comfort-zone"),
        (spend.target_below_breakeven is True, "breakeven"),
    ):
        if flag:
            reasons += [n for n in spend.notes if note_key in n]
    if check3.status != "go":
        reasons += check3.notes
    if reasons:
        return "FIX", reasons
    return "GO", ["All three checks pass on the data supplied, with the assumptions you entered."]
