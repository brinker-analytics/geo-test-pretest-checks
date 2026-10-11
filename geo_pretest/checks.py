"""The three pre-launch checks.

Author: Eric Brinker, Brinker Analytics (https://brinkeranalytics.com).

Thresholds are Eric Brinker's "how I read it" guidance, not universal rules. Every one of them
is a parameter you can change.

Noise model (how the MDE is estimated from your own history)
------------------------------------------------------------
The MDE is reported as a RANGE. Decisions (detectability, power score, recommended length) use
the UPPER end. When neither the empirical nor the same-season estimate is available, the upper
end equals the corrected estimate.

1. Out-of-sample weekly errors (rolling): for each week in your history, fit the scale factor
   on the preceding `scale_days` days, predict the next 7 days, and record the % error.
   Their standard deviation is `sd`.
2. Scale-factor error is included:  se = sd_eff * sqrt(7/test_days + 7/scale_days).
3. Dependence between weeks: sd_eff is the larger of
   - sd * sqrt((1+rho)/(1-rho)), rho = lag-1 autocorrelation of the out-of-sample weekly errors
     (floored at 0, capped at RHO_CAP), and
   - the long-run (Newey-West, Bartlett weights, lags up to test_days/7 - 1) SD of those errors,
     which also catches slower drift between test and control over longer tests.
4. Small-sample correction: se is multiplied by t(0.90, n_weeks-1) / z(0.90), because sd is
   estimated from a limited number of weeks.
   This "corrected" se gives MDE = z * se,  z = z(90% one-sided) + z(80% power).
5. Cross-checks that widen the range:
   - empirical fake tests: rolling placebo tests of the full test length inside your history
     (only when history is at least 2x the test length)
   - same-season error (when history covers a year before the planned start)
   The range runs from the lowest to the highest of these estimates; with no cross-check it runs
   from the uncorrected se (step 2 with plain sd) to the corrected one.

Calibration on the seeded synthetic data (tests/test_calibration.py, 28- and 84-day tests, geo
noise autocorrelation 0 / 0.5 / 0.8): the corrected estimate gives 7-13% false positives for a
90% one-sided test and 72-88% power at the MDE. That is a check of the method on synthetic data,
not a guarantee for your data. This is a readiness gate, not a substitute for the design and analysis methods in
GeoLift, Trimmed Match, or Meridian GeoX.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist

ALPHA_ONE_SIDED = 0.10
POWER = 0.80
Z_ALPHA = float(norm.ppf(1 - ALPHA_ONE_SIDED))                      # ~1.2816
Z_ONE_SIDED_90_CONF_80_POWER = float(Z_ALPHA + norm.ppf(POWER))     # ~2.1232
Z_TWO_SIDED_90 = float(norm.ppf(0.95))                               # ~1.645, placebo interval

# "How I read it" bands (guidance, not rules).
CORR_ACCEPTABLE, CORR_STRONG = 0.90, 0.95
MAPE_STRONG, MAPE_ACCEPTABLE, MAPE_CAUTION = 0.10, 0.15, 0.20       # >20% = weak
MIN_HOLDOUT_DAYS = 14
RHO_CAP = 0.8                                                        # caps the inflation at 3x
MAX_RECOMMENDED_DAYS = 364
WEEKLY_ONLY_LENGTH_FACTOR = 1.5   # weekly-only correlation: recommend at least 1.5x the planned length
MIN_RECOMMENDED_DAYS = 14   # never recommend under 2 weeks; your purchase cycle may need longer


# ============================================================================ helpers
def _series(wide: pd.DataFrame, geos: Sequence[str]) -> pd.Series:
    if not geos:
        raise ValueError("Pass at least one geo.")
    missing = [g for g in geos if g not in wide.columns]
    if missing:
        raise KeyError(f"Geos not in data: {missing}")
    # A day is missing for the cell if any geo in it is missing that day.
    return wide[list(geos)].sum(axis=1, min_count=len(geos)).where(wide[list(geos)].notna().all(axis=1))


def _weekly(s: pd.Series) -> pd.Series:
    """Sum to Monday-Sunday weeks; drop weeks with fewer than 7 observed days (partial or gappy)."""
    grouped = s.resample("W-SUN")
    sums, counts = grouped.sum(min_count=1), grouped.count()
    return sums[counts == 7]


def _check_test_days(test_days: int) -> None:
    if test_days < 7 or test_days % 7:
        raise ValueError("test_days must be a whole number of weeks (a multiple of 7, at least 7). "
                         "Weekly reads and the noise model work in full weeks.")


def _ratio_k(t: pd.Series, c: pd.Series) -> float:
    m = t.notna() & c.notna()
    cs = c[m].sum()
    if cs <= 0:
        raise ValueError("Control has no sales in the scale-factor period.")
    return float(t[m].sum() / cs)


def _band_corr(v: float) -> str:
    return "strong" if v >= CORR_STRONG else "acceptable" if v >= CORR_ACCEPTABLE else "weak"


def _band_mape(v: float) -> str:
    if v < MAPE_STRONG:
        return "strong"
    if v <= MAPE_ACCEPTABLE:
        return "acceptable"
    if v <= MAPE_CAUTION:
        return "caution"
    return "weak"


def _mape_wape(actual: pd.Series, pred: pd.Series):
    m = actual.notna() & pred.notna()
    a, p = actual[m], pred[m]
    zero = int((a == 0).sum())
    nz = a != 0
    mape = float((np.abs(a[nz] - p[nz]) / a[nz]).mean()) if nz.any() else float("nan")
    wape = float(np.abs(a - p).sum() / a.sum()) if a.sum() > 0 else float("nan")
    return mape, wape, zero


def _oos_weekly_errors(treat: pd.Series, ctrl: pd.Series, scale_days: int) -> pd.Series:
    """Rolling out-of-sample weekly % errors: k fit on the preceding scale_days days, predict 7 days."""
    out, idx = [], []
    n = len(treat)
    for e in range(scale_days, n - 6, 7):
        tw, cw = treat.iloc[e:e + 7], ctrl.iloc[e:e + 7]
        if tw.isna().any() or cw.isna().any():
            continue
        try:
            k = _ratio_k(treat.iloc[e - scale_days:e], ctrl.iloc[e - scale_days:e])
        except ValueError:
            continue
        denom = cw.sum() * k
        if denom > 0:
            out.append(tw.sum() / denom - 1.0)
            idx.append(treat.index[e])
    return pd.Series(out, index=pd.DatetimeIndex(idx), dtype=float)


def _oos_rho(errs: pd.Series) -> float:
    """Lag-1 autocorrelation of the out-of-sample weekly errors, floored at 0 and capped at RHO_CAP."""
    e = errs.to_numpy()
    if len(e) < 8:
        return 0.0
    e = e - e.mean()
    if np.sum(e * e) == 0:
        return 0.0
    r = float(np.sum(e[1:] * e[:-1]) / np.sum(e * e))
    return float(min(max(r, 0.0), RHO_CAP))


def _long_run_sd(errs: pd.Series, test_weeks: int) -> float:
    """Newey-West long-run SD of the weekly errors (Bartlett weights, lags < test_weeks).

    Negative autocovariances are floored at 0 so this never shrinks the estimate.
    """
    x = errs.to_numpy() - errs.to_numpy().mean()
    n = len(x)
    lr = float(np.sum(x * x) / n)
    for j in range(1, min(test_weeks, n - 1)):
        g = float(np.sum(x[j:] * x[:-j]) / n)
        lr += 2 * (1 - j / test_weeks) * max(g, 0.0)
    return float(np.sqrt(lr * n / (n - 1)))


def _empirical_fake_test_sd(treat, ctrl, test_days, scale_days) -> tuple[float, int]:
    """SD of rolling placebo 'lifts' of the full test length inside the history (step 7 days)."""
    lifts = []
    for e in range(scale_days, len(treat) - test_days + 1, 7):
        tw, cw = treat.iloc[e:e + test_days], ctrl.iloc[e:e + test_days]
        if tw.isna().any() or cw.isna().any():
            continue
        try:
            k = _ratio_k(treat.iloc[e - scale_days:e], ctrl.iloc[e - scale_days:e])
        except ValueError:
            continue
        lifts.append(tw.sum() / (cw.sum() * k) - 1.0)
    return (float(np.std(lifts, ddof=1)), len(lifts)) if len(lifts) >= 3 else (float("nan"), len(lifts))


# ============================================================================ noise model
@dataclass
class NoiseModel:
    test_days: int
    scale_days: int
    weekly_sd: float              # SD of out-of-sample weekly % errors
    n_weeks: int
    rho: float                    # lag-1 autocorrelation of OOS weekly errors (floored at 0, capped)
    ar_inflation: float
    long_run_sd: float            # Newey-West long-run SD of the OOS weekly errors
    small_sample_factor: float    # t(0.90, n_weeks-1) / z(0.90)
    se_base: float                # with scale-factor error, no dependence or small-sample correction
    se_corrected: float           # corrected estimate (scale-factor error + dependence + small sample)
    se_empirical: Optional[float]
    n_fake_tests: int
    se_same_season: Optional[float]
    se_low: float
    se_high: float
    notes: list = field(default_factory=list)

    @property
    def mde(self) -> float:
        return Z_ONE_SIDED_90_CONF_80_POWER * self.se_corrected

    @property
    def mde_range(self) -> tuple:
        return (Z_ONE_SIDED_90_CONF_80_POWER * self.se_low, Z_ONE_SIDED_90_CONF_80_POWER * self.se_high)

    def se_at(self, days: int, conservative: bool = True) -> float:
        """Projected se for another test length (scale period = max(scale_days, days)).

        Assumption: the corrected-to-plain noise ratio (dependence and small-sample corrections) and,
        when conservative, the se_high / se_corrected ratio both carry over unchanged from the
        planned length to other lengths. Re-run noise_model at the new length to check.
        """
        s = max(self.scale_days, days)
        eff = self.se_corrected / np.sqrt(7 / self.test_days + 7 / self.scale_days)
        base = eff * np.sqrt(7 / days + 7 / s)
        ratio = (self.se_high / self.se_corrected) if conservative else 1.0
        return float(base * ratio)


def noise_model(wide, test_geos, control_geos, test_days, scale_days=None,
                planned_start: Optional[pd.Timestamp] = None) -> NoiseModel:
    """Estimate the lift-estimate noise for a test of `test_days` from your own history."""
    _check_test_days(test_days)
    scale_days = scale_days or test_days
    if scale_days < test_days:
        raise ValueError("scale_days must be at least as long as the test.")
    treat, ctrl = _series(wide, test_geos), _series(wide, control_geos)
    errs = _oos_weekly_errors(treat, ctrl, scale_days)
    if len(errs) < 6:
        raise ValueError("Too little history for a noise estimate: need at least 6 out-of-sample weeks "
                         "after the scale-factor period.")
    sd = float(errs.std(ddof=1))
    rho = _oos_rho(errs)
    infl = float(np.sqrt((1 + rho) / (1 - rho)))
    lr_sd = _long_run_sd(errs, test_days // 7)
    tfac = float(t_dist.ppf(1 - ALPHA_ONE_SIDED, len(errs) - 1) / Z_ALPHA)
    scale_term = np.sqrt(7 / test_days + 7 / scale_days)
    base = sd * scale_term
    sd_eff = max(sd * infl, lr_sd)
    corrected = sd_eff * scale_term * tfac
    notes = []

    emp, n_fake = (float("nan"), 0)
    if treat.notna().sum() >= 2 * test_days:
        emp, n_fake = _empirical_fake_test_sd(treat, ctrl, test_days, scale_days)
    else:
        notes.append("History is under 2x the test length: no empirical fake-test estimate.")

    season_se = None
    start = pd.Timestamp(planned_start) if planned_start is not None else treat.index.max() + pd.Timedelta(days=1)
    last_year = start - pd.Timedelta(days=365)
    if treat.index.min() <= last_year - pd.Timedelta(days=scale_days):
        same = errs[(errs.index >= last_year) & (errs.index < last_year + pd.Timedelta(days=test_days))]
        if len(same) >= 2:
            season_se = float(same.std(ddof=1) * (sd_eff / sd if sd > 0 else 1.0) * scale_term * tfac)
            if season_se > corrected * 1.25:
                notes.append("Error in the same season last year was notably higher: expect a larger MDE in that season.")
    window = pd.date_range(start, periods=test_days, freq="D")
    if ((window.month == 11) | (window.month == 12)).any():
        notes.append("The planned window overlaps November-December: holiday peaks make the comparison less stable.")

    candidates = [x for x in (corrected, emp, season_se) if x is not None and np.isfinite(x)]
    if len(candidates) > 1:
        lo, hi = min(candidates), max(candidates)
    else:
        lo, hi = base, corrected
    return NoiseModel(test_days, scale_days, sd, len(errs), rho, infl, lr_sd, tfac, float(base), float(corrected),
                      emp if np.isfinite(emp) else None, n_fake, season_se, float(lo), float(hi), notes)


def power(lift: float, nm: NoiseModel, days: Optional[int] = None, conservative: bool = True) -> float:
    """Power score: P(detect) = Phi(|lift|/se - z_alpha) at 90% one-sided confidence."""
    se = nm.se_at(days or nm.test_days, conservative=conservative)
    return float(norm.cdf(abs(lift) / se - Z_ALPHA))


def days_needed(lift: float, nm: NoiseModel, target_power: float = POWER, conservative: bool = True) -> Optional[int]:
    """Shortest test (whole weeks, at least 2) reaching the target power for this lift.

    Returns None if no length up to MAX_RECOMMENDED_DAYS gets there. This is a power floor only:
    the customer consideration cycle can require a longer test.
    """
    for d in range(MIN_RECOMMENDED_DAYS, MAX_RECOMMENDED_DAYS + 1, 7):
        if power(lift, nm, d, conservative) >= target_power:
            return d
    return None


def weekly_only_min_days(test_days: int) -> int:
    """Minimum length when correlation passes on weekly data only: 1.5x planned, rounded up to whole weeks."""
    return int(np.ceil(WEEKLY_ONLY_LENGTH_FACTOR * test_days / 7) * 7)


# ============================================================================ check 1
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
    corr_daily_recent: float
    corr_weekly_recent: float
    recent_days: int
    mape_daily: float
    mape_weekly: float
    wape_daily: float
    wape_weekly: float
    zero_days_excluded: int
    missing_days: int
    corr_daily_band: str
    corr_weekly_band: str
    mape_weekly_band: str
    history_covers_test: bool
    history_covers_year: bool
    weekly_only: bool
    passes: bool
    notes: list = field(default_factory=list)


def comparison_strength(wide, test_geos, control_geos, test_days, scale_days=None) -> ComparisonResult:
    """Check 1a: correlation (full history and recent ~2x test length; daily and weekly),
    MAPE and WAPE of the scaled control (scale factor from the latest period >= test length)."""
    _check_test_days(test_days)
    scale_days = scale_days or test_days
    if scale_days < test_days:
        raise ValueError("scale_days must be at least as long as the test.")
    treat, ctrl = _series(wide, test_geos), _series(wide, control_geos)
    k = _ratio_k(treat.iloc[-scale_days:], ctrl.iloc[-scale_days:])
    scaled = ctrl * k
    recent = min(len(treat), 2 * test_days)

    cd, cw = float(treat.corr(ctrl)), float(_weekly(treat).corr(_weekly(ctrl)))
    cdr = float(treat.iloc[-recent:].corr(ctrl.iloc[-recent:]))
    cwr = float(_weekly(treat.iloc[-recent:]).corr(_weekly(ctrl.iloc[-recent:])))
    md, wd, zd = _mape_wape(treat, scaled)
    mw, ww, _ = _mape_wape(_weekly(treat), _weekly(scaled))
    observed = int((treat.notna() & ctrl.notna()).sum())
    missing = int(len(treat) - observed)

    bd, bw, bm = _band_corr(cd), _band_corr(cw), _band_mape(mw)
    weekly_only = bd == "weak" and bw != "weak"
    # Weekly correlation must not be weak. Daily-weak with weekly OK passes (read weekly, longer test).
    passes = bw != "weak" and bm != "weak" and observed >= test_days

    notes = []
    if weekly_only:
        notes.append(f"Correlation passes on weekly data only: read the test weekly and plan a longer test "
                     f"(at least {weekly_only_min_days(test_days)} days, {WEEKLY_ONLY_LENGTH_FACTOR:g}x the planned length).")
    if bw == "weak":
        notes.append("Weekly correlation is weak: choose different controls.")
    if _band_corr(cdr) == "weak" and _band_corr(cwr) == "weak" and not (bd == "weak" and bw == "weak"):
        notes.append(f"Correlation over the most recent {recent} days is weak on both daily and weekly data: "
                     "the relationship may have changed recently.")
    if bm == "weak":
        notes.append("Weekly MAPE of the scaled control is above 20%: the control does not track closely enough.")
    elif bm == "caution":
        notes.append("Weekly MAPE is 15-20% (caution): not a failure, but the MDE will be higher and the result less certain.")
    elif bm == "acceptable":
        notes.append("Weekly MAPE is 10-15%: acceptable; under 10% is preferred.")
    if zd:
        notes.append(f"{zd} zero-sales day(s) excluded from MAPE (WAPE includes them).")
    if missing:
        notes.append(f"{missing} day(s) with missing data in the test or control cell were left out (not zero-filled).")
    if observed < test_days:
        notes.append("History is shorter than the test: collect more history first.")
    if len(treat) < 365:
        notes.append("History is under a year: it may not include a seasonal peak.")
    return ComparisonResult(list(test_geos), list(control_geos), test_days, len(treat), scale_days, k,
                            cd, cw, cdr, cwr, recent, md, mw, wd, ww, zd, missing, bd, bw, bm,
                            observed >= test_days, len(treat) >= 365, weekly_only, passes, notes)


# ============================================================================ check 1b
@dataclass
class DryRunResult:
    holdout_days: int
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    scale_factor: float
    placebo_lift: float
    se_corrected: float
    mde: float                    # upper end of the MDE range at the actual test length
    interval_low: float
    interval_high: float
    interval_includes_zero: bool
    placebo_below_mde: bool
    passes: bool
    notes: list = field(default_factory=list)


def dry_run(wide, test_geos, control_geos, test_days, holdout_days=None, scale_days=None) -> DryRunResult:
    """Check 1b: short-form out-of-sample dry run on recent weeks when nothing changed.

    Hold back a final window about as long as the test (at least 2 weeks). The scale factor and the
    noise model are estimated on the pre-period only, so the holdout never leaks into them.
    Pass: |placebo lift| < the UPPER end of the MDE range (at the actual test length) AND the 90% interval
    (corrected se at the holdout length) includes zero.
    """
    _check_test_days(test_days)
    h = max(holdout_days or test_days, MIN_HOLDOUT_DAYS)
    if h % 7:
        raise ValueError("holdout_days must be a multiple of 7.")
    scale_days = max(scale_days or test_days, h)
    pre = wide.iloc[:-h]
    hold = wide.iloc[-h:]
    nm_h = noise_model(pre, test_geos, control_geos, h, scale_days=scale_days)        # interval: holdout length
    nm = noise_model(pre, test_geos, control_geos, test_days, scale_days=scale_days)  # MDE: actual test length
    pt, pc = _series(pre, test_geos), _series(pre, control_geos)
    k = _ratio_k(pt.iloc[-scale_days:], pc.iloc[-scale_days:])
    ht, hc = _series(hold, test_geos), _series(hold, control_geos)
    m = ht.notna() & hc.notna()
    lift = float(ht[m].sum() / (hc[m].sum() * k) - 1.0)
    se = nm_h.se_corrected
    lo, hi = lift - Z_TWO_SIDED_90 * se, lift + Z_TWO_SIDED_90 * se
    incl = lo <= 0.0 <= hi
    mde_upper = nm.mde_range[1]
    below = abs(lift) < mde_upper or (mde_upper == 0.0 and abs(lift) < 1e-12)
    notes = []
    if not incl:
        notes.append("Dry run: the placebo shows a 'lift' where nothing changed (interval excludes zero); "
                     "the comparison is drifting or overfit.")
    if not below:
        notes.append("Dry run: the placebo error is at least as large as the MDE (upper end); the test could not separate "
                     "a real effect from drift.")
    return DryRunResult(h, hold.index[0], hold.index[-1], k, lift, se, mde_upper, lo, hi, incl, below,
                        incl and below, notes)


# ============================================================================ check 2
@dataclass
class SpendResult:
    test_days: int
    design: str
    direction: str
    noise: NoiseModel
    mde: float
    mde_low: float
    mde_high: float
    baseline_treatment_sales: float
    required_incremental_sales: float
    required_spend: float
    required_spend_low: float
    required_spend_high: float
    target_iroas: float
    gross_margin: Optional[float]
    breakeven_iroas: Optional[float]
    target_below_breakeven: Optional[bool]
    planned_change: Optional[float]
    power_at_planned_change: Optional[float]
    power_at_planned_change_point: Optional[float]
    recommended_days: Optional[int]
    detectable: Optional[bool]
    spend_in_test_geos: Optional[float]
    expected_change_from_spend: Optional[float]
    holdout_spend_sufficient: Optional[bool]
    current_spend_in_test_geos: Optional[float]
    comfort_budget: Optional[float]
    within_comfort_budget: Optional[bool]
    notes: list = field(default_factory=list)


def required_spend(wide, test_geos, control_geos, test_days, target_iroas, design="holdout",
                   direction=None, gross_margin=None, planned_change=None, spend_in_test_geos=None,
                   current_spend_in_test_geos=None, comfort_budget=None, scale_days=None,
                   planned_start=None, weekly_only: bool = False) -> SpendResult:
    """Check 2: required spend = MDE x baseline treatment sales over the test window / target iROAS.

    - MDE from your own noise (see the module docstring), 90% confidence, 80% power, one-sided,
      reported as a range; the required spend is reported as the matching range.
    - Baseline treatment sales = the treatment geos' recent run rate x test_days. For a holdout this
      baseline INCLUDES the sales driven by the spend you will turn off; the gap you must create is
      MDE x that baseline, i.e. spend_off x iROAS >= MDE x baseline (direction='decrease').
    - Heavy-up (direction='increase'): the required spend is extra spend on top of business as usual.
      current_spend_in_test_geos enables a sanity check on how big that increase is.
    - target_iroas is revenue-basis and YOUR assumption; with gross_margin it is compared with the
      margin-based breakeven iROAS (1 / gross margin).
    """
    _check_test_days(test_days)
    if target_iroas <= 0:
        raise ValueError("target_iroas must be positive (revenue per $1 of spend).")
    if design not in ("holdout", "heavy-up"):
        raise ValueError("design must be 'holdout' or 'heavy-up'.")
    direction = direction or ("decrease" if design == "holdout" else "increase")
    if direction not in ("increase", "decrease"):
        raise ValueError("direction must be 'increase' or 'decrease'.")
    nm = noise_model(wide, test_geos, control_geos, test_days, scale_days, planned_start)
    treat = _series(wide, test_geos)
    recent = treat.iloc[-test_days:]
    baseline = float(recent.mean() * test_days)
    mde_lo, mde_hi = nm.mde_range
    incr = nm.mde * baseline
    spend, spend_lo, spend_hi = incr / target_iroas, mde_lo * baseline / target_iroas, mde_hi * baseline / target_iroas

    breakeven = (1.0 / gross_margin) if gross_margin else None
    below_be = (target_iroas < breakeven) if breakeven else None
    pw = pwp = rec = det = None
    if planned_change:
        pw = power(planned_change, nm)
        pwp = power(planned_change, nm, conservative=False)
        rec = days_needed(planned_change, nm)
        if weekly_only and rec is not None:
            rec = max(rec, weekly_only_min_days(test_days))
        det = pw >= POWER
    exp_change = suff = None
    if spend_in_test_geos is not None and baseline > 0:
        exp_change = (spend_in_test_geos * target_iroas / baseline) * (-1 if direction == "decrease" else 1)
        suff = spend_in_test_geos >= spend_hi
    within = (spend_hi <= comfort_budget) if comfort_budget is not None else None

    notes = list(nm.notes)
    if weekly_only:
        notes.append(f"Weekly-only correlation: plan at least {weekly_only_min_days(test_days)} days "
                     f"({WEEKLY_ONLY_LENGTH_FACTOR:g}x the planned {test_days}, rounded up to whole weeks) and read the test weekly.")
    if direction == "decrease":
        notes.append("Holdout (decrease): the baseline is the recent run rate with the spend still on, so it includes the sales driven by the spend you turn off; "
                     "that spend x iROAS must create a gap of at least MDE x baseline.")
    else:
        notes.append("Heavy-up (increase): the required spend is extra spend on top of business as usual.")
        if current_spend_in_test_geos is None:
            notes.append("Heavy-up sanity check: enter current spend in the test geos to see how large the increase is.")
        elif current_spend_in_test_geos > 0 and spend_hi > current_spend_in_test_geos:
            notes.append(f"Heavy-up sanity check: the extra spend is {spend_hi / current_spend_in_test_geos:.1f}x current "
                         "spend in the test geos; iROAS may not hold at that scale.")
    if below_be:
        notes.append("Target iROAS is below the margin-based breakeven: a 'win' at target would still lose money.")
    if det is False:
        msg = f"Power at the planned change is {pw:.0%} (below 80%)."
        msg += (f" Recommended length: {rec} days." if rec else
                f" No length up to {MAX_RECOMMENDED_DAYS} days reaches 80%: change the cells or the hypothesis.")
        notes.append(msg)
    if suff is False:
        notes.append("The spend you would turn off is too small to create the MDE gap at the target iROAS.")
    if within is False:
        notes.append("Required spend (upper end of the range) is above leadership's comfort-zone budget.")
    notes.append("Baseline uses the recent run rate; adjust if the test window falls in a seasonal peak or trough.")
    return SpendResult(test_days, design, direction, nm, nm.mde, mde_lo, mde_hi, baseline, incr, spend,
                       spend_lo, spend_hi, target_iroas, gross_margin, breakeven, below_be, planned_change,
                       pw, pwp, rec, det, spend_in_test_geos, exp_change, suff, current_spend_in_test_geos,
                       comfort_budget, within, notes)


# ============================================================================ check 3
@dataclass
class DecisionChecklist:
    """Answer these with the decision maker BEFORE launch. None means 'not answered yet'."""
    decision: str = ""
    decision_maker_precleared: Optional[bool] = None
    comfort_budget: Optional[float] = None
    would_roll_out_nationally: Optional[bool] = None
    stepped_plan_agreed: Optional[bool] = None
    null_means_not_proven_agreed: Optional[bool] = None
    band: float = 0.20


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
                              "stepped_plan_agreed", "null_means_not_proven_agreed") if getattr(c, n) is None]
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


# ============================================================================ verdict
@dataclass
class Verdict:
    label: str            # "GO", "FIX", "DON'T TEST"
    reasons: list         # blocking reasons
    warnings: list        # non-blocking cautions

    def __iter__(self):   # allows: label, reasons, warnings = verdict(...)
        return iter((self.label, self.reasons, self.warnings))


def verdict(comp: ComparisonResult, dry: DryRunResult, spend: SpendResult, check3: ChecklistResult) -> Verdict:
    """Combine the three checks into GO / FIX / DON'T TEST, with blocking reasons and warnings."""
    warnings = []
    for n in comp.notes + spend.noise.notes:
        if any(key in n for key in ("caution", "weekly data only", "10-15%", "zero-sales", "missing data",
                                    "under a year", "November", "same season", "under 3x", "recent")):
            warnings.append(n)
    warnings += [n for n in spend.notes if "sanity check" in n or "Weekly-only correlation" in n]
    if any("Weekly-only correlation" in n for n in spend.notes):   # same advice, keep the specific one
        warnings = [n for n in warnings if "weekly data only" not in n]
    if check3.status == "dont_test":
        return Verdict("DON'T TEST", check3.notes, warnings)
    reasons = []
    if not comp.passes:
        reasons += [n for n in comp.notes if any(k in n for k in ("choose different controls", "above 20%", "shorter than the test"))]
    if not dry.passes:
        reasons += dry.notes
    reasons += [n for n in spend.notes if any(k in n for k in ("Power at the planned change", "too small to create",
                                                                "comfort-zone", "breakeven"))]
    if check3.status != "go":
        reasons += check3.notes
    if reasons:
        return Verdict("FIX", reasons, warnings)
    return Verdict("GO", ["All three checks pass on the data supplied, with the assumptions you entered."], warnings)
