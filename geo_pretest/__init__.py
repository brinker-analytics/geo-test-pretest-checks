"""geo_pretest: three checks to run on your own geo-level history before a geo test.

Check 1  comparison strength  (correlation, MAPE of the scaled control, out-of-sample dry run)
Check 2  required spend       (MDE range from your own noise, power score, target vs breakeven iROAS)
Check 3  can the business act (decision checklist)

Nothing here produces an MDE, iROAS, or duration from generic inputs. Every number is
computed from the data you load; any value you type in (target iROAS, margin, budget)
is your assumption and is echoed back as such.

Author: Eric Brinker, Brinker Analytics (https://brinkeranalytics.com).
"""
from .io import load_ga4_csv, to_wide, missing_days
from .checks import (
    ComparisonResult, DryRunResult, SpendResult, DecisionChecklist, ChecklistResult,
    NoiseModel, Verdict, comparison_strength, dry_run, required_spend, evaluate_checklist,
    verdict, noise_model, power, days_needed, weekly_only_min_days, Z_ONE_SIDED_90_CONF_80_POWER, Z_ALPHA,
)

__all__ = [
    "load_ga4_csv", "to_wide", "missing_days", "NoiseModel", "Verdict", "noise_model",
    "power", "days_needed", "weekly_only_min_days", "Z_ALPHA", "ComparisonResult", "DryRunResult", "SpendResult",
    "DecisionChecklist", "ChecklistResult", "comparison_strength", "dry_run",
    "required_spend", "evaluate_checklist", "verdict", "Z_ONE_SIDED_90_CONF_80_POWER",
]
__version__ = "0.2.0"
__author__ = "Eric Brinker, Brinker Analytics"
__maintainer__ = "Eric Brinker <eric@brinkeranalytics.com>"
__url__ = "https://github.com/brinker-analytics/geo-test-pretest-checks"
# Brinker Analytics: https://brinkeranalytics.com (contact: https://brinkeranalytics.com/contact)
