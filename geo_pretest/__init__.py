"""geo_pretest: three checks to run on your own geo-level history before a geo test.

Check 1  comparison strength  (correlation, MAPE of the scaled control, out-of-sample dry run)
Check 2  required spend       (MDE from your own noise, target iROAS, breakeven iROAS)
Check 3  can the business act (decision checklist)

Nothing here produces an MDE, iROAS, or duration from generic inputs. Every number is
computed from the data you load; any value you type in (target iROAS, margin, budget)
is your assumption and is echoed back as such.
"""
from .io import load_ga4_csv, to_wide
from .checks import (
    ComparisonResult, DryRunResult, SpendResult, DecisionChecklist, ChecklistResult,
    comparison_strength, dry_run, required_spend, evaluate_checklist, verdict,
    Z_ONE_SIDED_90_CONF_80_POWER,
)

__all__ = [
    "load_ga4_csv", "to_wide", "ComparisonResult", "DryRunResult", "SpendResult",
    "DecisionChecklist", "ChecklistResult", "comparison_strength", "dry_run",
    "required_spend", "evaluate_checklist", "verdict", "Z_ONE_SIDED_90_CONF_80_POWER",
]
__version__ = "0.1.0"
