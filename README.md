# geo-test-pretest-checks

Run three checks on your own geo-level sales history before you launch a geo test.

> **Built by [Brinker Analytics](https://brinkeranalytics.com).** Eric Brinker's marketing-measurement consulting practice for DTC brands: geo tests, an MMM, and attribution.
> **Contact:** [eric@brinkeranalytics.com](mailto:eric@brinkeranalytics.com) or the [contact page](https://brinkeranalytics.com/contact).

> **Full guide (coming soon):** [Before You Call a Geo Test Feasible: 3 Pre-Launch Checks (With Code)](https://brinkeranalytics.com/insights/before-you-call-a-geo-test-feasible) on brinkeranalytics.com. It explains the reasoning behind each check.

## What it is, and what it isn't

**It is a pre-launch gate.** It answers one question: should you run this test at all, and with which design? It ends in a verdict:

- **GO:** all three checks pass on your data, with your assumptions.
- **FIX:** something needs to change first: different controls, a longer test, a bigger spend change, or answers from leadership.
- **DON'T TEST:** the business wouldn't act on the result, so test something else.

**It isn't a design or analysis library.** It doesn't do market search, full power simulation, or lift readouts. For those, use the open-source tools these checks are meant to sit in front of:

- [GeoLift](https://github.com/facebookincubator/GeoLift) (Meta)
- [Trimmed Match](https://github.com/google/trimmed_match) (Google)
- [Meridian GeoX](https://developers.google.com/meridian/geox/intro-to-geox) (Google)

Credit to those teams for the methods this builds on.

## The three checks

### 1. Comparison strength: does the control track the test geos?

- **Correlation** between the test geos and the control, on both **daily and weekly** data, over the full history and over the most recent ~2x test length.
  - **Weekly correlation must not be weak.** If the weekly correlation is weak, the check fails: choose different controls.
  - If the daily correlation is weak but the weekly one is OK, the check passes: read the test weekly and plan a longer test. The recommended length is **at least 1.5x the planned length** (rounded up to whole weeks), or longer if the power score needs it. This is a warning, not a FIX on its own.
- **MAPE of the scaled control.** The control is scaled to the test geos with a factor taken from a period **at least as long as the test**. Zero-sales days are left out of MAPE and counted; **WAPE** is reported alongside.
- **Short-form out-of-sample dry run.**
  - The check holds back a final window about as long as the test, with a minimum of 2 weeks (2–4 weeks or more is typical). The scale factor and the noise model are fit on the earlier pre-period only, then the comparison predicts the held-back window.
  - Nothing changed during that window, so the comparison should show about zero lift.
  - **Pass only if** the placebo lift is smaller than the **upper end of the MDE range** for your actual test length (in absolute value) **and** its 90% interval, using the corrected standard error for the holdout length, includes zero.

### 2. Required spend: what does it cost to detect the lift you care about?

```
required spend = MDE × baseline treatment sales over the test window ÷ target iROAS
```

- **MDE range.** The MDE is reported as a **range** (low to high), sized at **90% confidence, 80% power, one-sided**, from your own noise. **Decisions use the upper end.** When neither cross-check below is available, the upper end equals the corrected estimate.
  - **Noise:** the out-of-sample weekly % error of the scaled control, from rolling windows (scale factor fit on the preceding period, next week predicted).
  - **Scale-factor error included:** `se = sd × sqrt(7/test_days + 7/scale_days)`.
  - **Dependence between weeks:** `sd` is replaced by the larger of `sd × sqrt((1+ρ)/(1−ρ))` (ρ = lag-1 autocorrelation of the out-of-sample weekly errors, floored at 0, capped at 0.8) and the long-run (Newey-West) SD of those errors, which also catches slow drift over longer tests.
  - **Small-sample correction:** `se` is multiplied by `t(0.90, weeks−1) ÷ z(0.90)`, because the noise is estimated from a limited number of weeks. `MDE = z × se` is the **corrected estimate**.
  - **Cross-checks:** an empirical version from rolling fake tests of the full test length in your history (when the history is at least 2x the test length), and the same-season error from last year (when you have a year). The range runs from the lowest to the highest estimate; with no cross-check it runs from the uncorrected to the corrected estimate.
  - **Calibration** on the seeded synthetic data (`pytest -m slow`), corrected estimate, 300–360 fake tests per setting:

    | Test length | Geo noise autocorrelation | False positives (target 10%) | Power at the MDE (target 80%) |
    | --- | --- | --- | --- |
    | 28 days | 0 / 0.5 / 0.8 | 11.7% / 10.8% / 8.9% | 82.8% / 83.9% / 84.2% |
    | 84 days | 0 / 0.5 / 0.8 | 10.3% / 9.0% / 11.0% | 83.7% / 83.3% / 84.3% |

    That checks the method on synthetic data, not your data.
- **Power score:** `power = Φ(change ÷ se − z₉₀)`, using the upper end of the range: the chance the test detects your planned change. Under 80%, you get the shortest length (in whole weeks) that reaches 80%; with weekly-only correlation, at least 1.5x the planned length. For other lengths, the corrected-to-plain noise ratio and the upper-to-corrected ratio are assumed to carry over from the planned length; re-run the checks at the new length to confirm.
- **Baseline treatment sales:** the test geos' recent run rate × the test length.
- **Holdout (`direction="decrease"`):** the baseline is the test geos' recent run rate **with the spend still on**, so it **includes** the sales driven by the spend you will turn off. The gap you must create is measured against that baseline: spend turned off × iROAS ≥ MDE × baseline. The test reads a drop, and the expected change is reported as a negative %.
- **Heavy-up (`direction="increase"`):** the result is the extra spend on top of business as usual. Enter current spend in the test geos for a sanity check: if the extra spend is larger than current spend, iROAS may not hold at that scale.
- **Test length:** whole weeks only (a multiple of 7). Weekly reads use Monday–Sunday weeks; weeks with any missing day are dropped.
- **Profitability:** the revenue-basis target iROAS is compared with the margin-based breakeven iROAS (1 ÷ gross margin). If the target is below breakeven, a "win" would still lose money.

### 3. Can the business act on the result?

Answer these with the decision maker **before** launch:

- **Pre-clear the decision.** There's no point testing what you won't act on.
- **Ask leadership** for their comfort-zone budget numbers, and whether they'd roll a winner out nationally. If they wouldn't, test a different channel.
- **Write a stepped game plan before launch:**
  - If the measured iROAS lands within **±20%** of expectation, act.
  - The band is adjustable; it depends on your appetite for change and your volume.
  - If it lands far above or below, make a digestible change and retest, or run a different test.
- **Null = "not proven," not "doesn't work."** A geo test proves or disproves one effect at one spend level. It gives no response curve.

## "How I read it" thresholds

These are guidance from practice, not rules. They're all parameters in `geo_pretest/checks.py`.

| Measure | Acceptable | Preferred | Note |
| --- | --- | --- | --- |
| Correlation (daily and weekly, full history and recent) | above 90% | above 95% | Weekly must not be weak. Daily weak with weekly OK passes: read weekly, run at least 1.5x the planned length. |
| MAPE of the scaled control (weekly) | 10–15% acceptable | under 10% strong | 15–20% is **caution**: not a failure, but the MDE will be higher and the result less certain. Over 20% is weak. |
| Dry run | \|placebo\| < upper end of the MDE range (actual test length) **and** 90% interval includes zero | | Both are required. |
| Power at the planned change | 80% | | Below 80%: lengthen the test, change the cells, or plan a bigger change. |
| Scale-factor period | at least the test length | | |
| History | at least the test length | a year including a seasonal peak | |
| Controls | 1–2 | | Use as many as the data affords; in practice usually one or two. |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/pretest_checks.ipynb      # or: jupyter notebook
```

- **Run headless and keep outputs:** `jupyter nbconvert --to notebook --execute --inplace notebooks/pretest_checks.ipynb`
- **Run the tests:** `pytest -q` (fast). The calibration check is slow: `pytest -m slow -s`.
- **Regenerate the synthetic sample:** `python scripts/make_synthetic_data.py` (add `--ar 0.5` for autocorrelated geo noise; the committed sample uses none)

## Input format

The input is a **GA4-style CSV export of sales by date by DMA**. There's no GA4 API integration and no credentials: export the report from GA4 as CSV and point the notebook at the file.

| Column (default name) | Meaning |
| --- | --- |
| `Date` | Day, as `YYYYMMDD` (GA4's default) or `YYYY-MM-DD` |
| `DMA` (or `Region`, `City`, `Metro`, `Country`) | Geo |
| `Purchase revenue` (or `Total revenue`, `Revenue`, `Transactions`, `Purchases`) | Daily outcome for that day and geo |

- Column names above are auto-detected. Comment lines starting with `#` are skipped, and in multi-section exports the first table with a date and a geo column is used.
- `(not set)` / `(other)` geos and `Grand total` / `Totals` rows are dropped and reported in `df.attrs["load_report"]`.
- Duplicate date and geo rows are summed.
- **Missing days stay missing** (NaN) and are reported in `wide.attrs["gaps"]`; they are never treated as zero sales unless you call `to_wide(df, fill_missing="zero")`.
- To override the detection:

```python
load_ga4_csv("my_export.csv", date_col="Date", geo_col="Region", sales_col="Total revenue")
```

Keep real exports out of the repo. `data/private/` is git-ignored.

## About the numbers

- **Computed from your data only.** The MDE, required spend, and verdict all come from the data you load. Nothing returns an MDE, iROAS, or test duration from generic inputs.
- **Your inputs are assumptions.** Test length, planned change, target iROAS, margin, budgets, and checklist answers are your assumptions. In the notebook they're labeled **EXAMPLE**.
- **The sample data is synthetic.** `data/synthetic_ga4_sales_by_dma.csv` covers 20 fictional DMAs over 455 days and comes from the seeded `scripts/make_synthetic_data.py`. It contains no client data and says nothing about real performance.
  - `Example DMA 17` has a deliberate structural break, so it makes a poor control.
- **The MDE is a corrected estimate, not a promise.** It includes the scale factor's own error and week-to-week autocorrelation, and it is cross-checked against fake tests in your history. It is still an estimate from past noise. Season: if the test window overlaps November–December you get a warning.
- **Not covered:** spillover between geos, concurrent promotions, and regional events. Rule those out separately.

## Repository layout

```
geo_pretest/          small Python module (loader, checks, verdict)
notebooks/            pretest_checks.ipynb (executed, outputs kept)
data/                 synthetic_ga4_sales_by_dma.csv (synthetic sample)
scripts/              make_synthetic_data.py (seeded generator)
tests/                pytest tests on synthetic data (+ slow calibration test)
AGENTS.md             notes for coding agents working on this repo
```

## License

- **Code:** [MIT](LICENSE).
- **Documentation** (this README and the notebook's text): [CC BY 4.0](LICENSE-docs). Please attribute "Eric Brinker, Brinker Analytics" and link to the guide.
- **Citing:** see [CITATION.cff](CITATION.cff).

Written and maintained by Eric Brinker, [Brinker Analytics](https://brinkeranalytics.com). Questions: [eric@brinkeranalytics.com](mailto:eric@brinkeranalytics.com) or the [contact page](https://brinkeranalytics.com/contact).
