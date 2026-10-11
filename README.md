# geo-test-pretest-checks

Run three checks on your own geo-level sales history before you launch a geo test.

> **Full guide (coming soon):** [Before You Call a Geo Test Feasible: 3 Pre-Launch Checks (With Code)](https://brinkeranalytics.com/insights/before-you-call-a-geo-test-feasible) on brinkeranalytics.com. It explains the reasoning behind each check.

## What it is, and what it isn't

**It is a pre-launch gate.** It answers one question: should you run this test at all, and with which design? It ends in a verdict:

- **GO:** all three checks pass on your data, with your assumptions.
- **FIX:** something needs to change first: different controls, a longer test, a bigger spend change, or answers from leadership.
- **DON'T TEST:** the business wouldn't act on the result, so test something else.

**It isn't a design or analysis library.** It doesn't do market search, power simulation, or lift readouts. For those, use the open-source tools these checks are meant to sit in front of:

- [GeoLift](https://github.com/facebookincubator/GeoLift) (Meta)
- [Trimmed Match](https://github.com/google/trimmed_match) (Google)
- [Meridian GeoX](https://developers.google.com/meridian/geox/intro-to-geox) (Google)

Credit to those teams for the methods this builds on.

## The three checks

### 1. Comparison strength: does the control track the test geos?

- **Correlation** between the test geos and the control. It's measured over a long date range on both **daily and weekly** data.
- **MAPE of the scaled control.** The control is scaled to the test geos with a factor taken from a period **at least as long as the test**.
- **Short-form out-of-sample dry run.**
  - The check holds back a final window about as long as the test, with a minimum of 2 weeks (2–4 weeks or more is typical). It fits on the earlier pre-period and predicts the held-back window.
  - Nothing changed during that window, so the comparison should show about zero lift.
  - **Pass** if the 90% placebo lift interval includes zero **and** the placebo error is smaller than the planned change.

### 2. Required spend: what does it cost to detect the lift you care about?

```
required spend = MDE × baseline treatment sales over the test window ÷ target iROAS
```

- **MDE:** sized at **90% confidence, 80% power, one-sided** (lift only), from your own noise. It uses the weekly percentage error of the scaled control in your history.
- **Baseline treatment sales:** the test geos' recent run rate × the test length.
- **Holdout:** is the spend you turn off in the test geos large enough to create the MDE gap at the target iROAS?
- **Heavy-up:** the result is the extra spend on top of business as usual.
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
| Correlation (daily and weekly, long range) | above 90% | above 95% | If only weekly passes, plan a longer test and read it weekly. |
| MAPE of the scaled control | under 15% | under 10% | Especially on weekly data. |
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
- **Run the tests:** `pytest -q`
- **Regenerate the synthetic sample:** `python scripts/make_synthetic_data.py`

## Input format

The input is a **GA4-style CSV export of sales by date by DMA**. There's no GA4 API integration and no credentials: export the report from GA4 as CSV and point the notebook at the file.

| Column (default name) | Meaning |
| --- | --- |
| `Date` | Day, as `YYYYMMDD` (GA4's default) or `YYYY-MM-DD` |
| `DMA` | Geo (designated market area) |
| `Purchase revenue` | Sales for that day and geo |

- Comment lines starting with `#`, as in GA4's export header, are skipped.
- Duplicate date and geo rows are summed.
- Missing date and geo combinations count as zero sales.
- If your export uses other column names (for example `Region` or `Total revenue`), pass them:

```python
load_ga4_csv("my_export.csv", date_col="Date", geo_col="Region", sales_col="Total revenue")
```

Keep real exports out of the repo. `data/private/` is git-ignored.

## About the numbers

- **Computed from your data only.** The MDE, required spend, and verdict all come from the data you load. Nothing returns an MDE, iROAS, or test duration from generic inputs.
- **Your inputs are assumptions.** Test length, planned change, target iROAS, margin, budgets, and checklist answers are your assumptions. In the notebook they're labeled **EXAMPLE**.
- **The sample data is synthetic.** `data/synthetic_ga4_sales_by_dma.csv` covers 20 fictional DMAs over 455 days and comes from the seeded `scripts/make_synthetic_data.py`. It contains no client data and says nothing about real performance.
  - `Example DMA 17` has a deliberate structural break, so it makes a poor control.
- **Simplifications:** weeks are treated as independent, and the scale factor's own uncertainty is ignored. Read the MDE as a floor, not a promise.
- **Not covered:** spillover between geos, concurrent promotions, and regional events. Rule those out separately.

## Repository layout

```
geo_pretest/          small Python module (loader, checks, verdict)
notebooks/            pretest_checks.ipynb (executed, outputs kept)
data/                 synthetic_ga4_sales_by_dma.csv (synthetic sample)
scripts/              make_synthetic_data.py (seeded generator)
tests/                pytest tests on synthetic data
```

## License

- **Code:** [MIT](LICENSE).
- **Documentation** (this README and the notebook's text): [CC BY 4.0](LICENSE-docs). Please attribute "Eric Brinker, Brinker Analytics" and link to the guide.
- **Citing:** see [CITATION.cff](CITATION.cff).

Written by Eric Brinker, [Brinker Analytics](https://brinkeranalytics.com).
