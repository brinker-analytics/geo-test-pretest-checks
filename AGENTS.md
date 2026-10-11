# AGENTS.md

Notes for coding agents (and people) working on this repository.

## What this project is

`geo-test-pretest-checks` runs three checks on a brand's own geo-level sales history before a geo experiment is launched:

1. Comparison strength: correlation (daily and weekly), MAPE/WAPE of the scaled control, and an out-of-sample dry run.
2. Required spend: an MDE range estimated from the data's own noise (90% confidence, 80% power, one-sided), a power score, and required spend = MDE × baseline treatment sales ÷ target iROAS.
3. Whether the business can act on the result (a decision checklist).

It ends in a GO / FIX / DON'T TEST verdict. It is a pre-launch gate, not a design or analysis library (see GeoLift, Trimmed Match, Meridian GeoX for those).

## Layout

- `geo_pretest/io.py`: GA4-style CSV loader (`load_ga4_csv`) and `to_wide`. Missing days stay NaN unless `fill_missing="zero"`.
- `geo_pretest/checks.py`: noise model, the three checks, power score, verdict.
- `notebooks/pretest_checks.ipynb`: worked example on synthetic data; committed with outputs.
- `scripts/make_synthetic_data.py`: seeded generator for `data/synthetic_ga4_sales_by_dma.csv` (synthetic, no client data).
- `tests/`: unit tests (`test_checks.py`) and the slow calibration test (`test_calibration.py`).

## Running

```bash
pip install -r requirements.txt
pytest -q                 # fast tests
pytest -q -m slow -s      # calibration tests (~1-2 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/pretest_checks.ipynb
```

CI (`.github/workflows/tests.yml`) runs all three on every push and pull request.

## Conventions

- Never return an MDE, iROAS, or test duration from generic inputs; compute them only from the data supplied. Values typed into the notebook are labeled EXAMPLE.
- Thresholds are "how I read it" guidance and stay as named, visible parameters in `checks.py`.
- Test lengths are whole weeks (multiples of 7).
- Changes to the noise model must keep `pytest -m slow` passing (7-13% false positives, 72-88% power at 28 and 84 days).
- Keep the committed sample reproducible: `generate()` with defaults must reproduce `data/synthetic_ga4_sales_by_dma.csv` exactly.
- Writing style: "an MMM", never "MMM" alone as a noun.
- Licenses: code MIT (`LICENSE`), documentation CC BY 4.0 (`LICENSE-docs`).

## Maintainer and questions

Maintained by Eric Brinker of Brinker Analytics (https://brinkeranalytics.com). Questions: eric@brinkeranalytics.com or https://brinkeranalytics.com/contact.
