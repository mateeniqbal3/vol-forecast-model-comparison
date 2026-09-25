# Volatility Forecasting: Was the Complexity Justified?

[![CI](https://github.com/mateeniqbal3/vol-forecast-model-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/mateeniqbal3/vol-forecast-model-comparison/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status: work in progress.** No model results have been generated yet —
> every TBD below is genuinely unknown, not a placeholder for an assumed
> outcome. See [`TASKS.md`](TASKS.md) for progress.

A simple volatility baseline (EWMA / GARCH) vs. a more complex ML model,
compared honestly under proper walk-forward validation — with the naive
(random-split) comparison shown alongside it, so you can see exactly how
much a sloppy validation scheme would have misled the conclusion.

---

## 1. Problem statement & hypothesis

Does a more complex ML model actually forecast short-horizon realized
volatility better than a simple GARCH/EWMA baseline — once you validate
properly? The hypothesis being tested isn't "complexity wins" or
"simplicity wins" — it's specifically whether a naive validation approach
would have given a misleading answer to that question. A large share of
real quant ML work ends with the simple model winning; this project is
built to reach and report that conclusion honestly if that's what the
evidence shows, rather than assume added complexity is automatically an
improvement.

## 2. Methodology — bias controls named explicitly

- **Look-ahead bias**: the volatility target at day *t* is built from
  strictly forward returns (after *t*) and never leaks into any feature —
  verified by an automated perturbation test (`tests/test_target.py`),
  not just design intention.
- **Survivorship bias**: a single, currently-listed liquid instrument is
  used (see [`DECISIONS.md`](DECISIONS.md) ADR-001) — this project makes
  no claim about a broader universe, so survivorship bias in the
  traditional cross-sectional sense doesn't directly apply, but the
  single-instrument scope is itself a disclosed limitation on
  generalizability.
- **Multiple-testing bias**: if more than one feature set or
  hyperparameter configuration is tried for the complex model, the exact
  count is disclosed in [`DECISIONS.md`](DECISIONS.md) ADR-004 — an
  undisclosed search would make the reported result impossible to
  properly evaluate.
- **Fairness in both directions**: the simple baseline's naive-split
  result is recorded and set aside *before* the complex model is built,
  so the complex model's design can't be reverse-engineered to beat a
  known target. Equally, the complex model is given a genuinely
  reasonable feature set and search — not deliberately weakened to
  guarantee the simple model wins.

## 3. Results — naive validation vs. walk-forward validation, side by side

<!-- TODO (Phase 5): embed docs/results_comparison.md content here once
     generated. This replaces a gross-vs-net-of-costs table (which
     doesn't apply to a forecasting task) with the equivalent honesty
     axis for this project: naive vs. proper validation. -->

| | Naive (random-split) validation | Walk-forward validation |
|---|---|---|
| Simple model (EWMA/GARCH) | TBD | TBD |
| Complex model (ML) | TBD | TBD |

**Verdict:** _TBD — see [`DECISIONS.md`](DECISIONS.md) ADR-005. Stated
plainly, whichever way the evidence points._

## 4. What didn't work / limitations

<!-- TODO (Phase 7): fill in honestly once the project is built. This
     section is a positive signal of rigor, not hedging. -->

- _TBD_
- Single instrument, daily frequency, free-data-only scope — see
  [`DECISIONS.md`](DECISIONS.md) ADR-001 for what this does and doesn't
  let this project's verdict generalize to.
- Daily data only, so the "realized variance" target is a sum of squared
  daily returns rather than an intraday realized measure — a noisier
  proxy (see [`DECISIONS.md`](DECISIONS.md) ADR-002).

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full data, model, and
validation-comparison flow.

## Project Documentation

- [`PROJECT.md`](PROJECT.md) — full specification
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design
- [`DECISIONS.md`](DECISIONS.md) — architecture decision records
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — build plan
- [`TASKS.md`](TASKS.md) — granular task checklist
- [`CHANGELOG.md`](CHANGELOG.md) — version history

## Running Locally

```bash
git clone https://github.com/mateeniqbal3/vol-forecast-model-comparison.git
cd vol-forecast-model-comparison
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

bash scripts/run_pipeline.sh
```

The price download is cached in `data/raw/` and reused on later runs; pass
`--refresh` to `python -m src.data` to re-download. Any drift is recorded
in `data/dataset_manifest.json`. The EDA notebook
(`notebooks/01_eda.ipynb`) additionally needs Jupyter
(`pip install jupyter`).

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

Python · pandas · NumPy · statsmodels / `arch` (GARCH) · scikit-learn or a
lightweight gradient-boosting library · matplotlib

No Docker, no deployment — this is a focused research-script project by
design; see [`PROJECT.md`](PROJECT.md) §11.

## License

MIT — see [`LICENSE`](LICENSE)
