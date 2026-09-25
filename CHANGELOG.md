# CHANGELOG.md

All notable changes to this project are documented here. Loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

---

## [1.0.0] — 2026-09-25 — Comparison complete

First complete release. EWMA, GARCH(1,1) and LightGBM volatility forecasts
for SPY, each scored under a naive random split and under walk-forward
validation. Verdict: the ML model's added complexity was **not
justified**. Its walk-forward QLIKE advantage over GARCH(1,1) is 0.012
(Diebold–Mariano p = 0.47), its RMSE is worse, and the naive split had
made it look like a decisive winner. See `README.md` and
`docs/results_comparison.md`.

### Changed
- `README.md`: rewritten around the results, following `PROJECT.md` §10.
  It covers the problem and hypothesis, bias controls with the order in
  which each design was fixed, the naive-vs-walk-forward table, the
  tree-ceiling mechanism, limitations and the verdict. Every comparative
  claim cites its numbers.
- `DECISIONS.md`: final review. Consequences that earlier ADRs deferred
  to later phases are resolved with the measured numbers. ADR-005's
  description of the neighboring-target leak is aligned with the results
  document (consistent with the data, not measured separately). The
  unused ADR template was removed.
- `ARCHITECTURE.md`: the complex model is described as built (LightGBM),
  `evaluate.py`'s responsibilities are corrected, and the recorded outputs
  are listed.
- `PROJECT.md`: status set to complete; success criteria checked.
- `paper/writeup.md`: points to the README and the results document
  instead of duplicating them.
- `pyproject.toml`: version 1.0.0.

---

## [0.7.0] — 2026-09-25 — CI

### Changed
- `.github/workflows/ci.yml`: workflow token restricted to read-only
  repository contents
- `requirements.txt`: `ruff==0.16.9` pinned, so lint results do not
  change with new ruff releases

### Verified
- The CI steps (`ruff check src/ tests/`, `pytest tests/ -v`) pass on a
  fresh Python 3.11.13 install of `requirements.txt`: lint clean, 92 tests
  passed. The tests use synthetic data only, so CI needs no download.

---

## [0.6.0] — 2026-09-25 — Comparison and verdict

### Added
- `src/evaluate.py`: Diebold–Mariano test with Newey–West (Bartlett) HAC
  errors and 95% intervals; `compare` command writing
  `docs/phase5_comparison.json`, with the naive-vs-walk-forward table on
  all and identical dates, DM tests for every model pair, metric and lag,
  per-year results, the extrapolation and error-concentration
  diagnostics, and the LightGBM ceiling probe
- `docs/results_comparison.md`: comparison table, why the naive split
  misled, why QLIKE and RMSE disagree, the verdict and limitations
- Tests for the Newey–West variance (against `statsmodels`), the DM
  statistic, p-value and interval, and the yearly decomposition
- ADR-005: pre-registered test design and the verdict (not justified);
  ADR-004 configuration log closed at one configuration

---

## [0.5.0] — 2026-09-25 — Walk-forward validation

### Added
- `src/validation.py`: walk-forward folds (expanding window, one calendar
  year per test block from 2000, 5-row purge) and a runner that refits
  each fold and refuses any fold whose information set passes its origin
- `src/evaluate.py`: `validate` command, which scores EWMA, GARCH(1,1) and
  LightGBM under both the naive random split and walk-forward, checks that
  the Phase 2 baseline record reproduces exactly, and writes
  `docs/phase4_validation_results.json` plus per-date forecasts
  (`data/processed/forecasts.csv`, not tracked)
- Walk-forward tests: one fold per year, test blocks partition the test
  period, no test date precedes or overlaps its training window, purge of
  unobserved targets, information set ends at the fold origin, expanding
  window, leaking folds rejected
- ADR-003: walk-forward design, fixed before the first run

---

## [0.4.0] — 2026-09-25 — Complex model

### Added
- `src/complex_model.py`: LightGBM with the gamma objective (equivalent to
  QLIKE up to terms that do not depend on the forecast), 18 causal features
  from SPY's own OHLCV (HAR-style realized variance, signed returns,
  downside semivariance, Parkinson range, overnight gap, relative volume,
  drawdown), and hyperparameter selection inside `fit` via a purged
  chronological inner split over a fixed 12-point grid with early stopping
- Tests for feature values, burn-in, the inner split and its purge,
  determinism, input checks, and recovery of the true conditional variance
  on simulated GARCH data; the feature builder is added to the look-ahead
  check
- ADR-004: pre-registered design and configuration log (one configuration)

### Changed
- `tests/conftest.py`: synthetic price generator takes a length and seed
- `requirements.txt`: `lightgbm>=4.7` (`eval_X`/`eval_y` API)

---

## [0.3.0] — 2026-09-25 — Simple baselines and naive-split record

### Added
- `src/baseline_ewma.py`: RiskMetrics EWMA (λ = 0.94, not estimated)
- `src/baseline_garch.py`: zero-mean GARCH(1,1), Gaussian QML via `arch`,
  analytic 5-day forecast; estimation separated from causal filtering
- `src/validation.py`: model interface, shared information-set rule, and
  the naive random split (walk-forward follows in Phase 4)
- `src/evaluate.py`: QLIKE and MSE/RMSE on the variance scale;
  `naive-baselines` command
- `docs/phase2_naive_baselines.json`: naive-split scores for EWMA and
  GARCH(1,1), recorded before the complex model exists
- Tests for both baselines (including agreement with `arch`), the naive
  split and information set, and the losses; both baseline forecasts added
  to the look-ahead check
- ADR-007 (baseline specifications), ADR-008 (naive-split protocol and
  recorded numbers)

---

## [0.2.0] — 2026-09-25 — Data and volatility target

### Added
- `src/data.py`: SPY daily OHLCV via `yfinance` (1993-01-29 to
  2025-12-31, adjusted), cached raw download, cleaning, daily log returns
- `data/dataset_manifest.json`: instrument, date range, cleaning counts,
  log-return checksums for drift detection
- `src/target.py`: 5-day forward realized variance target
- `tests/lookahead.py`: perturbation harness for look-ahead bias
- `tests/test_data.py`, `tests/test_target.py`, including the target
  look-ahead checks and a positive control for the harness
- `notebooks/01_eda.ipynb`: return distribution, volatility clustering,
  target shape, leverage effect
- ADR-001 (instrument), ADR-002 (horizon, target, metric)

### Fixed
- `clean_prices` uses a stable sort, so "keep the last record" for
  duplicate dates is deterministic

---

## [0.1.0] — Scaffold

### Added
- Repository structure
- `PROJECT.md`, `TASKS.md`, `IMPLEMENTATION_PLAN.md`, `ARCHITECTURE.md`,
  `DECISIONS.md` — full project specification, including the "genuinely
  fair chance for both models" and "naive-vs-proper validation" honesty
  requirements, before any code was written
- MIT License

---

<!--
Template for future entries — append new entries above this comment.
Suggested version bumps: 0.2.0 after data+target, 0.3.0 after simple
baselines, 0.4.0 after complex model, 0.5.0 after walk-forward
validation, 1.0.0 at comparison + verdict + full documentation.

## [0.X.0] — YYYY-MM-DD — Short title

### Added
-

### Changed
-

### Fixed
-
-->
