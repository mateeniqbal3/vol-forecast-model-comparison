# CHANGELOG.md

All notable changes to this project are documented here. Loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Planned
- Simple baselines (EWMA, GARCH(1,1))
- Complex ML model (gradient boosting / small neural network)
- Walk-forward validation harness (plus naive-split mode for comparison)
- Naive-vs-walk-forward, simple-vs-complex comparison and verdict

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
