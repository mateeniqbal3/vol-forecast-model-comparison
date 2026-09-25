# TASKS.md — vol-forecast-model-comparison

Ordered checklist. Work top to bottom, commit after each phase group.
Read `PROJECT.md` §2 before Phase 3 (complex model) — the simple
baseline (Phase 2) must be built and its results recorded first.

## Phase 0 — Setup

- [x] Init git repo, `.gitignore` (Python, data/raw, data/processed,
      .env, __pycache__)
- [x] `pyproject.toml`, `requirements.txt`
- [x] `LICENSE` (MIT, mateeniqbal3, current year)
- [x] Folder structure per `PROJECT.md` §11
- [x] Commit: "chore: initial project scaffold"

## Phase 1 — Data & Target

- [x] Pull historical daily price data via `yfinance` for one documented
      instrument (see `PROJECT.md` §5)
- [x] Write `src/data.py`: load, clean
- [x] Write `src/target.py`: construct the realized volatility forecasting
      target from actual forward returns — verify no leakage into features
- [x] `notebooks/01_eda.ipynb`: volatility clustering, basic distributional
      properties of returns
- [x] `tests/test_data.py`, `tests/test_target.py` (the latter must
      include a look-ahead check: confirm the target at time T uses only
      returns strictly after T, and confirm no feature construction
      elsewhere accidentally uses the same forward window)
- [x] Write `data/dataset_manifest.json`: instrument, date range, source
- [x] Commit: "feat: data pipeline and volatility target construction"

## Phase 2 — Simple Baseline (build and record BEFORE the complex model)

- [x] Write `src/baseline_ewma.py`: EWMA volatility estimate
- [x] Write `src/baseline_garch.py`: GARCH(1,1) (or a documented
      alternative order) via the `arch` package
- [x] Evaluate both under a NAIVE (random-split) validation first, purely
      to have that naive number on record for the Phase 5 comparison —
      then set it aside
- [x] `tests/test_baseline_ewma.py`, `tests/test_baseline_garch.py`
- [x] Commit: "feat: simple baselines (EWMA, GARCH) implemented and recorded"

## Phase 3 — Complex Model

- [x] Write `src/complex_model.py`: a gradient-boosting or small
      neural-network model using engineered features (lagged realized
      volatility, returns, volume, etc.)
- [x] Give it a genuinely fair chance — reasonable feature set,
      reasonable hyperparameter search — per `PROJECT.md` §2; do not
      deliberately weaken it
- [x] Add every feature builder to `FEATURE_BUILDERS` in
      `tests/test_target.py` so the look-ahead check covers it
- [x] `tests/test_complex_model.py`
- [x] Commit: "feat: complex ML model for volatility forecasting"

## Phase 4 — Walk-Forward Validation (mandatory)

- [x] Write `src/validation.py`: walk-forward validation harness (rolling
      or expanding training window, fixed forecast horizon, documented
      step size — see `DECISIONS.md`), PLUS a naive-random-split mode
      used only to reproduce the Phase 2 naive numbers for comparison
- [x] Purge training rows whose forward target window extends past the
      forecast origin (see `DECISIONS.md` ADR-002)
- [x] Re-evaluate BOTH the simple baseline(s) and the complex model under
      genuine walk-forward validation
- [x] `tests/test_validation.py`: specifically tests that the walk-forward
      split logic never lets a test-window timestamp precede or overlap
      its corresponding training window
- [ ] Commit: "feat: walk-forward validation applied to all models"

## Phase 5 — Comparison & Evaluation

- [ ] Write `src/evaluate.py`: QLIKE and/or RMSE computation, generates
      the naive-vs-walk-forward comparison table for both models (per
      `PROJECT.md` §9)
- [ ] Generate `docs/results_comparison.md` with the full table and a
      SPECIFIC mechanistic discussion of any differences found — not a
      vague "validation matters" statement
- [ ] State the plain verdict: was the complex model's added complexity
      justified out-of-sample, under proper validation?
- [ ] If multiple feature sets/hyperparameter configurations were tried
      for the complex model, log and disclose the count in `DECISIONS.md`
- [ ] Commit: "feat: full model comparison complete — see docs/results_comparison.md"

## Phase 6 — CI

- [ ] `.github/workflows/ci.yml`: lint + test on push/PR to `main`
- [ ] Commit: "ci: add lint + test workflow"

## Phase 7 — Documentation Polish

- [ ] Finalize `README.md` per the required structure in `PROJECT.md` §10:
      problem statement, methodology with named bias controls, the
      naive-vs-walk-forward comparison table, honest limitations, and a
      plainly stated verdict
- [ ] Fill `DECISIONS.md` with all real decisions (instrument choice,
      window sizes, metric choice, configuration-count disclosure, and
      critically the verdict itself, stated precisely)
- [ ] `CHANGELOG.md` v1.0.0 entry
- [ ] (Optional) `paper/writeup.md` if the README doesn't fully capture
      the story on its own
- [ ] Final review: does the README ever claim the complex model is
      "better" (or "not better") without pointing to the specific
      walk-forward numbers that support that claim?

## Definition of Done

All boxes checked, both models evaluated under genuine walk-forward
validation, naive-vs-proper comparison reported honestly, a plain verdict
stated, README structured per `PROJECT.md` §10, CI green.
