# ARCHITECTURE.md — vol-forecast-model-comparison

## 1. System Overview

A single research pipeline: one price series feeds a volatility target
and two competing forecasting approaches (simple baseline, complex ML),
each evaluated twice — once under a naive validation scheme (for
comparison purposes only) and once under genuine walk-forward validation
(the real result). The naive-vs-proper contrast is itself a first-class
output, not a discarded intermediate step.

## 2. Data + Target Flow

```
yfinance (single documented instrument)
              |
              v
        src/data.py
   load, clean daily price series
              |
              v
        src/target.py
   construct realized volatility target
   from FORWARD returns (never used as
   a feature -- verified by a look-ahead
   test)
```

## 3. Model Flow

```
Cleaned price series + target
              |
   +----------+----------+
   v                     v
src/baseline_ewma.py   src/baseline_garch.py
EWMA volatility          GARCH(1,1) via `arch`
estimate                   package
   |                     |
   +----------+----------+
              v
   Simple baseline(s) --
   naive-split number
   recorded, then SET ASIDE
   (does not inform complex
    model design)
              |
              v
      src/complex_model.py
   LightGBM, gamma (QLIKE)
   objective, 18 engineered
   features (realized variance,
   returns, range, volume),
   tuned inside each training set
```

## 4. Validation + Comparison Flow (the project's core)

```
Simple baseline(s) + Complex model
              |
              v
        src/validation.py
   +----------+----------+
   v                     v
Naive (random-split)   Walk-forward
mode -- reproduces       (rolling/expanding
Phase 2's recorded        window, mandatory,
numbers for both           real result)
models
   |                     |
   +----------+----------+
              v
        src/evaluate.py
   QLIKE / RMSE for all four
   cells (2 models x 2 validation
   schemes)
              v
    docs/results_comparison.md
    naive-vs-walk-forward,
    simple-vs-complex, side by side
              v
    Plain verdict: was the added
    complexity justified
    out-of-sample?
```

## 5. Component Responsibilities

| Component | Responsibility | Does NOT do |
|---|---|---|
| `src/data.py` | Load, clean price data | Target/feature construction |
| `src/target.py` | Build the forward-looking volatility target | Feature construction for models |
| `src/baseline_ewma.py`, `src/baseline_garch.py` | Simple volatility forecasts | ML modeling |
| `src/complex_model.py` | ML-based volatility forecast | Baseline modeling |
| `src/validation.py` | Both naive and walk-forward validation harnesses | Metric computation |
| `src/evaluate.py` | QLIKE/RMSE, Diebold–Mariano tests, comparison records; drives runs through the validation harness | Model definitions (its only direct fit is the Phase 5 ceiling-probe refit of one fold) |

### Recorded outputs

Each is deterministic and produced by one command (see `scripts/run_pipeline.sh`).

| File | Command | Contents |
|---|---|---|
| `data/dataset_manifest.json` | `python -m src.data` | Instrument, date range, cleaning counts, drift checksums |
| `docs/phase2_naive_baselines.json` | `python -m src.evaluate naive-baselines` | Baseline naive-split scores, recorded before the complex model existed |
| `docs/phase4_validation_results.json` | `python -m src.evaluate validate` | All models under both schemes, with per-fold fits |
| `docs/phase5_comparison.json` | `python -m src.evaluate compare` | Comparison table, DM tests, diagnostics |

## 6. Key Architectural Decisions (summary — full rationale in DECISIONS.md)

- **Naive-split validation retained as a comparison artifact, not
  discarded**: this is what makes the value of walk-forward validation
  visible and concrete, rather than asserted — see `PROJECT.md` §9.
- **Simple baseline built and recorded before the complex model exists**:
  prevents the complex model's design from being (even unconsciously)
  reverse-engineered to beat a known target — see `PROJECT.md` §2 and
  `IMPLEMENTATION_PLAN.md`'s dependency notes.
- **QLIKE as the primary evaluation metric** (alongside RMSE): standard in
  the volatility forecasting literature specifically because it penalizes
  under-prediction of volatility asymmetrically, which plain RMSE does
  not — see `PROJECT.md` §7.
- **Single instrument, daily frequency**: consistent with the $0 data
  constraint and this project's deliberately small scope.
