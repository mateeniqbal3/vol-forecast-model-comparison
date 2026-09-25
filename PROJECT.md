# PROJECT.md — Volatility Forecasting with Proper Time-Series Validation

> **Status:** Complete (v1.0.0). Verdict and results in `README.md` and
> `docs/results_comparison.md`.
> **Owner:** mateeniqbal3
> **Track:** Quant Finance (standalone — not tied to any academic
> semester/summer phase, not part of the RTP/Fulbright thesis narrative,
> and not dependent on `lob-alpha-engine`, `stat-arb-optimizer`, or
> `pairs-trading-honest-costs` — this is its own independent,
> self-contained project, though it shares that project's "small,
> foundational, honesty-first" spirit)
> **Repo:** `vol-forecast-model-comparison`
> **License:** MIT
> **Purpose:** an intentionally focused project demonstrating a specific,
> valuable form of judgment: knowing when a complex model is NOT
> justified. A large share of real "quant ML" work ends in exactly this
> conclusion — the simple model wins out-of-sample — and demonstrating
> the discipline to reach and report that conclusion honestly, rather than
> defaulting to "more complex must be better," is the actual deliverable.

This is the canonical specification. If ambiguous, make the most
reasonable engineering decision, record it in `DECISIONS.md`, and
continue.

---

## 1. Purpose

Forecast short-horizon realized volatility for a financial instrument
using a simple baseline (EWMA) and/or a GARCH-family model, compare it
against a more complex ML model on identical, properly time-series-
validated data, and report honestly which one actually wins
out-of-sample — and by how much. **The project's success is not tied to
either model winning.** Its success is tied to whether the comparison
itself is methodologically sound and honestly reported.

## 2. The judgment this project is built to demonstrate

This project exists to let the author credibly say, in an interview:

> "I compared a simple volatility model against a more complex ML model
> under proper walk-forward validation, and [whichever result actually
> occurred] — here's the evidence, and here's what it tells you about
> when added complexity is or isn't worth it."

For this claim to be true, not just stated, the project must:
- **Build and evaluate the simple baseline first, honestly, before
  building the complex model** — the baseline's real out-of-sample
  performance must be recorded before the complex model exists, so there
  is no way (even unconsciously) to tune the complex model specifically
  to beat a moving target
- **Give the complex model a genuinely fair chance** — reasonable
  features, reasonable hyperparameter search, reasonable training data —
  not a deliberately weakened complex model built to guarantee the simple
  one wins. An artificially hobbled "complex" model would be just as
  dishonest as an artificially inflated one.
- **Evaluate both models under identical, walk-forward (never random
  k-fold) validation** — see §8. This is the methodological core of the
  project and the single most common way volatility/time-series model
  comparisons go wrong.
- **Report whichever result actually occurs, with a specific mechanism,
  not a vague gesture.** If the simple model wins, explain concretely why
  (e.g. GARCH's built-in mean-reversion structure matches the data's
  actual behavior better than the ML model's flexibility can exploit
  given the available sample size). If the complex model wins, explain
  what specifically it captured that the simple model couldn't. Either
  outcome is a fine, reportable, valuable finding.

## 3. Problem Statement

Given historical price data for a financial instrument, forecast
short-horizon (e.g. next-day or next-few-day) realized volatility using
(a) a simple baseline — EWMA and/or a GARCH-family model — and (b) a more
complex ML model (e.g. gradient boosting or a small neural network using
engineered features), evaluate both under walk-forward time-series
validation, and determine — honestly — whether the added complexity of
the ML model is actually justified by its out-of-sample forecasting
performance.

## 4. Success Criteria (Definition of Done)

- [x] Historical price data acquired (free/public source — see §5)
- [x] Realized volatility target constructed (e.g. from squared/absolute
      returns over a forward window — the forecasting target, computed
      without look-ahead into the training features)
- [x] Simple baseline implemented: EWMA volatility estimate and/or a
      GARCH-family model (e.g. GARCH(1,1) via the `arch` package)
- [x] Complex model implemented: a gradient-boosting or small neural-
      network model using engineered features (e.g. lagged realized
      volatility, returns, volume, other available signals)
- [x] **Walk-forward validation** implemented and used for both models —
      never a random train/test split or k-fold cross-validation on this
      time-series data (see §8)
- [x] Both models compared on identical out-of-sample windows, using the
      same forecasting metric(s) (e.g. QLIKE loss and/or RMSE against
      realized volatility — see §7)
- [x] **Side-by-side results**: naive/random-split validation numbers vs.
      proper walk-forward validation numbers, for both models — showing
      whether (and how much) a naive validation approach would have
      overstated either model's real performance (see §9 — this replaces
      the "with/without transaction costs" comparison from this
      standalone project's brief template, adapted to fit a forecasting
      task rather than a trading strategy; see `DECISIONS.md` for this
      adaptation rationale)
- [x] Explicit bias-control section: look-ahead bias, survivorship bias
      (if applicable to the instrument/universe choice), and multiple-
      testing bias (if multiple feature sets or hyperparameter
      configurations were tried) — each named and addressed concretely
- [x] Honest verdict stated plainly: did the added complexity help, and
      by how much, given the data actually available?
- [x] Honest "what didn't work / limitations" section
- [x] README structured per §10
- [x] CI pipeline (lint + test) — no Docker/deployment required, same
      minimal-footprint discipline as `pairs-trading-honest-costs`

## 5. Dataset — Free/Public Sources Only

**Budget constraint: $0.** Use `yfinance` for historical daily price data
for a single liquid instrument (a major index or large-cap equity is a
reasonable, defensible choice — document why). Daily-frequency data is
expected and sufficient for this project's scope; do not seek intraday
data given the free-data constraint. Document the exact instrument, date
range, and rationale in `data/dataset_manifest.json` and `DECISIONS.md`.

## 6. Tech Stack

Python · pandas · NumPy · statsmodels and/or `arch` (for GARCH) ·
scikit-learn or a lightweight gradient-boosting library (for the complex
model) · matplotlib

Deliberately minimal, same discipline as `pairs-trading-honest-costs` —
this project does not need a deep learning framework or heavy
infrastructure; a well-executed, honestly validated comparison is the
entire point.

## 7. Forecasting Target & Evaluation Metric

- **Target**: realized volatility over a short forward window (e.g. next
  1-5 trading days), constructed from actual subsequent returns — this
  target must never leak into the features used to predict it
- **Evaluation metric**: QLIKE loss (a standard, asymmetric-penalty loss
  function for volatility forecasts, preferred in the volatility
  forecasting literature over plain RMSE because it penalizes
  under-prediction of volatility more appropriately) and/or RMSE — use at
  least one metric standard in the volatility forecasting literature, not
  an ad-hoc one, and justify the choice in `DECISIONS.md`

## 8. Validation Methodology — Walk-Forward (mandatory, never random)

Time-series volatility data is serially correlated and regime-dependent —
a random train/test split or k-fold cross-validation will leak
information via temporal adjacency and produce an optimistic, unrealistic
estimate of real-world forecasting performance. Use **walk-forward
validation**: the model is trained on data up to time T, evaluated on a
subsequent out-of-sample window, then the training window rolls forward
and the process repeats. Document the exact window sizes (training window
length, forecast horizon, step size) in `DECISIONS.md`.

## 9. The Naive-vs-Proper Validation Comparison (this project's central mechanism)

To make the value of walk-forward validation concrete and visible (the
same spirit as `pairs-trading-honest-costs`'s gross-vs-net comparison,
adapted to this project's actual axis of honesty), this project must show:

| | Naive (random split) validation | Walk-forward validation |
|---|---|---|
| Simple model (EWMA/GARCH) performance | | |
| Complex model (ML) performance | | |

Report this side by side and discuss explicitly whether the naive
approach would have overstated either model's real-world performance,
and — critically — whether it would have changed which model appears to
"win." A common and important finding in real quant work is that a
complex model's apparent edge over a simple baseline shrinks or reverses
once proper validation is used; report whichever pattern is actually
found.

## 10. Required README Structure

1. Clear problem statement and the mechanism/hypothesis being tested (why
   might a more complex model outperform a simple volatility model here —
   or why might it not?)
2. Methodology section explicitly naming the bias controls (look-ahead,
   survivorship if applicable, multiple-testing if applicable)
3. Results shown side by side: naive-validation vs. walk-forward-
   validation performance for both models (§9) — this is this project's
   adapted equivalent of a gross-vs-net-of-costs comparison
4. An honest "what didn't work / limitations" section, framed as a
   positive signal of rigor, not hedging
5. A plainly stated verdict: was the added complexity justified?

## 11. Repository Structure

```
vol-forecast-model-comparison/
├── PROJECT.md
├── TASKS.md
├── IMPLEMENTATION_PLAN.md
├── ARCHITECTURE.md
├── DECISIONS.md
├── CHANGELOG.md
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── data.py                  # price data loading (yfinance)
│   ├── target.py                  # realized volatility target construction
│   ├── baseline_ewma.py             # EWMA volatility baseline
│   ├── baseline_garch.py             # GARCH-family model (arch package)
│   ├── complex_model.py               # ML model (gradient boosting / small NN) with engineered features
│   ├── validation.py                   # walk-forward validation harness (mandatory) + naive-split comparison mode
│   └── evaluate.py                      # QLIKE/RMSE metrics, comparison table generation
├── tests/
│   ├── test_data.py
│   ├── test_target.py               # includes a look-ahead-bias check on the target construction
│   ├── test_baseline_ewma.py
│   ├── test_baseline_garch.py
│   ├── test_complex_model.py
│   └── test_validation.py             # specifically tests the walk-forward split logic
├── notebooks/
│   └── 01_eda.ipynb
├── data/
│   ├── dataset_manifest.json
│   ├── raw/                    # gitignored
│   └── processed/                # gitignored
├── docs/
│   └── results_comparison.md    # the naive-vs-walk-forward, simple-vs-complex table (section 9)
├── paper/
│   └── writeup.md                  # optional short writeup, see TASKS.md
└── scripts/
    └── run_pipeline.sh
```

No `Dockerfile` — same reasoning as `pairs-trading-honest-costs`: this is
a focused research-script project, and containerizing it would be scope
inflation working against its own stated purpose of focus and simplicity.

## 12. Constraints & Guardrails

- $0 data constraint — yfinance only
- No Docker, no API, no deployment
- Walk-forward validation is mandatory for both models — no random splits
  or k-fold cross-validation anywhere in the reported results
- The simple baseline must be built and its out-of-sample performance
  recorded before the complex model is built and tuned — not the reverse
- Report the actual verdict, whichever way it goes — do not adjust
  features, hyperparameters, or the complex model's design after seeing
  an unflattering comparison in order to manufacture a "complexity wins"
  result (or, equally, do not handicap the complex model to guarantee the
  simple one wins — both directions of manipulation are prohibited)
- If multiple feature sets or hyperparameter configurations were tried for
  the complex model, disclose how many, same discipline as
  `stat-arb-optimizer` and `pairs-trading-honest-costs`

## 13. Definition of Done

See `TASKS.md`. At the project level: all boxes in §4 checked, both models
evaluated under genuine walk-forward validation, the naive-vs-proper
validation comparison (§9) reported honestly, a plain verdict stated on
whether complexity was justified, README structured per §10, CI green.
