# Volatility Forecasting: Was the Complexity Justified?

[![CI](https://github.com/mateeniqbal3/vol-forecast-model-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/mateeniqbal3/vol-forecast-model-comparison/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project compares a simple volatility model (EWMA and GARCH(1,1))
with a gradient-boosted ML model (LightGBM) for forecasting SPY's 5-day
realized variance. Every model is scored twice: under a naive random split
and under proper walk-forward validation. The naive numbers are shown next
to the real ones so you can see exactly how the naive scheme would have
misled.

**Short answer: no, the complexity was not justified.** Under walk-forward
validation LightGBM's QLIKE advantage over GARCH(1,1) is 0.012, which is
not statistically distinguishable from zero (Diebold–Mariano p = 0.47). On
RMSE it is worse (1.589e-3 vs. 1.443e-3), though not significantly
(p = 0.18). The naive random split made the same model look like a
decisive winner on both metrics (QLIKE p < 0.0001, MSE p = 0.016).

The cause is specific: a tree ensemble cannot forecast beyond its
training range. The model trained up to 2007 could not forecast more than
about 33% annualized volatility during the autumn of 2008, which realized
over 130%. The random split hid this by putting crisis weeks into
training.

---

## 1. Problem statement and hypothesis

Does a flexible ML model forecast short-horizon volatility better than a
GARCH-family model once it is validated properly? A second question sits
inside the first: would a naive validation scheme have given a different
answer?

**Why the ML model might win.** It sees information the baselines do not:
the daily high–low range, which estimates variance far less noisily than a
squared close-to-close return, plus the overnight gap and volume. It can
use signed returns to capture the leverage effect, which symmetric
GARCH(1,1) cannot. It can also learn nonlinear, regime-dependent dynamics.

**Why it might not.** The target is a noisy proxy for true variance (a sum
of five squared daily returns), which limits how much any model can learn.
The largest, most informative episodes are few. GARCH's persistence and
mean-reversion structure may already capture most of what is predictable
from daily data.

The project was set up to report whichever answer the evidence gave. Its
success does not depend on either model winning.

## 2. Methodology and bias controls

**Data and target.**
- **Data:** SPY daily OHLCV from `yfinance`, 1993-01-29 to 2025-12-31,
  split- and dividend-adjusted (ADR-001).
- **Target:** at the close of day *t*, the realized variance of the next
  five daily log returns, `RV_t = r²_{t+1} + … + r²_{t+5}` (ADR-002).

**Models.**

| Model | Specification | Estimated parameters |
|---|---|---|
| EWMA | RiskMetrics, λ = 0.94 (fixed) | 0 |
| GARCH(1,1) | Zero mean, Gaussian quasi-MLE via `arch` | 3 |
| LightGBM | Gamma objective (the same as QLIKE up to terms that do not depend on the forecast), 18 features from SPY's own OHLCV, 12-point hyperparameter grid chosen inside each training set on a purged, time-ordered inner split | 36–412 trees per fold, plus the grid choice |

Details are in ADR-007 (baselines) and ADR-004 (ML model).

**Metrics.** Both are on the variance scale, where they rank forecasts
consistently against a noisy proxy (Patton, 2011):
- **QLIKE** `log F + RV/F`, the primary metric;
- **RMSE**, the secondary metric.

Model differences are tested with Diebold–Mariano, using Newey–West
standard errors (ADR-005).

**Bias controls.**

- **Look-ahead bias.** No feature or baseline forecast may use data after
  day *t*. This is tested, not just intended: a perturbation harness
  ([`tests/lookahead.py`](tests/lookahead.py)) scrambles every input after
  a cut-off and checks that nothing dated at or before the cut-off
  changes. Every feature builder and both baseline forecasts run through
  it. The target itself is included as a positive control: the harness
  must flag it, or it proves nothing.
- **Validation leakage.** Targets on consecutive days share four of their
  five returns.
  - Walk-forward uses an expanding window, one fold per calendar year from
    2000 to 2025 (26 folds, 6,534 forecast dates).
  - Each fold's training stops 5 rows before the first test date, so every
    training target is fully observed at that date.
  - Each model receives only the data up to that date, and tests enforce
    all of this (ADR-003, ADR-008).
  - The LightGBM tuning split inside each training set is purged the same
    way.
- **Survivorship bias.** There is a single instrument whose traded price
  already reflects its index-membership changes, so there is no
  survivorship bias in the return series. The choice of SPY is itself
  disclosed as a limit on generality (ADR-001).
- **Multiple-testing and selection bias.**
  - Exactly **one** ML configuration was designed, run and scored. Its
    design was written into ADR-004 before any fit on real data, and the
    configuration log records that nothing was changed after results were
    seen.
  - The only tuning is the automated grid inside each training set.
  - Many DM tests are reported, but a single pre-registered test decides
    the verdict: walk-forward QLIKE, LightGBM vs GARCH(1,1).
- **Order of work.** Each design was fixed before the results that could
  have influenced it. The git history dates each record.

  | Fixed | Before any |
  |---|---|
  | Baseline specs, naive-split protocol, baseline naive scores (Phase 2, ADR-007/008) | ML model existed |
  | ML model design (Phase 3, ADR-004) | ML score, and any walk-forward score for any model |
  | Walk-forward design (Phase 4, ADR-003) | walk-forward score |
  | DM test and decision rule (Phase 5, ADR-005) | p-value (the Phase 4 point estimates had been seen) |

- **Fairness in both directions.** The ML model was given more information
  than the baselines (range, open, volume) and a loss that matches the
  evaluation metric. It was not weakened to make the simple model win.

## 3. Results: naive vs. walk-forward, side by side

Full tables, per-year results and diagnostics are in
[`docs/results_comparison.md`](docs/results_comparison.md). All numbers are
in [`docs/phase5_comparison.json`](docs/phase5_comparison.json).

The two middle columns score the **same 1,321 dates** (the naive test
dates from 2000 on), so they differ only in the validation scheme. Lower is
better.

| Model | Naive split, all 1,606 dates | Naive split, same dates | Walk-forward, same dates | Walk-forward, all 6,534 dates |
|---|---|---|---|---|
| **QLIKE** | | | | |
| EWMA | −6.6257 | −6.5997 | −6.5997 | −6.5887 |
| GARCH(1,1) | −6.6671 | −6.6438 | −6.6356 | −6.6327 |
| LightGBM | −6.7413 | −6.7240 | −6.6586 | −6.6451 |
| **RMSE** (5-day variance) | | | | |
| EWMA | 1.490e-03 | 1.607e-03 | 1.607e-03 | 1.508e-03 |
| GARCH(1,1) | 1.403e-03 | 1.508e-03 | 1.545e-03 | 1.443e-03 |
| LightGBM | 1.328e-03 | 1.426e-03 | 1.680e-03 | 1.589e-03 |

**LightGBM minus GARCH(1,1)** (negative favors LightGBM):

| Scheme | Metric | Mean difference | Diebold–Mariano p |
|---|---|---|---|
| Naive split | QLIKE | −0.074 | < 0.0001 |
| Naive split | MSE | −2.1e-07 | 0.016 |
| **Walk-forward** | **QLIKE** | **−0.012** (95% CI −0.046 to +0.022) | **0.47** (0.65 at lag 63) |
| Walk-forward | MSE | +4.5e-07 | 0.18 |

### What the naive split got wrong

- **It turned an undetectable edge into a decisive one.** Under the naive
  split LightGBM beats GARCH on both metrics with p ≤ 0.016. Under
  walk-forward neither difference is significant, and the RMSE ranking
  reverses.
- **It flattered each model in proportion to how much it learns from the
  data.** On identical dates, naive-split QLIKE minus walk-forward QLIKE
  is 0 for EWMA (nothing is estimated, so the forecasts are identical),
  −0.008 for GARCH (p = 0.03) and −0.065 for LightGBM (p = 0.004).
- **Crisis weeks leaked into training.** 2008 alone accounts for −0.027 of
  LightGBM's −0.065. The rest (−0.038 outside 2008 and 2020, against
  −0.007 for GARCH) fits a second path: the model learns from neighboring
  training dates whose targets overlap each test target. The two paths
  were not measured separately.

### The mechanism: tree forecasts have a ceiling

- **Forecasts are capped.** A boosted-tree forecast is a sum of leaf
  values learned from training data. Past the most extreme training
  inputs it stops rising.
- **The 2008 fold shows it.** That fold was trained on 1994–2007. In
  autumn 2008 its 22-day realized-variance input reached 4.2 times its
  training maximum. On the worst week (realized 5-day variance 0.0352,
  about 133% annualized) it forecast 0.00216. Multiplying every variance
  input by 100 leaves that forecast unchanged.
- **The cap sits below its own training data.** Its largest forecast
  anywhere in the sample is 0.00222, about 33% annualized.
- **GARCH has no such cap.** Its forecast is linear in the latest squared
  return, so it rose to 0.0138 (about 83% annualized) in the same weeks.
- **2008 decides the comparison.** That year alone adds +0.027 to the
  mean QLIKE difference, more than twice LightGBM's net advantage of
  0.012.

### Why QLIKE and RMSE disagree

The two losses weight different weeks.

- **RMSE is decided by crisis weeks.** The top 1% of weeks (66 dates)
  carry 83% of LightGBM's squared error, against 68% for GARCH. LightGBM's
  average forecast is 0.79 times average realized variance; GARCH's is
  1.00. The shortfall is almost all in those weeks.
- **QLIKE scores relative errors, so every week counts about equally.**
  LightGBM, which was trained on this loss, is slightly better in ordinary
  weeks. It has the lower QLIKE in 21 of 26 years, and its losses are
  concentrated in 2008 and 2020.

## 4. What didn't work, and limitations

- **The ML model's flexibility did not survive a regime it had not
  seen.** That is the finding, not a footnote. The model gains a little in
  ordinary conditions and gives it back when volatility matters most.
- **Information vs. flexibility.** LightGBM had extra inputs (range,
  open, volume) and still showed no significant gain. No experiment
  separates the value of the inputs from the value of the model.
- **One design, one protocol, by choice.** There was one ML configuration
  and one walk-forward scheme (expanding window, annual refit). Designs
  that might remove the ceiling were not tried after seeing the results,
  because that would be post-hoc redesign. Examples are linear-leaf trees
  and modeling RV relative to a GARCH forecast. They are hypotheses, not
  findings.
- **Limited statistical power.** Daily data gives a noisy variance proxy.
  The confidence interval (−0.046 to +0.022 QLIKE) cannot rule out a
  moderate LightGBM advantage. It can only rule out a large one. An
  intraday realized-variance target would sharpen the comparison.
- **Two episodes dominate.** The net result rests heavily on 2008 and
  2020. The figures above that exclude those years are descriptive,
  chosen after seeing the results, and are not evidence about calm
  markets.
- **Scope.** One instrument (SPY), daily data, a 5-day horizon, and
  2000–2025 out of sample. The simple side has no leverage effect:
  GJR-GARCH was deliberately not used, to avoid choosing the baseline from
  full-sample EDA (ADR-007).
- **Free, unofficial data.** `yfinance` adjusted history can be revised by
  the provider. The dataset manifest records checksums so that drift is
  detectable.

## 5. Verdict

**The added complexity was not justified.** The decision rule was fixed
before testing (ADR-005). Under walk-forward validation:

- **QLIKE:** LightGBM −6.6451 vs. GARCH(1,1) −6.6327. The 0.012 difference
  is not significant (p = 0.47; 0.65 at lag 63).
- **RMSE:** LightGBM is worse, 1.589e-3 vs. 1.443e-3, though not
  significantly (p = 0.18).

Under the naive random split, the same model appeared to win decisively on
both metrics. It did not.

The one step up in complexity that did pay off was EWMA to GARCH(1,1).
Three estimated parameters bought a QLIKE improvement of 0.044
(p ≤ 0.0002 at both lags).

For daily volatility forecasting of a broad equity index, GARCH's
structure (variance linear in the latest shock, mean-reverting)
extrapolates into crises by construction. A tree ensemble's flexibility
cannot.

---

## Reproducing

Requires Python 3.11+.

```bash
git clone https://github.com/mateeniqbal3/vol-forecast-model-comparison.git
cd vol-forecast-model-comparison
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_pipeline.sh   # about 3 minutes
```

The pipeline runs these steps in order. On Windows, run the same commands
directly:

```bash
python -m src.data                        # download (cached in data/raw/), clean, manifest
python -m src.evaluate naive-baselines    # Phase 2 record; must reproduce exactly
python -m src.evaluate validate           # all models, naive split and walk-forward
python -m src.evaluate compare            # tables, Diebold–Mariano tests, diagnostics
```

- **Downloads:** the price download is cached, so later runs use
  identical data. Pass `--refresh` to `python -m src.data` to download
  again.
- **Checks:** `validate` fails if the baselines' naive scores no longer
  match the Phase 2 record.
- **Notebook:** the EDA notebook ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb))
  also needs Jupyter.

## Testing

```bash
pytest tests/ -v    # 92 tests, synthetic data only, no download needed
ruff check src/ tests/
```

The tests cover:
- the target's look-ahead checks and the perturbation harness;
- agreement of the GARCH filter and forecast with `arch`;
- the walk-forward split (no test date precedes or overlaps its training
  window, unobserved targets are purged, and each fold's data ends at its
  first test date);
- the Newey–West variance against `statsmodels`.

## Repository layout

```
src/
  data.py             download, clean, log returns, dataset manifest
  target.py           5-day forward realized variance
  baseline_ewma.py    EWMA (λ = 0.94)
  baseline_garch.py   GARCH(1,1): estimation via arch, causal filter, 5-day forecast
  complex_model.py    LightGBM: features, purged inner tuning, fit/predict
  validation.py       naive random split, walk-forward folds, information-set rule
  evaluate.py         QLIKE/RMSE, Diebold–Mariano, run and compare commands
tests/                one test module per source module, plus the look-ahead harness
docs/                 results_comparison.md and the recorded JSON outputs
notebooks/01_eda.ipynb
```

## Documentation

- [`docs/results_comparison.md`](docs/results_comparison.md): full results
  and discussion
- [`DECISIONS.md`](DECISIONS.md): every design decision, the configuration
  log and the verdict
- [`ARCHITECTURE.md`](ARCHITECTURE.md): data, model and validation flow
- [`PROJECT.md`](PROJECT.md): the original specification
- [`CHANGELOG.md`](CHANGELOG.md): version history

## Tech stack

Python · pandas · NumPy · `arch` (GARCH) · LightGBM (with its
scikit-learn API) · statsmodels · `yfinance` · matplotlib · pytest · ruff.
No Docker, no deployment: this is a focused research project by design
(`PROJECT.md` §11).

## License

MIT. See [`LICENSE`](LICENSE).
