# DECISIONS.md — Architecture Decision Records

Format: ADR number, status, Context / Decision / Consequences. Append real
ADRs as decisions are made.

Status values: `Proposed`, `Accepted`, `Superseded by ADR-00X`, `Rejected`.

---

## ADR-001: Instrument selection

**Status:** Accepted (Phase 1)

**Context:** `PROJECT.md` §5 requires a single, documented, liquid
instrument (a major index or large-cap equity is a reasonable default).

**Decision:** SPY (SPDR S&P 500 ETF Trust), daily, from its first trading
day (1993-01-29) through 2025-12-31 inclusive, via `yfinance` with
`auto_adjust=True` (OHLC adjusted for splits and dividends). The end date
is fixed in `src/data.py` so the sample cannot drift between runs, and
the raw download is cached so repeated runs use identical data. Actual
coverage, row counts, cleaning counts and drift checksums are recorded in
`data/dataset_manifest.json`.

Why SPY rather than the alternatives considered:
- vs. `^GSPC` (the S&P 500 index): SPY is a tradable instrument with a
  meaningful traded-volume series (a candidate feature for the complex
  model), and adjusted prices include dividends. `^GSPC` has a longer
  history, but its volume field is not a trading volume of the instrument.
- vs. a single large-cap stock: index-level volatility is less dominated
  by idiosyncratic events (earnings, M&A), and its stylized facts
  (clustering, leverage effect) are well documented, so any verdict is
  about the models rather than one company's news history.
- ~33 years of daily data spans several distinct volatility regimes
  (1990s, 2000–02, 2008–09, 2020, 2022), which matters for walk-forward
  validation.

Returns are close-to-close log returns of adjusted close. Adjusting
rescales all earlier prices by a constant factor at each dividend, so log
returns are unaffected except on ex-dividend days, where the adjustment
correctly removes the mechanical price drop.

**Consequences:**
- The verdict applies to one broad US equity index ETF at daily frequency.
  It says nothing directly about single stocks, other asset classes,
  intraday horizons, or instruments with different volatility dynamics.
- Survivorship: SPY still exists, and choosing it in 2026 is a choice made
  with hindsight. But the question is about forecasting its volatility,
  not its returns, and the index's own constituent changes are reflected
  in its actual traded price at each date, so there is no survivorship
  bias in the return series itself.
- yfinance is a free, unofficial source. Adjusted history can be revised
  by the provider; `log_return_checks` in the manifest allows a
  re-download to be checked against the recorded sample.
- EDA (`notebooks/01_eda.ipynb`) was run on the full sample, including
  periods that will later be out-of-sample. It examines only
  well-established stylized facts (fat tails, volatility clustering,
  persistence) and computes no model performance numbers; no model
  parameter is tuned from it. This is disclosed rather than hidden.

---

## ADR-002: Forecasting horizon, target, and evaluation metric

**Status:** Accepted (Phase 1). Metrics implemented in Phase 2 (`src/evaluate.py`).

**Context:** `PROJECT.md` §7 specifies a short forward window (e.g. 1-5
trading days) and QLIKE and/or RMSE as the evaluation metric.

**Decision:**
- **Horizon:** h = 5 trading days (one trading week).
- **Target:** forward realized variance
  `RV_t = r_{t+1}^2 + ... + r_{t+5}^2`, where `r` is the daily log return
  and the forecast is made at the close of day t (`src/target.py`). The
  target is in variance units; volatility forecasts from every model are
  converted to 5-day variance before scoring.
- **Primary metric** (`src/evaluate.py`): QLIKE in the form `L(RV, F) = log(F) + RV / F`,
  where F is the forecast 5-day variance.
- **Secondary metric:** MSE of the variance forecast (reported as RMSE).

Why:
- With only daily data, true realized variance (from intraday returns) is
  unavailable; a single squared daily return is an extremely noisy proxy
  for one day's variance. Summing five of them reduces that noise while
  keeping the horizon short.
- Patton (2011, *J. Econometrics*, "Volatility forecast comparison using
  imperfect volatility proxies") shows that MSE and QLIKE on the
  **variance** scale are among the losses that rank forecasts consistently
  when the target is a noisy but conditionally unbiased proxy such as
  squared returns. RMSE on the volatility (square-root) scale is not
  robust in this sense, which is why the target and both metrics are in
  variance units.
- QLIKE is primary because it scores forecast errors in relative terms
  and penalizes under-prediction more heavily, and because it is less
  dominated by a handful of extreme-volatility weeks than MSE.
- The `log(F) + RV/F` form is used rather than
  `RV/F - log(RV/F) - 1` because it stays defined when RV = 0 (possible if
  a week has five unchanged closes) and ranks forecasts identically.

**Consequences:**
- The noisy proxy lowers the power to tell models apart: the same
  forecasts would be distinguished more sharply against an intraday
  realized measure. Differences between models must be judged with a
  proper test (e.g. Diebold–Mariano), not raw loss gaps alone.
- Targets on consecutive days share 4 of their 5 returns, so losses are
  serially correlated; any significance test in Phase 5 needs
  autocorrelation-robust (HAC) standard errors.
- The target at t is only fully observed at the close of t+5. At forecast
  origin T, training may use target rows t <= T - 5 only; the Phase 4
  walk-forward harness must purge those 5 rows. Without the purge, the
  last training labels would overlap the test window, which is exactly
  the leakage walk-forward validation exists to prevent.

---

## ADR-003: Walk-forward window design

**Status:** Proposed — resolves at the end of Phase 4

**Context:** `PROJECT.md` §8 requires documented training window length,
forecast horizon, and step size for the walk-forward harness.

**Decision:** _Fill in: exact window sizes and stepping scheme (rolling
vs. expanding window)._

**Consequences:** _Fill in — does the chosen window size trade off
adaptability to regime change against training data sufficiency? Note
this if relevant._

---

## ADR-004: Complex model feature set and configuration count

**Status:** Proposed — resolves at the end of Phase 3/5

**Context:** `PROJECT.md` §2 requires the complex model be given a
genuinely fair chance, and `PROJECT.md` §12 requires disclosing how many
feature sets/hyperparameter configurations were tried.

**Decision:** _Fill in: final feature set, model type and
hyperparameters, and the actual count of configurations evaluated before
arriving at this one._

**Consequences:** _Fill in — does the configuration count suggest any
risk that the reported complex-model result is itself optimistic due to
selection, even under walk-forward validation?_

---

## ADR-005: The verdict — was the added complexity justified?

**Status:** Proposed — resolves at the end of Phase 5

**Context:** This is the project's central deliverable — see `PROJECT.md`
§2 and §9.

**Decision:** _Fill in the actual finding, stated precisely: e.g. "Under
walk-forward validation, GARCH(1,1) achieved a QLIKE of X vs. the
gradient-boosting model's Y — the simple model won by Z%. Under the naive
random-split validation, the complex model had appeared to win by W%,
which walk-forward validation revealed was an artifact of leakage." (Or
whichever pattern was actually found — report it precisely.)_

**Consequences:** _Fill in — what does this suggest about when added
model complexity is/isn't worth it for this kind of forecasting task? This
is the actual interview-ready insight this project produces._

---

## ADR-006: Naive-vs-walk-forward comparison replaces gross-vs-net costs

**Status:** Accepted (Phase 0)

**Context:** The standalone-project brief template asks for a
with/without transaction costs comparison. This project is a forecasting
task, not a trading strategy, so there are no trades to cost.

**Decision:** The equivalent "honesty axis" is validation methodology:
every model is scored under both a naive random split and walk-forward
validation, and the two are reported side by side (`PROJECT.md` §9).

**Consequences:** The naive numbers are reported only as a contrast and
never as the project's result. If the two schemes happen to agree, that
is reported as-is.

---

## ADR-007: Simple baseline specifications

**Status:** Accepted (Phase 2)

**Context:** `PROJECT.md` §3 asks for EWMA and/or a GARCH-family model as
the simple baseline, and `TASKS.md` Phase 2 names GARCH(1,1) or a
documented alternative order. Both were specified before any model was
scored.

**Decision:**
- **EWMA** (`src/baseline_ewma.py`):
  `s2_{t+1|t} = 0.94 * s2_{t|t-1} + 0.06 * r_t^2`, zero mean, seeded with
  the mean squared return of the first 30 returns. λ = 0.94 is the
  RiskMetrics (1996) daily value and is **not estimated**, so EWMA has zero
  fitted parameters. EWMA treats variance as a random walk, so the 5-day
  forecast is `5 * s2_{t+1|t}`.
- **GARCH(1,1)** (`src/baseline_garch.py`): zero mean, Gaussian
  quasi-maximum likelihood via `arch` (returns scaled ×100 for the
  optimizer, parameters converted back). The 5-day forecast is the sum of
  the 1..5-step analytic forecasts,
  `s2_{t+k|t} = vbar + (α+β)^(k-1) (s2_{t+1|t} - vbar)`. Estimation and
  filtering are separate: parameters come from the harness's information
  set, and the recursion then runs with those parameters fixed, starting
  from the parameter-implied unconditional variance. Tests check the filter
  and the 5-day forecast against `arch`'s own output to 1e-9.

Why these choices:
- Zero mean for both: the target is a sum of squared returns without
  demeaning (ADR-002), and the mean daily return (≈0.0004) is negligible
  next to daily volatility (≈0.012).
- Gaussian QML rather than Student-t: QML estimates of the variance
  parameters stay consistent when returns are fat-tailed. The Gaussian
  one-step likelihood is also a QLIKE loss, so the estimator and the
  primary metric share one criterion.
- Fixed rather than estimated λ keeps EWMA as a zero-parameter reference.
  Nothing about it can be fitted to the test period under any validation
  scheme.
- **GJR-GARCH was considered and not used.** The EDA (run on the full
  sample) found a leverage effect, which symmetric GARCH(1,1) cannot
  represent. Switching the baseline on the strength of a full-sample EDA
  observation would be exactly the data-driven model choice ADR-001 says
  the EDA does not make. GARCH(1,1) is what the task specified, and it was
  kept.

**Consequences:**
- The simple side of the comparison cannot capture the leverage effect. A
  complex model that uses signed returns may gain from this, and that is a
  legitimate advantage, not an artefact. The results discussion must
  attribute it that way rather than to "ML" in general.
- EWMA with a fixed λ is not tuned to SPY, so it may be a weaker baseline
  than an estimated-λ EWMA. GARCH(1,1) is the estimated simple model.

---

## ADR-008: Naive random-split protocol and the recorded Phase 2 baseline numbers

**Status:** Accepted (Phase 2)

**Context:** `TASKS.md` Phase 2 requires the baselines to be scored under a
naive random split, and those numbers recorded before the complex model
exists. The same split has to be reusable for the complex model in Phase 5.

**Decision:**
- **Forecast dates:** every date with a complete 5-day target, after a
  burn-in of 252 rows (one trading year). That gives 8,030 dates, from
  1994-01-28 to 2025-12-23. The burn-in is shared by all models, so every
  model is scored on identical dates. It also caps the lookback of any
  Phase 3 feature at 252 rows unless the burn-in is revised for all models
  together.
- **Split:** the dates are shuffled once (`numpy` `default_rng(0)`) and
  split 80/20: 6,424 training dates and 1,606 test dates, interleaved in
  time.
- **Information set (shared with walk-forward):** a model is handed only
  the data its training rows touch. A training row at `t` has features
  dated `<= t` and a target built from returns `t+1..t+5`, so the history
  ends 5 rows after the last training date (`information_set_end` in
  `src/validation.py`). Under walk-forward, with training dates purged to
  `<= T-5`, this rule ends the history exactly at the forecast origin `T`.
  Under the random split, the last training date is near the end of the
  sample, so the history runs to 2025-12-31. For GARCH this means
  estimation on the full sample, including every test week. That is what a
  random split amounts to for a model estimated on one contiguous return
  series, rather than an extra choice.
- **Record:** `python -m src.evaluate naive-baselines` writes
  `docs/phase2_naive_baselines.json`. The output is deterministic and
  contains no timestamp. The commit that adds it dates the record.

**Recorded result (naive random split, 1,606 test dates; contrast only,
not the project's result):**

| Model | QLIKE (lower is better) | RMSE of 5-day variance |
|---|---|---|
| EWMA (λ = 0.94) | −6.6257 | 1.490e-03 |
| GARCH(1,1) | −6.6671 | 1.403e-03 |

The full-sample GARCH(1,1) estimate is ω = 2.13e-06, α = 0.111,
β = 0.873, persistence 0.984, and implied unconditional volatility 18.2%
annualized.

These numbers were recorded, then set aside. They carry no significance
test and are not evidence about out-of-sample performance.

**Consequences:**
- For these two baselines, the random split can affect the score through
  only two channels. First, the set of dates scored. Second, for GARCH
  only, estimating three parameters on data that include the test weeks.
  Neither model learns from individual target rows, so the overlap between
  training and test targets gives them nothing to exploit. A model fitted
  row by row to targets (Phase 3) is exposed to that overlap as well. How
  much any of this matters is a Phase 5 measurement, not an assumption.
- `PROJECT.md` §2 says the baseline's out-of-sample performance should be
  recorded before the complex model exists, while `TASKS.md` places the
  walk-forward harness in Phase 4. This phase follows `TASKS.md`, and only
  the naive number is on record so far.

---

## ADR-009: [Template for future ADRs — delete this line and use the format below]

**Status:** Proposed / Accepted / Superseded / Rejected

**Context:** What situation forced this decision?

**Decision:** What was decided?

**Consequences:** What trade-offs does this create? Be honest.
