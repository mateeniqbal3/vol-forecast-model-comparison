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

**Status:** Accepted (Phase 4). Fixed before any walk-forward run of any
model.

**Context:** `PROJECT.md` §8 requires documented training window length,
forecast horizon, and step size for the walk-forward harness.

**Decision:** (`walk_forward_folds` in `src/validation.py`)
- **Window:** expanding. Each fold trains on every eligible forecast date
  (after the ADR-008 burn-in) up to the purge limit.
- **First test date:** 2000-01-03, the first trading day of 2000. The first
  training set holds 1,496 dates (1994-01-28 to late 1999).
- **Step and test block:** one calendar year. Each model is refitted once
  per year, at the first test date of the block, and forecasts every
  trading day of that year with its parameters fixed. Its inputs still
  update daily: the GARCH/EWMA filters and the ML features use data up to
  each forecast date. That makes 26 folds, 2000 to 2025, and 6,534 test
  dates (2000-01-03 to 2025-12-23).
- **Horizon and purge:** h = 5. The fold whose first test date is `T0`
  trains only on dates `t` at least 5 rows before `T0`, so every training
  target (`t+1..t+5`) is observed by the close of `T0`. The information set
  passed to `fit` therefore ends at `T0` (ADR-008's rule).
- **All three models use identical folds.**

Why:
- The 2000 start gives the ML model about six years of training in the
  first fold (about 1,200 inner-training and 300 inner-validation dates).
  The test span covers the 2000–02 bear market, 2008–09, 2011, 2015–16,
  2018, 2020 and 2022.
- An expanding window uses all available history, which suits the ML model
  (it needs data) and GARCH (whose estimates are noisy on short samples).
  A rolling window would adapt faster to regime change, at the cost of less
  data. It was not tried, so that no second protocol would be available to
  choose between after seeing results.
- An annual refit is a common, cheap schedule. Applying it to every model
  keeps the protocol identical.

**Consequences:**
- Parameters can be up to a year stale. This affects all models equally in
  schedule, but not necessarily in impact: the ML model may be more
  sensitive to staleness than the three-parameter GARCH.
- Expanding windows weight the calm 1990s and the 2008 episode ever less as
  the sample grows, but never drop them.
- The naive split scores random dates from 1994–2025, while walk-forward
  scores 2000–2025. Phase 5 must separate the effect of the sample period
  from the effect of the validation scheme, for example by also scoring the
  naive forecasts on dates from 2000 on only.

---

## ADR-004: Complex model feature set and configuration count

**Status:** Accepted (Phase 3). The configuration log below stays open
until Phase 5.

**Context:** `PROJECT.md` §2 requires the complex model be given a
genuinely fair chance, and `PROJECT.md` §12 requires disclosing how many
feature sets/hyperparameter configurations were tried.

**Decision:** The design below was fixed in Phase 3. The ML model had not
been scored under any scheme, and walk-forward numbers did not exist for
any model. The only scores in existence were the Phase 2 naive-split
baseline numbers (ADR-008). The design was chosen on the literature and on
general principles, not by reference to those numbers.

*Model.* LightGBM gradient-boosted trees (`src/complex_model.py`) with the
**gamma objective (log link)**. The gamma deviance of a forecast `F` for
`RV` is `2 (RV/F - log(RV/F) - 1)`, which is the project's QLIKE loss plus
terms that do not depend on `F`. The model is therefore trained on the
primary evaluation metric, and it predicts the conditional mean of RV,
which is the QLIKE-optimal forecast. That avoids the retransformation bias
of fitting log RV with squared error and exponentiating. RV is strictly
positive on every date (Phase 1 EDA: no week with RV = 0), as the gamma
objective requires.

*Features:* 18 in total, all computed at the close of `t` from SPY's own
OHLCV. The maximum lookback is 252 rows, the burn-in fixed in ADR-008.

| Group | Features | Rationale |
|---|---|---|
| Realized variance, HAR-style | mean r² over 1, 5, 22, 66, 252 days | Corsi (2009) HAR: daily/weekly/monthly components, plus quarterly and yearly levels that anchor mean reversion |
| Signed returns | sum of r over 1, 5, 22 days | Leverage effect and trend/drawdown dependence, which GARCH(1,1) cannot represent |
| Downside semivariance | mean of r²·1{r<0} over 5, 22 days | Barndorff-Nielsen, Kinnebrock & Shephard (2010); Patton & Sheppard (2015) |
| Range-based variance | Parkinson `ln(H/L)²/(4 ln 2)`, mean over 1, 5, 22 days | Uses intraday high/low; much less noisy than a squared close-to-close return |
| Overnight gap | `ln(O_t/C_{t-1})²`, mean over 1, 5 days | Separates the overnight part of close-to-close variance |
| Volume | `ln(V_t / mean V over 22d)`, `ln(mean V over 5d / mean V over 252d)` | Abnormal activity; relative to its own history because SPY volume grew by orders of magnitude after 1993 |
| Drawdown | `ln(C_t / max C over 252d)` | Stress regime indicator |

Tree models are invariant to monotone transforms of individual features,
so the features are left in their natural units. Excluded on purpose:
other tickers (e.g. VIX), which are a different information set, not more
model complexity. Also excluded: baseline forecasts as inputs, which would
make the model a stack of the baseline rather than an alternative to it.

*Training procedure (identical under every validation scheme; all
selection happens inside `fit`, on the training dates only):*
1. Split the training dates in time order: the first 80% for inner
   training, and the last 20% for inner validation. Inner-validation dates
   within 5 rows of the last inner-training date are purged, so no
   inner-training target overlaps an inner-validation target.
2. For each of 12 grid points, `num_leaves ∈ {7, 15, 31}` ×
   `min_child_samples ∈ {50, 200}` × `reg_lambda ∈ {0, 10}`, with fixed
   `learning_rate = 0.03`, `subsample = 0.8` (every iteration),
   `colsample_bytree = 0.8`, `seed = 0` and deterministic mode: train with
   early stopping (100 rounds, at most 2,000 trees) on inner-validation
   gamma deviance.
3. Pick the grid point with the lowest inner-validation QLIKE.
4. Refit on all training dates with that grid point and its early-stopped
   number of trees.

*Configuration log.* A configuration is one design of the model: its
features, objective, grid and training procedure. The automated 12-point
inner grid above belongs to configuration #1 and runs identically in every
fold. Any later change to the design, for any reason, is appended here
with the reason and with what the author had seen when making it.

| # | Date | Change | Reason | Model performance seen at the time |
|---|---|---|---|---|
| 1 | 2026-09-25 | Initial design (above) | Literature and principles, before any fit on real data | None for the ML model. Phase 2 naive baseline numbers only |

Operational smoke run after the design was fixed (not a configuration and
not an evaluation). It fitted configuration #1 on the real forecast dates up
to 2004-12-31 (2,752 dates, 2.4 s) and up to 2019-12-31 (6,527 dates,
4.1 s). It checked that no forecast date has incomplete features, and it
printed only run time, the selected grid point, the tree count (148 and
130) and the forecast range (median ≈13–14% annualized). No loss was
computed or printed, including the inner-validation scores. Nothing was
changed as a result.

Phase 4 validation run (2026-09-25, `python -m src.evaluate validate`): this
is the first time the ML model was scored under either scheme, and the first
walk-forward score for any model. Configuration #1 was run unchanged under
both schemes, and nothing in any model was changed after the scores were
seen. The per-fold fits were inspected for bugs only: tree counts 36–412,
the selected grid point varies by fold, and GARCH persistence drifts
smoothly from 0.995 to 0.985. No bug was found. Scores are in
`docs/phase4_validation_results.json`.

**Consequences:**
- The ML model has more information than the baselines, not just more
  flexibility: intraday range, open and volume. Any advantage it shows
  cannot be attributed to model complexity alone. Phase 5 must say so.
- Early stopping and the inner grid are chosen from the most recent 20%
  of each training set, which favors settings that work in the latest
  regime.
- The design is fixed before any evaluation, but it still uses the
  author's general knowledge of the volatility literature. That is a fair
  prior and does not come from this sample's out-of-sample results.

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
  walk-forward harness in Phase 4. The owner decided at the start of Phase 3
  to keep the `TASKS.md` order. No walk-forward number exists for any model
  until the complex model is built, so its feature design cannot target the
  baseline's walk-forward score. Only the naive baseline numbers existed
  while the complex model was designed.

---

## ADR-009: [Template for future ADRs — delete this line and use the format below]

**Status:** Proposed / Accepted / Superseded / Rejected

**Context:** What situation forced this decision?

**Decision:** What was decided?

**Consequences:** What trade-offs does this create? Be honest.
