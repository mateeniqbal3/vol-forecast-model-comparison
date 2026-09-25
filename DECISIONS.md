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

**Status:** Accepted (Phase 1). Metric implementation lands in Phase 5.

**Context:** `PROJECT.md` §7 specifies a short forward window (e.g. 1-5
trading days) and QLIKE and/or RMSE as the evaluation metric.

**Decision:**
- **Horizon:** h = 5 trading days (one trading week).
- **Target:** forward realized variance
  `RV_t = r_{t+1}^2 + ... + r_{t+5}^2`, where `r` is the daily log return
  and the forecast is made at the close of day t (`src/target.py`). The
  target is in variance units; volatility forecasts from every model are
  converted to 5-day variance before scoring.
- **Primary metric:** QLIKE in the form `L(RV, F) = log(F) + RV / F`,
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

## ADR-007: [Template for future ADRs — delete this line and use the format below]

**Status:** Proposed / Accepted / Superseded / Rejected

**Context:** What situation forced this decision?

**Decision:** What was decided?

**Consequences:** What trade-offs does this create? Be honest.
