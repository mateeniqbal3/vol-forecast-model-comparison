# Results: naive vs. walk-forward validation, simple vs. complex

Every number here is produced by `python -m src.evaluate compare` and
stored in [`phase5_comparison.json`](phase5_comparison.json). The test
design and decision rule were fixed in
[`DECISIONS.md`](../DECISIONS.md) ADR-005 before any significance test was
computed.

**Setup.** SPY daily data, 1993–2025. The target is the 5-day forward
realized variance, `RV_t = r²_{t+1} + … + r²_{t+5}`. There are three
models:

- **EWMA:** λ = 0.94, fixed, so nothing is estimated.
- **GARCH(1,1):** zero mean, Gaussian QML, 3 parameters.
- **LightGBM:** gamma objective (the same as QLIKE up to terms that do not
  depend on the forecast), 18 features built from SPY's own OHLCV, and a
  12-point hyperparameter grid searched inside each training set. One
  design was tried (ADR-004).

Two validation schemes:

- **Naive:** 20% of forecast dates held out at random.
- **Walk-forward:** expanding window, refitted every calendar year from
  2000 to 2025, with a 5-day purge.

Losses are on the variance scale: QLIKE `log F + RV/F` (primary) and
RMSE. Lower is better for both.

## 1. The comparison table (PROJECT.md §9)

"Same dates" is the 1,321 naive test dates from 2000-01-03 on. Every
walk-forward model also forecast those dates, so the two middle columns
differ only in the validation scheme, not in the period scored.

**QLIKE**

| Model | Naive, all (n = 1,606) | Naive, same dates (n = 1,321) | Walk-forward, same dates (n = 1,321) | Walk-forward, all (n = 6,534) |
|---|---|---|---|---|
| EWMA | −6.6257 | −6.5997 | −6.5997 | −6.5887 |
| GARCH(1,1) | −6.6671 | −6.6438 | −6.6356 | −6.6327 |
| LightGBM | **−6.7413** | **−6.7240** | **−6.6586** | **−6.6451** |

**RMSE of the 5-day variance forecast**

| Model | Naive, all | Naive, same dates | Walk-forward, same dates | Walk-forward, all |
|---|---|---|---|---|
| EWMA | 1.490e-03 | 1.607e-03 | 1.607e-03 | 1.508e-03 |
| GARCH(1,1) | 1.403e-03 | 1.508e-03 | **1.545e-03** | **1.443e-03** |
| LightGBM | **1.328e-03** | **1.426e-03** | 1.680e-03 | 1.589e-03 |

**Diebold–Mariano tests, LightGBM minus GARCH(1,1).** Negative values
favor LightGBM. Standard errors are Newey–West. The primary lag follows
the pre-registered rule; the sensitivity lag is 63.

| Scheme | Metric | Mean loss difference | 95% CI | p (primary lag) | p (lag 63) |
|---|---|---|---|---|---|
| Naive | QLIKE | −0.0742 | — | < 0.0001 (lag 7) | < 0.0001 |
| Naive | MSE | −2.06e-07 | — | 0.016 (lag 7) | 0.018 |
| **Walk-forward** | **QLIKE** | **−0.0124** | **[−0.046, +0.022]** | **0.47 (lag 10)** | **0.65** |
| Walk-forward | MSE | +4.45e-07 | [−2.1e-07, +1.1e-06] | 0.18 (lag 10) | 0.36 |

The other walk-forward pairs:
- GARCH(1,1) beats EWMA on QLIKE by 0.044, with p = 0.0001 at lag 10 and
  0.0002 at lag 63.
- LightGBM beats EWMA on QLIKE by 0.056, with p = 0.011 at lag 10 but
  p = 0.070 at lag 63.

## 2. Did the naive split mislead the conclusion?

Yes, in two ways. The scores are on identical dates, so the sample period
plays no part.

1. **It turned an undetectable edge into a decisive-looking one.** On the
   same 1,321 dates, LightGBM's QLIKE lead over GARCH is 0.080 under the
   naive split but 0.023 under walk-forward. Over the full walk-forward
   sample the lead is 0.012, with p = 0.47. The naive-split DM test would
   have reported p < 0.0001.
2. **It reversed the RMSE ranking.** Under the naive split LightGBM has
   the lowest RMSE (p = 0.016 against GARCH). Under walk-forward it has
   the highest of the three models.

The naive split flatters the models in proportion to how much they learn
from the data. Each model's own QLIKE, naive minus walk-forward, on
identical dates:

| Model | Naive minus walk-forward QLIKE | DM p |
|---|---|---|
| EWMA | 0 (the forecasts are identical: nothing is estimated) | — |
| GARCH(1,1) | −0.0082 | 0.033 |
| LightGBM | −0.0654 | 0.004 |

The overstatement for LightGBM is eight times that for GARCH. It has two
identifiable sources.

- **Crisis weeks leak into training.** A random split puts weeks from 2008
  and 2020 into the training set of a model that is then tested on other
  weeks from the same episodes. On the 115 naive test dates in 2008 and
  2020, realized variance averaged 0.00284. The naive-split LightGBM
  forecast 0.00185 on average and the walk-forward LightGBM 0.00127.
  GARCH's forecasts were almost the same under both schemes (0.00232 vs.
  0.00233), because its three parameters barely move. 2008 alone accounts
  for −0.027 of LightGBM's −0.065 scheme effect.
- **The remainder holds outside the crisis years.** Excluding 2008 and
  2020 (a post-hoc, descriptive cut), the naive split still flatters
  LightGBM by 0.038, against 0.007 for GARCH. This is consistent with the
  leakage path ADR-008 identified in advance. A model fitted row by row
  to targets can learn from training dates next to each test date, whose
  targets share four of the test target's five returns. Slow-moving
  features such as the 252-day realized variance, drawdown and relative
  volume make it easy for trees to localize a date in time. GARCH is not
  fitted to individual targets, so it cannot exploit this. This project
  does not measure the two paths separately.

## 3. Why QLIKE and RMSE disagree under walk-forward

LightGBM has the best walk-forward QLIKE and the worst walk-forward RMSE.
Neither difference against GARCH is statistically significant. The
disagreement comes from the two losses weighting different weeks, and from
a structural limit of tree ensembles.

**The mechanism: tree forecasts have a ceiling, and walk-forward exposes
it.**

A boosted tree ensemble's forecast is a sum of leaf values learned from
the training set. Past the most extreme training inputs, the prediction is
flat. The 2008 fold shows this directly (`lightgbm_ceiling_probe`):

- The fold was trained on 1994 to late 2007. Its largest training target
  was 0.0099, about 70% annualized volatility.
- In the autumn of 2008 its inputs went far beyond that range. The 22-day
  realized variance reached 4.2 times its training maximum and was above
  that maximum on 65 test dates. The 5-day Parkinson range estimator
  reached 3.1 times its training maximum.
- On the worst week (forecast date 2008-10-08, realized 0.0352, about 133%
  annualized), the model forecast 0.00216. Multiplying every variance
  feature by 2, 10 or 100 leaves that forecast at exactly 0.00216.
- The model's largest forecast on any date in the sample is 0.00222,
  about 33% annualized volatility. That is a hard ceiling, below even its
  own training maximum, because in that fold each leaf averages at least
  50 training dates.
- In the same weeks GARCH forecast up to 0.0138 (about 83% annualized).
  Its forecast is linear in the latest squared return, so it rises with
  the shock whether or not anything like it appeared in training.

2008 therefore costs LightGBM +0.70 QLIKE per date, relative to GARCH,
over that year. Averaged over all 6,534 dates this is +0.027, more than
twice LightGBM's total advantage of 0.012. 2020 adds +0.005, with 3 test
dates above the training maximum.

**How this produces the metric disagreement.**

- **RMSE is dominated by a few extreme weeks.** The top 1% of realized
  weeks (66 dates) carry 83% of LightGBM's squared error, against 68% for
  GARCH and 65% for EWMA. LightGBM's mean forecast is 0.79 times the mean
  realized variance; GARCH's and EWMA's are 1.00. The shortfall is almost
  entirely in those weeks.
- **QLIKE scores relative errors, so every week counts about equally.** On
  the other 99% of dates LightGBM's QLIKE is −6.724, against −6.686 for
  GARCH. Its RMSE there is also the lowest: 6.5e-4, against 8.2e-4 for
  GARCH. That is a descriptive split, not a test.
- **In the top 66 weeks LightGBM is far worse on QLIKE too** (+1.05 vs.
  −1.38 for GARCH). But those weeks are 1% of the sample, so the QLIKE
  mean still favors it slightly. MSE weights errors by the square of the
  variance level, so those same weeks decide it.
- **LightGBM was trained on the gamma deviance, which is QLIKE.** It was
  optimized for relative accuracy in typical weeks, which is where it
  gains.

Year by year, LightGBM has the lower QLIKE in 21 of 26 walk-forward years
and the lower MSE in 18. It is worse on QLIKE in 2000, 2003, 2008, 2017
and 2020. Its advantage is small and consistent in ordinary conditions,
and its losses are concentrated in the rare episodes where volatility
leaves the range the model has seen.

## 4. Verdict: was the added complexity justified?

**No.** By the rule registered before testing (ADR-005), the result is
*not justified*.

Under walk-forward validation, LightGBM's mean QLIKE is −6.6451 against
−6.6327 for GARCH(1,1). The difference is 0.0124 in LightGBM's favor, with
DM p = 0.47 (lag 10) and p = 0.65 (lag 63). The 95% interval is −0.046 to
+0.022. Its RMSE is worse (1.589e-3 vs. 1.443e-3), though not
significantly (p = 0.18). Under the naive random split, the same model
appeared to beat GARCH decisively on both metrics (QLIKE p < 0.0001,
MSE p = 0.016).

The specific mechanism:
- The ML model gains a little in ordinary weeks.
- It gives that gain back when volatility moves outside the range of its
  training data. A tree ensemble's forecast saturates there, while
  GARCH's scales linearly with the latest shock.
- In the 2008 fold that ceiling was about 33% annualized volatility,
  during weeks that realized over 130%.
- The naive split hides this failure. It puts weeks from the crisis being
  tested into the training data, and it lets the model learn from
  neighboring, overlapping targets.

**Limits on the verdict.**
- The evidence does not show the models are equally good. The confidence
  interval does not rule out a LightGBM advantage of up to about 0.046
  QLIKE, or a disadvantage of up to about 0.022. What the evidence shows
  is that the added complexity did not deliver a detectable improvement,
  and that the best-looking result came from the naive scheme.
- The one step up in complexity that did pay off was EWMA to GARCH(1,1).
  Three estimated parameters bought a QLIKE improvement of 0.044 that
  holds at both lags (p ≤ 0.0002).

## 5. What didn't work, and limitations

- **Information, not only flexibility.** LightGBM also saw the daily
  high/low range, the open and volume, which the baselines did not. Even
  with that extra information it did not improve significantly. No
  experiment separates the value of the extra inputs from the value of
  the flexible model.
- **One design, one protocol.** The ML model had one configuration
  (ADR-004), and walk-forward had one scheme: expanding window, annual
  refit (ADR-003). A rolling window, more frequent refits, or a tree model
  that can extrapolate may behave differently in crises. One example of
  the last is LightGBM's `linear_tree`, or a model of the ratio of RV to a
  GARCH forecast. None of these was tried, because changing the design
  after seeing the results is exactly what the protocol forbids. They are
  hypotheses, not findings.
- **Noisy target, limited power.** A sum of five squared daily returns is
  a noisy proxy for realized variance. With intraday data the same
  forecasts would be ranked more sharply, and the ±0.034 confidence
  interval could narrow enough to be decisive.
- **Two crises dominate.** The verdict rests on a sample where two
  episodes, 2008 and 2020, decide the net result. The "excluding 2008 and
  2020" figures above are descriptive, chosen after seeing the results,
  and are not evidence that the ML model is better in calm markets.
- **Many tests reported, one decides.** Four schemes and metrics × three
  pairs × two lags are reported. Only the pre-registered primary test
  (walk-forward, QLIKE, LightGBM vs GARCH) determines the verdict.
- **Scope.** One instrument (SPY), daily frequency, 5-day horizon,
  2000–2025 out of sample. GJR-GARCH was deliberately not used (ADR-007),
  so the simple side does not model the leverage effect at all.
