import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.evaluate import (
    _contributions,
    diebold_mariano,
    newey_west_lag,
    newey_west_variance,
    qlike_losses,
    score,
    squared_errors,
)


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)), dtype=float)


def test_qlike_and_mse_values():
    rv, f = _series([1e-4, 4e-4]), _series([2e-4, 2e-4])
    np.testing.assert_allclose(
        qlike_losses(rv, f).to_numpy(), [np.log(2e-4) + 0.5, np.log(2e-4) + 2.0]
    )
    np.testing.assert_allclose(squared_errors(rv, f).to_numpy(), [1e-8, 4e-8])


def test_score_aggregates():
    rv, f = _series([1e-4, 4e-4]), _series([2e-4, 2e-4])
    s = score(rv, f)
    assert s["n"] == 2
    assert s["rmse"] == pytest.approx(np.sqrt(2.5e-8))
    assert s["qlike"] == pytest.approx(np.log(2e-4) + 1.25)


def test_qlike_is_minimized_by_the_true_variance():
    rv = _series([3e-4])
    grid = np.linspace(1e-4, 9e-4, 81)
    losses = [qlike_losses(rv, _series([g])).iloc[0] for g in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(3e-4)


def test_qlike_penalizes_under_prediction_more():
    rv = _series([4e-4])
    under = qlike_losses(rv, _series([2e-4])).iloc[0]
    over = qlike_losses(rv, _series([8e-4])).iloc[0]
    assert under > over


def test_qlike_is_defined_when_realized_is_zero():
    assert np.isfinite(qlike_losses(_series([0.0]), _series([1e-4])).iloc[0])


def test_rejects_misaligned_nan_or_non_positive_inputs():
    rv = _series([1e-4, 2e-4])
    with pytest.raises(ValueError, match="index"):
        score(rv, rv.iloc[:1])
    with pytest.raises(ValueError, match="NaN"):
        score(rv, _series([np.nan, 1e-4]))
    with pytest.raises(ValueError, match="positive"):
        score(rv, _series([0.0, 1e-4]))


# --- Diebold-Mariano ----------------------------------------------------------


def test_newey_west_lag_rule():
    assert newey_west_lag(6534) == 10
    assert newey_west_lag(1606) == 7
    assert newey_west_lag(50) == 4  # never below h - 1


def test_newey_west_lag_zero_is_the_plain_variance():
    x = np.random.default_rng(0).normal(size=500)
    assert newey_west_variance(x, 0) == pytest.approx(np.var(x))


@pytest.mark.parametrize("lag", [1, 4, 10])
def test_newey_west_matches_statsmodels(lag):
    rng = np.random.default_rng(1)
    e = rng.normal(size=800)
    x = np.convolve(e, np.ones(5), mode="valid")  # MA(4), like overlapping 5-day targets
    ols = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    assert newey_west_variance(x, lag) / len(x) == pytest.approx(ols.bse[0] ** 2, rel=1e-10)


def test_dm_detects_a_better_forecast_and_is_antisymmetric():
    rng = np.random.default_rng(2)
    base = _series(rng.gamma(2.0, 1.0, 1000))
    better = base - 0.3
    ab = diebold_mariano(better, base + rng.normal(0, 0.1, 1000), lag=5)
    ba = diebold_mariano(base + rng.normal(0, 0.1, 1000), better, lag=5)
    assert ab["mean_diff"] < 0 and ab["dm_stat"] < -5 and ab["p_value"] < 1e-6
    assert ba["dm_stat"] > 5


def test_dm_p_value_uses_the_normal_distribution():
    # Construct a differential whose statistic is exactly 1.96 with lag 0.
    d = np.array([1.0, -1.0] * 50)
    d = d + 1.96 * d.std() / np.sqrt(len(d))
    res = diebold_mariano(_series(d), _series(np.zeros_like(d)), lag=0)
    assert res["dm_stat"] == pytest.approx(1.96)
    assert res["p_value"] == pytest.approx(0.05, abs=1e-3)


def test_dm_rejects_misaligned_or_degenerate_losses():
    a = _series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="index"):
        diebold_mariano(a, a.iloc[:2], lag=1)
    with pytest.raises(ValueError, match="variance"):
        diebold_mariano(a, a, lag=1)


def test_dm_confidence_interval_brackets_the_mean():
    rng = np.random.default_rng(3)
    res = diebold_mariano(_series(rng.normal(0.1, 1, 500)), _series(np.zeros(500)), lag=4)
    lo, hi = res["ci95"]
    assert lo < res["mean_diff"] < hi
    assert hi - lo == pytest.approx(2 * 1.96 * res["se"])
    assert res["dm_stat"] == pytest.approx(res["mean_diff"] / res["se"])


def test_yearly_contributions_sum_to_the_mean():
    idx = pd.bdate_range("2007-01-01", "2009-12-31")
    diff = pd.Series(np.random.default_rng(4).normal(size=len(idx)), index=idx)
    out = _contributions(diff)
    assert sum(out["by_year"].values()) == pytest.approx(out["mean"])
    assert out["posthoc_mean_excluding_extrapolation_years"] == pytest.approx(
        diff[diff.index.year != 2008].mean()
    )
