import numpy as np
import pandas as pd
import pytest

from src.baseline_ewma import INIT_WINDOW, LAMBDA, EWMAModel, forecast, one_step_variance
from src.data import add_log_returns


def _returns(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)))


def test_recursion_matches_hand_computation():
    r = _returns([0.01, -0.02, 0.03, 0.00, -0.01])
    s2 = one_step_variance(r, lam=0.9, init_window=2)

    seed = (0.01**2 + 0.02**2) / 2
    expected = [np.nan, seed]
    for x in [0.03, 0.00, -0.01]:
        expected.append(0.9 * expected[-1] + 0.1 * x**2)
    np.testing.assert_allclose(s2.to_numpy(), expected, equal_nan=True)


def test_rows_before_seed_are_nan():
    r = _returns(np.full(50, 0.01))
    s2 = one_step_variance(r)
    assert s2.iloc[: INIT_WINDOW - 1].isna().all()
    assert s2.iloc[INIT_WINDOW - 1:].notna().all()


def test_constant_returns_give_constant_variance():
    r = _returns(np.full(60, 0.02))
    s2 = one_step_variance(r).dropna()
    np.testing.assert_allclose(s2.to_numpy(), 0.02**2)


def test_h_day_forecast_is_h_times_one_step(synthetic_prices):
    r = add_log_returns(synthetic_prices)["log_return"]
    np.testing.assert_allclose(
        forecast(r, horizon=5).to_numpy(), 5 * one_step_variance(r).to_numpy(), equal_nan=True
    )


def test_latest_return_enters_with_weight_one_minus_lambda(synthetic_prices):
    r = add_log_returns(synthetic_prices)["log_return"]
    base = one_step_variance(r)
    bumped_r = r.copy()
    bumped_r.iloc[100] += 0.05
    bumped = one_step_variance(bumped_r)

    assert bumped.iloc[99] == base.iloc[99]
    delta = bumped_r.iloc[100] ** 2 - r.iloc[100] ** 2
    assert bumped.iloc[100] - base.iloc[100] == pytest.approx((1 - LAMBDA) * delta)


def test_rejects_bad_inputs():
    r = _returns(np.full(40, 0.01))
    with pytest.raises(ValueError, match="lam"):
        one_step_variance(r, lam=1.0)
    with pytest.raises(ValueError, match="at least"):
        one_step_variance(r.iloc[:5])
    with_nan = r.copy()
    with_nan.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        one_step_variance(with_nan)


def test_model_wrapper_estimates_nothing(synthetic_prices):
    data = add_log_returns(synthetic_prices)
    model = EWMAModel().fit(data.iloc[:10], data.index[:5])
    pd.testing.assert_series_equal(model.predict(data), forecast(data["log_return"]))
    assert model.describe()["estimated_parameters"] == 0
