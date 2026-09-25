import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from src.baseline_garch import (
    SCALE,
    GARCHModel,
    GarchParams,
    fit,
    forecast,
    one_step_variance,
)
from src.data import add_log_returns

TRUE = GarchParams(omega=2e-6, alpha=0.08, beta=0.9)


def _simulate(params: GarchParams, n: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    s2 = params.unconditional_variance
    r = np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_normal()
        s2 = params.omega + params.alpha * r[i] ** 2 + params.beta * s2
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=n))


@pytest.fixture(scope="module")
def simulated() -> pd.Series:
    return _simulate(TRUE, 4000)


@pytest.fixture(scope="module")
def arch_result(simulated):
    return arch_model(
        SCALE * simulated, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")


def test_fit_recovers_simulated_parameters(simulated):
    est = fit(simulated)
    assert est.alpha == pytest.approx(TRUE.alpha, abs=0.03)
    assert est.beta == pytest.approx(TRUE.beta, abs=0.04)
    assert est.unconditional_variance == pytest.approx(TRUE.unconditional_variance, rel=0.3)


def test_fit_converts_arch_scale(simulated, arch_result):
    est = fit(simulated)
    assert est.omega == pytest.approx(arch_result.params["omega"] / SCALE**2)
    assert est.alpha == pytest.approx(arch_result.params["alpha[1]"])
    assert est.beta == pytest.approx(arch_result.params["beta[1]"])


def test_filter_matches_arch_conditional_variance(simulated, arch_result):
    """Same parameters and starting value -> identical s2_{t+1|t} path to arch's."""
    params = fit(simulated)
    arch_var = arch_result.conditional_volatility.to_numpy() ** 2 / SCALE**2
    ours = one_step_variance(simulated, params, initial_variance=arch_var[0])
    # Ours at row t is s2_{t+1|t}; arch's at row t+1 is the same quantity.
    np.testing.assert_allclose(ours.to_numpy()[:-1], arch_var[1:], rtol=1e-9)


def test_five_day_forecast_matches_arch(simulated, arch_result):
    params = fit(simulated)
    arch_var = arch_result.conditional_volatility.to_numpy() ** 2 / SCALE**2
    ours = forecast(simulated, params, horizon=5, initial_variance=arch_var[0])
    arch_5d = arch_result.forecast(horizon=5, reindex=False).variance.iloc[-1].sum() / SCALE**2
    assert ours.iloc[-1] == pytest.approx(arch_5d, rel=1e-9)


def test_forecast_at_unconditional_variance_is_h_times_vbar():
    params = TRUE
    r = pd.Series(np.full(3, np.sqrt(params.unconditional_variance)))
    f = forecast(r, params, horizon=5)
    np.testing.assert_allclose(f.to_numpy(), 5 * params.unconditional_variance)


def test_multi_step_forecast_reverts_towards_vbar():
    params = TRUE
    vbar = params.unconditional_variance
    calm = forecast(pd.Series([0.0] * 50), params, horizon=5).iloc[-1]
    stressed = forecast(pd.Series([0.05]), params, horizon=5).iloc[-1]
    one_calm = one_step_variance(pd.Series([0.0] * 50), params).iloc[-1]
    one_stressed = one_step_variance(pd.Series([0.05]), params).iloc[-1]
    # Mean reversion: the 5-day forecast lies between 5 * s2_{t+1|t} and 5 * vbar.
    assert 5 * one_calm < calm < 5 * vbar
    assert 5 * vbar < stressed < 5 * one_stressed


def test_rejects_non_stationary_or_invalid_params():
    r = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="stationary"):
        one_step_variance(r, GarchParams(omega=1e-6, alpha=0.1, beta=0.9))
    with pytest.raises(ValueError, match="positive"):
        one_step_variance(r, GarchParams(omega=0.0, alpha=0.1, beta=0.8))


def test_model_wrapper_estimates_on_history_only(synthetic_prices):
    data = add_log_returns(synthetic_prices)
    history = data.iloc[:200]
    model = GARCHModel().fit(history, history.index[:150])
    assert model.params == fit(history["log_return"])

    with pytest.raises(RuntimeError, match="fit"):
        GARCHModel().predict(data)
    pd.testing.assert_series_equal(model.predict(data), forecast(data["log_return"], model.params))
