"""
GARCH(1,1) volatility baseline (see DECISIONS.md ADR-007).

    r_t = s_t * z_t,    s2_{t+1|t} = omega + alpha * r_t^2 + beta * s2_{t|t-1}

Zero conditional mean, parameters estimated by Gaussian quasi-maximum
likelihood with the ``arch`` package. The ``h``-day variance forecast made at
the close of ``t`` is the sum of the 1..h step-ahead forecasts, which revert
geometrically towards the unconditional variance:

    s2_{t+k|t} = vbar + (alpha + beta)^(k-1) * (s2_{t+1|t} - vbar)

Estimation and filtering are separate. ``fit`` estimates parameters from the
returns it is given, and ``one_step_variance``/``forecast`` run the recursion
over any return series with those parameters fixed. The walk-forward harness
therefore estimates on the information set at each origin and forecasts
causally past it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.target import HORIZON

SCALE = 100.0  # arch's optimizer is better conditioned on percent returns


@dataclass(frozen=True)
class GarchParams:
    omega: float  # squared log-return units
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def validate(self) -> None:
        if self.omega <= 0 or self.alpha < 0 or self.beta < 0:
            raise ValueError(f"GARCH parameters must be positive: {self}")
        if self.persistence >= 1:
            raise ValueError(f"GARCH is not covariance-stationary (alpha + beta >= 1): {self}")


def fit(log_returns: pd.Series) -> GarchParams:
    """Gaussian QML estimate of a zero-mean GARCH(1,1) on ``log_returns``."""
    from arch import arch_model

    if log_returns.isna().any():
        raise ValueError("log_returns contain NaN.")
    model = arch_model(
        SCALE * log_returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    result = model.fit(disp="off")
    if result.convergence_flag != 0:
        raise RuntimeError(f"GARCH estimation did not converge (flag {result.convergence_flag}).")
    p = result.params
    params = GarchParams(
        omega=float(p["omega"]) / SCALE**2, alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"])
    )
    params.validate()
    return params


def one_step_variance(
    log_returns: pd.Series, params: GarchParams, initial_variance: float | None = None
) -> pd.Series:
    """
    ``s2_{t+1|t}`` aligned to row ``t``.

    The recursion starts from ``s2`` for the first row, which defaults to the
    unconditional variance implied by ``params``. That default depends on the
    parameters only, not on any returns.
    """
    params.validate()
    if log_returns.isna().any():
        raise ValueError("log_returns contain NaN.")

    r2 = log_returns.to_numpy(dtype=float) ** 2
    s2 = params.unconditional_variance if initial_variance is None else initial_variance
    out = np.empty(len(r2))
    for i in range(len(r2)):
        s2 = params.omega + params.alpha * r2[i] + params.beta * s2
        out[i] = s2
    return pd.Series(out, index=log_returns.index, name="garch_var_1d")


def forecast(
    log_returns: pd.Series,
    params: GarchParams,
    horizon: int = HORIZON,
    initial_variance: float | None = None,
) -> pd.Series:
    """``horizon``-day variance forecast (sum of 1..horizon step forecasts) at each row."""
    s2_next = one_step_variance(log_returns, params, initial_variance)
    vbar = params.unconditional_variance
    decay = float(np.sum(params.persistence ** np.arange(horizon)))
    out = horizon * vbar + decay * (s2_next - vbar)
    out.name = f"garch_var_{horizon}d"
    return out


class GARCHModel:
    """Validation-harness wrapper: estimates on every return in ``history``."""

    name = "GARCH(1,1)"

    def __init__(self) -> None:
        self.params: GarchParams | None = None

    def fit(self, history: pd.DataFrame, train_dates: pd.DatetimeIndex) -> GARCHModel:
        self.params = fit(history["log_return"])
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.params is None:
            raise RuntimeError("Call fit() before predict().")
        return forecast(data["log_return"], self.params)

    def describe(self) -> dict:
        if self.params is None:
            return {}
        return {
            **asdict(self.params),
            "persistence": self.params.persistence,
            "unconditional_annualized_vol": float(np.sqrt(252 * self.params.unconditional_variance)),
            "estimated_parameters": 3,
        }
