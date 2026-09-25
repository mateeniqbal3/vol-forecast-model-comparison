"""
EWMA volatility baseline (see DECISIONS.md ADR-007).

RiskMetrics exponentially weighted moving average of squared returns:

    s2_{t+1|t} = lam * s2_{t|t-1} + (1 - lam) * r_t^2

computed at the close of day ``t`` from returns up to and including ``r_t``.
EWMA treats variance as a random walk, so its forecast is the same for every
day ahead and the ``h``-day variance forecast is ``h * s2_{t+1|t}``.

``lam`` is fixed at the RiskMetrics (1996) daily value of 0.94, not
estimated, so this baseline has no fitted parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.target import HORIZON

LAMBDA = 0.94
INIT_WINDOW = 30  # returns averaged to seed the recursion


def one_step_variance(
    log_returns: pd.Series, lam: float = LAMBDA, init_window: int = INIT_WINDOW
) -> pd.Series:
    """
    ``s2_{t+1|t}`` aligned to row ``t``.

    The recursion is seeded at row ``init_window - 1`` with the mean squared
    return of the first ``init_window`` rows; earlier rows are NaN.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    if log_returns.isna().any():
        raise ValueError("log_returns contain NaN.")
    if len(log_returns) < init_window:
        raise ValueError(f"Need at least {init_window} returns, got {len(log_returns)}.")

    r2 = log_returns.to_numpy(dtype=float) ** 2
    out = np.full(len(r2), np.nan)
    s2 = r2[:init_window].mean()
    out[init_window - 1] = s2
    for i in range(init_window, len(r2)):
        s2 = lam * s2 + (1.0 - lam) * r2[i]
        out[i] = s2
    return pd.Series(out, index=log_returns.index, name="ewma_var_1d")


def forecast(log_returns: pd.Series, horizon: int = HORIZON, lam: float = LAMBDA) -> pd.Series:
    """``horizon``-day variance forecast made at the close of each row."""
    out = horizon * one_step_variance(log_returns, lam)
    out.name = f"ewma_var_{horizon}d"
    return out


class EWMAModel:
    """Validation-harness wrapper. Nothing is estimated, so ``fit`` is a no-op."""

    name = "EWMA"

    def fit(self, history: pd.DataFrame, train_dates: pd.DatetimeIndex) -> EWMAModel:
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return forecast(data["log_return"])

    def describe(self) -> dict:
        return {"lambda": LAMBDA, "init_window": INIT_WINDOW, "estimated_parameters": 0}
