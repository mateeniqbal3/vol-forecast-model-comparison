"""
Forward realized-variance forecasting target (see DECISIONS.md ADR-002).

A forecast is made at the close of day ``t``. Its target is the realized
variance of the next ``h`` daily log returns:

    RV_t = r_{t+1}^2 + r_{t+2}^2 + ... + r_{t+h}^2

Only returns strictly after ``t`` enter the target. The last ``h`` rows have
no complete forward window and are NaN, never partially filled.

Because the target at ``t`` is only fully observed at the close of ``t+h``,
a model trained at forecast origin ``T`` may only use target rows
``t <= T - h``. The walk-forward harness (Phase 4) must purge accordingly.
"""

from __future__ import annotations

import pandas as pd

HORIZON = 5  # trading days


def forward_realized_variance(log_returns: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Sum of squared log returns over ``t+1 .. t+horizon``, aligned to row ``t``."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not log_returns.index.is_monotonic_increasing:
        raise ValueError("log_returns must be sorted by date.")

    # rolling(h).sum() at row s covers r_{s-h+1..s}; shifting by -h moves that
    # value to row t = s - h, so it covers r_{t+1..t+h}.
    trailing = log_returns.pow(2).rolling(horizon, min_periods=horizon).sum()
    target = trailing.shift(-horizon)
    target.name = f"rv_fwd_{horizon}d"
    return target
