import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """300 business days of OHLCV with GARCH(1,1)-style volatility clustering."""
    rng = np.random.default_rng(42)
    n = 300
    omega, alpha, beta = 2e-6, 0.08, 0.9
    var = omega / (1 - alpha - beta)
    returns = np.empty(n)
    for i in range(n):
        returns[i] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * returns[i] ** 2 + beta * var

    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
