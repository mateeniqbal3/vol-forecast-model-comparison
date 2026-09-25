import numpy as np
import pandas as pd
import pytest

from src.evaluate import qlike_losses, score, squared_errors


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
