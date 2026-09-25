"""
Tests for the validation harness.

Phase 2 covers the naive random split and the information-set rule shared
by both schemes. Phase 4 adds the walk-forward tests: no test-window
timestamp may precede or overlap its training window, and training rows
whose forward target window extends past the forecast origin must be purged
(see DECISIONS.md ADR-002).
"""

import numpy as np
import pandas as pd
import pytest

from src.data import add_log_returns
from src.target import HORIZON, forward_realized_variance
from src.validation import (
    BURN_IN,
    Split,
    forecast_dates,
    information_set_end,
    naive_random_split,
    run_split,
)


@pytest.fixture
def data(synthetic_prices):
    return add_log_returns(synthetic_prices)


def test_forecast_dates_skip_burn_in_and_incomplete_targets(data):
    target = forward_realized_variance(data["log_return"])
    dates = forecast_dates(target, burn_in=20)
    assert dates[0] == data.index[20]
    assert dates[-1] == data.index[-HORIZON - 1]
    assert target.loc[dates].notna().all()


def test_default_burn_in_is_one_trading_year():
    assert BURN_IN == 252


def test_naive_split_partitions_dates(data):
    dates = data.index
    split = naive_random_split(dates, test_fraction=0.2, seed=3)
    assert split.train.intersection(split.test).empty
    assert split.train.union(split.test).equals(dates)
    assert len(split.test) == round(0.2 * len(dates))
    assert split.train.is_monotonic_increasing and split.test.is_monotonic_increasing


def test_naive_split_is_reproducible_and_seed_dependent(data):
    a = naive_random_split(data.index, seed=0)
    b = naive_random_split(data.index, seed=0)
    c = naive_random_split(data.index, seed=1)
    assert a.test.equals(b.test)
    assert not a.test.equals(c.test)


def test_naive_split_interleaves_train_and_test(data):
    """The point of the naive scheme: test dates sit between training dates."""
    split = naive_random_split(data.index, seed=0)
    inside = (split.test > split.train.min()) & (split.test < split.train.max())
    assert inside.mean() > 0.9


def test_naive_split_rejects_bad_fraction(data):
    with pytest.raises(ValueError, match="test_fraction"):
        naive_random_split(data.index, test_fraction=1.0)


def test_information_set_ends_h_rows_after_last_training_date(data):
    idx = data.index
    assert information_set_end(idx, idx[[3, 50, 100]], horizon=5) == idx[105]
    # Clipped at the end of the sample.
    assert information_set_end(idx, idx[[len(idx) - 2]], horizon=5) == idx[-1]


class _RecordingModel:
    name = "recorder"

    def fit(self, history, train_dates):
        self.history_end = history.index.max()
        self.train_dates = train_dates
        return self

    def predict(self, data):
        return pd.Series(np.arange(1, len(data) + 1, dtype=float), index=data.index)


def test_run_split_passes_only_the_information_set(data):
    idx = data.index
    split = Split(train=idx[10:100], test=idx[150:160])
    model = _RecordingModel()
    forecasts = run_split(model, data, split)

    assert model.history_end == idx[99 + HORIZON]
    assert model.train_dates.equals(split.train)
    assert forecasts.index.equals(split.test)


def test_run_split_rejects_missing_forecasts(data):
    class Broken(_RecordingModel):
        def predict(self, data):
            return pd.Series(np.nan, index=data.index)

    idx = data.index
    with pytest.raises(ValueError, match="non-positive"):
        run_split(Broken(), data, Split(train=idx[:50], test=idx[60:70]))
