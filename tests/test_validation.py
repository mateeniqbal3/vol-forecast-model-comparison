"""
Tests for the validation harness.

Covers the naive random split, the information-set rule shared by both
schemes, and walk-forward: no test-window timestamp may precede or overlap
its training window, and training rows whose forward target window extends
past the forecast origin are purged (DECISIONS.md ADR-002, ADR-003).
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src.baseline_ewma import EWMAModel
from src.baseline_garch import GARCHModel
from src.data import add_log_returns
from src.target import HORIZON, forward_realized_variance
from src.validation import (
    BURN_IN,
    Split,
    forecast_dates,
    information_set_end,
    naive_random_split,
    run_split,
    run_walk_forward,
    walk_forward_folds,
)
from tests.conftest import make_synthetic_prices


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

    def __init__(self):
        self.history_ends = []

    def fit(self, history, train_dates):
        self.history_end = history.index.max()
        self.history_ends.append(self.history_end)
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


# --- Walk-forward -----------------------------------------------------------


@pytest.fixture(scope="module")
def long_data():
    return add_log_returns(make_synthetic_prices(1300, seed=5))


@pytest.fixture(scope="module")
def wf(long_data):
    dates = forecast_dates(forward_realized_variance(long_data["log_return"]), burn_in=100)
    folds = walk_forward_folds(long_data.index, dates, start="2022-01-01")
    return dates, folds


def test_walk_forward_has_one_fold_per_calendar_year(wf):
    dates, folds = wf
    expected_years = sorted(set(dates[dates >= "2022-01-01"].year))
    assert [f.test[0].year for f in folds] == expected_years
    for fold in folds:
        assert set(fold.test.year) == {fold.test[0].year}


def test_walk_forward_test_blocks_partition_the_test_period(wf):
    dates, folds = wf
    all_test = pd.DatetimeIndex(np.concatenate([f.test.to_numpy() for f in folds]))
    assert all_test.is_unique and all_test.is_monotonic_increasing
    assert all_test.equals(dates[dates >= "2022-01-01"])


def test_no_test_date_precedes_or_overlaps_its_training_window(wf):
    _, folds = wf
    for fold in folds:
        assert fold.train.max() < fold.test.min()
        assert fold.train.intersection(fold.test).empty


def test_training_targets_are_purged_before_the_fold_origin(wf, long_data):
    """Every training target t+1..t+h must be observed by the close of the first test date."""
    _, folds = wf
    idx = long_data.index
    for fold in folds:
        origin = idx.get_loc(fold.test[0])
        last_target_return = idx.get_indexer(fold.train) + HORIZON
        assert last_target_return.max() <= origin
        # The purge removes exactly the h dates whose targets are not yet observed.
        assert idx.get_loc(fold.train.max()) == origin - HORIZON


def test_information_set_ends_at_the_fold_origin(wf, long_data):
    _, folds = wf
    for fold in folds:
        assert information_set_end(long_data.index, fold.train) == fold.test[0]


def test_walk_forward_window_expands(wf):
    dates, folds = wf
    for earlier, later in pairwise(folds):
        assert earlier.train.isin(later.train).all()
        assert len(later.train) > len(earlier.train)
    assert folds[0].train[0] == dates[0]


def test_run_walk_forward_never_passes_data_after_the_origin(wf, long_data):
    _, folds = wf
    model = _RecordingModel()
    forecasts, fits = run_walk_forward(model, long_data, folds)

    assert model.history_ends == [f.test[0] for f in folds]
    assert forecasts.index.equals(pd.DatetimeIndex(np.concatenate([f.test for f in folds])))
    assert [fit["information_set_end"] for fit in fits] == [
        f.test[0].strftime("%Y-%m-%d") for f in folds
    ]


def test_run_walk_forward_rejects_a_leaking_fold(long_data):
    idx = long_data.index
    leaking = Split(train=idx[100:400], test=idx[398:450])
    with pytest.raises(ValueError, match="after its origin"):
        run_walk_forward(_RecordingModel(), long_data, [leaking])


def test_walk_forward_rejects_start_without_training_dates(long_data):
    dates = forecast_dates(forward_realized_variance(long_data["log_return"]), burn_in=100)
    with pytest.raises(ValueError, match="No training dates"):
        walk_forward_folds(long_data.index, dates, start="2000-01-01")


def test_walk_forward_runs_the_baselines_end_to_end(wf, long_data):
    _, folds = wf
    for model in (EWMAModel(), GARCHModel()):
        forecasts, fits = run_walk_forward(model, long_data, folds)
        assert (forecasts > 0).all() and len(fits) == len(folds)


def test_garch_is_refitted_every_fold(wf, long_data):
    """Each fold re-estimates on a longer history, so estimates differ between folds."""
    _, folds = wf
    _, fits = run_walk_forward(GARCHModel(), long_data, folds)
    alphas = [fit["fit"]["alpha"] for fit in fits]
    assert len(set(alphas)) == len(alphas)
