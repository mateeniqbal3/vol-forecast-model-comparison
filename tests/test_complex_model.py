import numpy as np
import pandas as pd
import pytest

from src.baseline_garch import GarchParams
from src.baseline_garch import forecast as garch_forecast
from src.complex_model import (
    MAX_LOOKBACK,
    PARAM_GRID,
    GBMModel,
    build_features,
    inner_split,
)
from src.data import add_log_returns
from src.target import HORIZON
from src.validation import BURN_IN, forecast_dates
from tests.conftest import SYNTHETIC_GARCH, make_synthetic_prices

SMALL_GRID = {"num_leaves": [7], "min_child_samples": [20], "reg_lambda": [0.0]}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return add_log_returns(make_synthetic_prices(1500, seed=11))


@pytest.fixture(scope="module")
def split_dates(data):
    from src.target import forward_realized_variance

    dates = forecast_dates(forward_realized_variance(data["log_return"]))
    cut = 1000
    return dates[:cut], dates[cut + HORIZON:]


@pytest.fixture(scope="module")
def fitted(data, split_dates):
    train, _ = split_dates
    history = data.loc[: data.index[data.index.get_loc(train[-1]) + HORIZON]]
    return GBMModel(param_grid=SMALL_GRID, max_trees=300).fit(history, train)


# --- Features ---------------------------------------------------------------


def test_feature_set_matches_adr_004(data):
    assert build_features(data).shape[1] == 18


def test_features_complete_exactly_after_max_lookback(data):
    complete = build_features(data).notna().all(axis=1)
    assert not complete.iloc[MAX_LOOKBACK - 2]
    assert complete.iloc[MAX_LOOKBACK - 1:].all()
    assert MAX_LOOKBACK <= BURN_IN


def test_feature_values_by_hand(data):
    f = build_features(data)
    r = data["log_return"]
    t = 400
    assert f["rv_1d"].iloc[t] == pytest.approx(r.iloc[t] ** 2)
    assert f["rv_22d"].iloc[t] == pytest.approx((r.iloc[t - 21:t + 1] ** 2).mean())
    assert f["ret_5d"].iloc[t] == pytest.approx(r.iloc[t - 4:t + 1].sum())
    window = r.iloc[t - 4:t + 1]
    assert f["semivar_down_5d"].iloc[t] == pytest.approx((window.clip(upper=0) ** 2).mean())
    hi, lo = data["high"].iloc[t], data["low"].iloc[t]
    assert f["parkinson_1d"].iloc[t] == pytest.approx(np.log(hi / lo) ** 2 / (4 * np.log(2)))
    gap = np.log(data["open"].iloc[t] / data["close"].iloc[t - 1])
    assert f["overnight_1d"].iloc[t] == pytest.approx(gap**2)
    vol = data["volume"]
    assert f["volume_rel_22d"].iloc[t] == pytest.approx(
        np.log(vol.iloc[t] / vol.iloc[t - 21:t + 1].mean())
    )
    close = data["close"]
    assert f["drawdown_252d"].iloc[t] == pytest.approx(
        np.log(close.iloc[t] / close.iloc[t - 251:t + 1].max())
    )
    assert f["drawdown_252d"].iloc[t] <= 0


def test_zero_volume_gives_nan_not_infinite(data):
    zeroed = data.copy()
    zeroed.iloc[500, zeroed.columns.get_loc("volume")] = 0.0
    f = build_features(zeroed)
    assert not np.isinf(f.to_numpy()).any()
    assert np.isnan(f["volume_rel_22d"].iloc[500])


# --- Inner split ------------------------------------------------------------


def test_inner_split_is_chronological_and_purged(data, split_dates):
    train, _ = split_dates
    inner_train, inner_val = inner_split(data.index, train)
    assert inner_train.max() < inner_val.min()
    gap = data.index.get_loc(inner_val.min()) - data.index.get_loc(inner_train.max())
    assert gap >= HORIZON
    # Every non-purged date is used exactly once.
    assert inner_train.intersection(inner_val).empty
    assert len(train) - len(inner_train) - len(inner_val) == HORIZON - 1
    assert len(inner_train) == round(0.8 * len(train))


def test_inner_split_purge_works_on_non_contiguous_dates(data):
    """Naive random-split training dates have gaps; the purge is in rows of the full index."""
    train = data.index[300:1300:2]
    inner_train, inner_val = inner_split(data.index, train)
    gap = data.index.get_loc(inner_val.min()) - data.index.get_loc(inner_train.max())
    assert gap >= HORIZON


# --- Fit / predict ----------------------------------------------------------


def test_default_grid_has_twelve_points():
    assert len(GBMModel()._grid()) == 12
    assert PARAM_GRID == {
        "num_leaves": [7, 15, 31],
        "min_child_samples": [50, 200],
        "reg_lambda": [0.0, 10.0],
    }


def test_predictions_positive_on_complete_rows_nan_in_burn_in(fitted, data):
    pred = fitted.predict(data)
    assert pred.iloc[: MAX_LOOKBACK - 1].isna().all()
    assert (pred.iloc[MAX_LOOKBACK - 1:] > 0).all()


def test_fit_is_deterministic(data, split_dates, fitted):
    train, _ = split_dates
    history = data.loc[: data.index[data.index.get_loc(train[-1]) + HORIZON]]
    again = GBMModel(param_grid=SMALL_GRID, max_trees=300).fit(history, train)
    pd.testing.assert_series_equal(again.predict(data), fitted.predict(data))


def test_fit_summary_records_selection(fitted, split_dates):
    s = fitted.describe()
    assert s["n_train"] == len(split_dates[0])
    assert s["grid_points"] == 1
    assert 1 <= s["n_trees"] <= 300


def test_learns_conditional_variance_out_of_sample(fitted, data, split_dates):
    """On simulated GARCH data, forecasts should track the true conditional 5-day variance."""
    _, test = split_dates
    truth = garch_forecast(data["log_return"], GarchParams(**SYNTHETIC_GARCH)).loc[test]
    pred = fitted.predict(data).loc[test]
    assert np.corrcoef(np.log(pred), np.log(truth))[0, 1] > 0.6


def test_fit_rejects_train_dates_without_complete_target(data, split_dates):
    train, _ = split_dates
    history = data.loc[: train[-1]]  # last targets need returns past the end
    with pytest.raises(ValueError, match="complete target"):
        GBMModel(param_grid=SMALL_GRID).fit(history, train)


def test_fit_rejects_train_dates_inside_burn_in(data):
    train = data.index[100:600]
    with pytest.raises(ValueError, match="incomplete features"):
        GBMModel(param_grid=SMALL_GRID).fit(data, train)


def test_predict_before_fit_raises(data):
    with pytest.raises(RuntimeError, match="fit"):
        GBMModel().predict(data)
