import numpy as np
import pandas as pd
import pytest

from src.data import (
    PRICE_COLUMNS,
    add_log_returns,
    clean_prices,
    load_processed,
    normalize_yfinance_frame,
)


def test_normalize_flattens_yfinance_multiindex(synthetic_prices):
    raw = synthetic_prices.copy()
    raw.columns = pd.MultiIndex.from_product(
        [[c.capitalize() for c in raw.columns], ["SPY"]], names=["Price", "Ticker"]
    )
    out = normalize_yfinance_frame(raw)
    assert list(out.columns) == PRICE_COLUMNS
    assert out.index.name == "date"


def test_clean_sorts_unsorted_input(synthetic_prices):
    shuffled = synthetic_prices.sample(frac=1.0, random_state=0)
    cleaned, _ = clean_prices(shuffled)
    assert cleaned.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(cleaned, synthetic_prices, check_freq=False)


def test_clean_drops_missing_close_without_forward_fill(synthetic_prices):
    raw = synthetic_prices.copy()
    gap_date = raw.index[50]
    raw.loc[gap_date, "close"] = np.nan

    cleaned, report = clean_prices(raw)

    assert gap_date not in cleaned.index
    assert report.missing_close_dropped == 1
    assert report.rows_out == len(raw) - 1
    # A forward fill would have produced an artificial zero return.
    returns = add_log_returns(cleaned)["log_return"]
    assert (returns != 0).all()


def test_clean_keeps_last_record_for_duplicate_dates(synthetic_prices):
    duplicate = synthetic_prices.iloc[[10]].copy()
    duplicate["close"] = 123.0
    raw = pd.concat([synthetic_prices, duplicate])

    cleaned, report = clean_prices(raw)

    assert report.duplicate_dates_dropped == 1
    assert cleaned.index.is_unique
    assert cleaned.loc[synthetic_prices.index[10], "close"] == 123.0


def test_clean_strips_timezone(synthetic_prices):
    raw = synthetic_prices.tz_localize("America/New_York")
    cleaned, _ = clean_prices(raw)
    assert cleaned.index.tz is None


def test_clean_rejects_non_positive_close(synthetic_prices):
    raw = synthetic_prices.copy()
    raw.iloc[20, raw.columns.get_loc("close")] = 0.0
    with pytest.raises(ValueError, match="Non-positive"):
        clean_prices(raw)


def test_clean_rejects_missing_columns(synthetic_prices):
    with pytest.raises(ValueError, match="missing columns"):
        clean_prices(synthetic_prices.drop(columns="volume"))


def test_clean_rejects_empty_frame(synthetic_prices):
    with pytest.raises(ValueError, match="empty"):
        clean_prices(synthetic_prices.iloc[:0])


def test_log_returns_values(synthetic_prices):
    out = add_log_returns(synthetic_prices)
    close = synthetic_prices["close"]
    expected = np.log(close.iloc[1:].to_numpy() / close.iloc[:-1].to_numpy())
    np.testing.assert_allclose(out["log_return"].to_numpy(), expected)
    assert out.index[0] == synthetic_prices.index[1]
    assert out["log_return"].notna().all()


def test_log_returns_reject_unsorted(synthetic_prices):
    with pytest.raises(ValueError, match="sorted"):
        add_log_returns(synthetic_prices.iloc[::-1])


def test_load_processed_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.data"):
        load_processed(tmp_path / "missing.csv")
