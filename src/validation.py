"""
Validation harnesses (see DECISIONS.md ADR-003 and ADR-008).

Every model implements

    fit(history, train_dates)  estimate using only the rows of ``history``
    predict(data)              h-day variance forecast for every row of ``data``,
                               where the forecast at t uses data up to t only

The harness decides which forecast dates are training rows and hands the
model only the data those rows touch. A training row at ``t`` has features
dated ``<= t`` and a target built from returns ``t+1 .. t+h``, so ``history``
ends ``h`` rows after the last training date (``information_set_end``). The
same rule serves both schemes: under walk-forward with training dates purged
to ``<= T - h`` it ends exactly at the forecast origin ``T``.

Naive random split: eligible forecast dates are shuffled once with a fixed
seed and split 80/20. Test dates are interleaved with training dates, so a
training target overlaps the returns of neighbouring test targets. This is
kept only as the contrast in PROJECT.md section 9, never as the result.

Walk-forward (the reported scheme, ADR-003): expanding window, one fold per
calendar year from 2000. The fold whose first test date is ``T0`` trains on
eligible dates at least ``h`` rows before ``T0`` (purged), so its
information set ends at ``T0``. Parameters are then held fixed while the
model forecasts every date of that year from data up to each date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.target import HORIZON

# Forecast dates start after one trading year. Fixed before any model was
# compared, and shared by every model so all are scored on identical dates;
# it also bounds the lookback any feature may need (ADR-008).
BURN_IN = 252
NAIVE_TEST_FRACTION = 0.2
NAIVE_SEED = 0
WALK_FORWARD_START = "2000-01-01"


class Model(Protocol):
    name: str

    def fit(self, history: pd.DataFrame, train_dates: pd.DatetimeIndex) -> Model: ...

    def predict(self, data: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class Split:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def forecast_dates(target: pd.Series, burn_in: int = BURN_IN) -> pd.DatetimeIndex:
    """Dates usable as forecast origins: after the burn-in and with a complete target."""
    return pd.DatetimeIndex(target.iloc[burn_in:].dropna().index)


def naive_random_split(
    dates: pd.DatetimeIndex,
    test_fraction: float = NAIVE_TEST_FRACTION,
    seed: int = NAIVE_SEED,
) -> Split:
    """Shuffle ``dates`` and split them, ignoring time order (the naive scheme)."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    order = np.random.default_rng(seed).permutation(len(dates))
    n_test = round(test_fraction * len(dates))
    return Split(
        train=pd.DatetimeIndex(dates[np.sort(order[n_test:])]),
        test=pd.DatetimeIndex(dates[np.sort(order[:n_test])]),
    )


def information_set_end(
    index: pd.DatetimeIndex, train_dates: pd.DatetimeIndex, horizon: int = HORIZON
) -> pd.Timestamp:
    """Last date of data touched by the training rows (their latest target return)."""
    last = index.get_loc(train_dates.max())
    return index[min(last + horizon, len(index) - 1)]


def run_split(model: Model, data: pd.DataFrame, split: Split, horizon: int = HORIZON) -> pd.Series:
    """Fit ``model`` on the training rows' information set and forecast the test dates."""
    history = data.loc[: information_set_end(data.index, split.train, horizon)]
    model.fit(history, split.train)
    forecasts = model.predict(data).reindex(split.test)
    if forecasts.isna().any() or (forecasts <= 0).any():
        raise ValueError(f"{model.name} produced missing or non-positive forecasts on test dates.")
    return forecasts


def walk_forward_folds(
    index: pd.DatetimeIndex,
    dates: pd.DatetimeIndex,
    start: str = WALK_FORWARD_START,
    horizon: int = HORIZON,
) -> list[Split]:
    """
    Expanding-window folds with one calendar-year test block each.

    ``index`` is the full data index (row positions define the purge);
    ``dates`` are the eligible forecast dates.
    """
    positions = index.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Every forecast date must be in the data index.")
    test_dates = dates[dates >= pd.Timestamp(start)]
    folds = []
    for year in sorted(set(test_dates.year)):
        test = test_dates[test_dates.year == year]
        cutoff = index.get_loc(test[0]) - horizon
        train = dates[positions <= cutoff]
        if len(train) == 0:
            raise ValueError(f"No training dates before the {year} test block.")
        folds.append(Split(train=pd.DatetimeIndex(train), test=pd.DatetimeIndex(test)))
    return folds


def run_walk_forward(
    model: Model, data: pd.DataFrame, folds: list[Split], horizon: int = HORIZON
) -> tuple[pd.Series, list[dict]]:
    """Refit ``model`` at the start of each fold; return all test forecasts and per-fold fit info."""
    pieces, fits = [], []
    for fold in folds:
        info_end = information_set_end(data.index, fold.train, horizon)
        if info_end > fold.test[0]:
            raise ValueError(f"Fold starting {fold.test[0].date()} would see data after its origin.")
        pieces.append(run_split(model, data, fold, horizon))
        describe = getattr(model, "describe", dict)
        fits.append(
            {
                "test_start": fold.test[0].strftime("%Y-%m-%d"),
                "test_end": fold.test[-1].strftime("%Y-%m-%d"),
                "n_train": len(fold.train),
                "n_test": len(fold.test),
                "information_set_end": info_end.strftime("%Y-%m-%d"),
                "fit": describe(),
            }
        )
    forecasts = pd.concat(pieces)
    forecasts.name = model.name
    return forecasts, fits
