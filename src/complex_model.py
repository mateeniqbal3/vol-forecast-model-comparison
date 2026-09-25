"""
Gradient-boosted volatility model (see DECISIONS.md ADR-004).

LightGBM with the gamma objective (log link), whose deviance equals the
project's QLIKE loss up to terms that do not depend on the forecast, so the
model is trained on the primary metric and predicts the conditional mean of
the 5-day realized variance directly.

Features are 18 transforms of SPY's own daily OHLCV, all known at the close
of ``t`` and with a lookback of at most ``MAX_LOOKBACK`` rows (the shared
burn-in, ADR-008). Hyperparameters are chosen inside ``fit`` from the
training dates alone: a chronological, purged inner split, a fixed 12-point
grid with early stopping, then a refit on every training date.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.target import HORIZON, forward_realized_variance

MAX_LOOKBACK = 252
INNER_VALIDATION_FRACTION = 0.2
EARLY_STOPPING_ROUNDS = 100
MAX_TREES = 2000

PARAM_GRID = {
    "num_leaves": [7, 15, 31],
    "min_child_samples": [50, 200],
    "reg_lambda": [0.0, 10.0],
}
FIXED_PARAMS = {
    "objective": "gamma",
    "metric": "gamma_deviance",
    "learning_rate": 0.03,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}


def _mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix aligned to row ``t``, using data dated ``<= t`` only."""
    r = data["log_return"]
    r2 = r**2
    down2 = r2.where(r < 0, 0.0)
    parkinson = np.log(data["high"] / data["low"]) ** 2 / (4 * np.log(2))
    overnight2 = np.log(data["open"] / data["close"].shift(1)) ** 2
    volume = data["volume"].where(data["volume"] > 0)

    features = {
        "rv_1d": r2,
        "rv_5d": _mean(r2, 5),
        "rv_22d": _mean(r2, 22),
        "rv_66d": _mean(r2, 66),
        "rv_252d": _mean(r2, 252),
        "ret_1d": r,
        "ret_5d": r.rolling(5, min_periods=5).sum(),
        "ret_22d": r.rolling(22, min_periods=22).sum(),
        "semivar_down_5d": _mean(down2, 5),
        "semivar_down_22d": _mean(down2, 22),
        "parkinson_1d": parkinson,
        "parkinson_5d": _mean(parkinson, 5),
        "parkinson_22d": _mean(parkinson, 22),
        "overnight_1d": overnight2,
        "overnight_5d": _mean(overnight2, 5),
        "volume_rel_22d": np.log(volume / _mean(volume, 22)),
        "volume_5d_rel_252d": np.log(_mean(volume, 5) / _mean(volume, 252)),
        "drawdown_252d": np.log(
            data["close"] / data["close"].rolling(252, min_periods=252).max()
        ),
    }
    return pd.DataFrame(features, index=data.index)


def inner_split(
    index: pd.DatetimeIndex,
    train_dates: pd.DatetimeIndex,
    validation_fraction: float = INNER_VALIDATION_FRACTION,
    horizon: int = HORIZON,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Chronological split of the training dates into inner-train and
    inner-validation. Validation dates less than ``horizon`` rows after the
    last inner-training date are purged, so no inner-training target shares
    a return with an inner-validation target.
    """
    dates = train_dates.sort_values()
    n_inner = round((1 - validation_fraction) * len(dates))
    inner_train, rest = dates[:n_inner], dates[n_inner:]
    last = index.get_loc(inner_train[-1])
    inner_val = rest[index.get_indexer(rest) >= last + horizon]
    if len(inner_train) == 0 or len(inner_val) == 0:
        raise ValueError("Too few training dates for an inner validation split.")
    return pd.DatetimeIndex(inner_train), pd.DatetimeIndex(inner_val)


def _qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.log(forecast) + realized / forecast))


@dataclass
class FitSummary:
    best_params: dict
    n_trees: int
    n_train: int
    n_inner_train: int
    n_inner_validation: int
    inner_validation_qlike: dict  # grid point -> inner-validation QLIKE (in-training)


class GBMModel:
    """Validation-harness wrapper; all tuning happens inside ``fit``."""

    name = "LightGBM"

    def __init__(self, param_grid: dict | None = None, max_trees: int = MAX_TREES) -> None:
        self.param_grid = PARAM_GRID if param_grid is None else param_grid
        self.max_trees = max_trees
        self.estimator: lgb.LGBMRegressor | None = None
        self.summary: FitSummary | None = None

    def _grid(self) -> list[dict]:
        keys = list(self.param_grid)
        return [dict(zip(keys, values)) for values in itertools.product(*self.param_grid.values())]

    def fit(self, history: pd.DataFrame, train_dates: pd.DatetimeIndex) -> GBMModel:
        features = build_features(history)
        target = forward_realized_variance(history["log_return"], HORIZON)
        X, y = features.loc[train_dates], target.loc[train_dates]
        if y.isna().any():
            raise ValueError("Training dates without a complete target inside `history`.")
        if (y <= 0).any():
            raise ValueError("The gamma objective needs strictly positive targets.")
        if X.isna().any().any():
            raise ValueError("Training dates with incomplete features (inside the burn-in?).")

        inner_train, inner_val = inner_split(history.index, pd.DatetimeIndex(train_dates))
        scores, trees = {}, {}
        for params in self._grid():
            est = lgb.LGBMRegressor(**FIXED_PARAMS, **params, n_estimators=self.max_trees)
            est.fit(
                X.loc[inner_train],
                y.loc[inner_train],
                eval_X=(X.loc[inner_val],),
                eval_y=(y.loc[inner_val],),
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            key = ", ".join(f"{k}={v}" for k, v in params.items())
            pred = est.predict(X.loc[inner_val], num_iteration=est.best_iteration_)
            scores[key] = _qlike(y.loc[inner_val].to_numpy(), pred)
            trees[key] = (params, max(int(est.best_iteration_), 1))

        best_key = min(scores, key=scores.get)
        best_params, n_trees = trees[best_key]
        self.estimator = lgb.LGBMRegressor(**FIXED_PARAMS, **best_params, n_estimators=n_trees)
        self.estimator.fit(X, y)
        self.summary = FitSummary(
            best_params=best_params,
            n_trees=n_trees,
            n_train=len(X),
            n_inner_train=len(inner_train),
            n_inner_validation=len(inner_val),
            inner_validation_qlike=scores,
        )
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """5-day variance forecast; NaN on rows whose features are incomplete."""
        if self.estimator is None:
            raise RuntimeError("Call fit() before predict().")
        features = build_features(data)
        complete = features.notna().all(axis=1)
        out = pd.Series(np.nan, index=data.index, name=f"gbm_var_{HORIZON}d")
        out[complete] = self.estimator.predict(features[complete])
        return out

    def describe(self) -> dict:
        if self.summary is None:
            return {}
        return {
            "best_params": self.summary.best_params,
            "n_trees": self.summary.n_trees,
            "n_train": self.summary.n_train,
            "n_inner_train": self.summary.n_inner_train,
            "n_inner_validation": self.summary.n_inner_validation,
            "grid_points": len(self._grid()),
        }
