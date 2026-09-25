"""
Forecast evaluation (see DECISIONS.md ADR-002).

Losses are on the variance scale against the 5-day forward realized
variance:

    QLIKE(RV, F) = log(F) + RV / F     (primary)
    MSE(RV, F)   = (RV - F)^2          (secondary, reported as RMSE)

Both rank forecasts consistently when RV is a noisy but conditionally
unbiased proxy (Patton, 2011).

Run: ``python -m src.evaluate naive-baselines`` records the Phase 2 naive
random-split numbers for the simple baselines in
``docs/phase2_naive_baselines.json``.

TODO (TASKS.md Phase 5): the naive-vs-walk-forward comparison table for
all models (PROJECT.md section 9) and the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_ewma import EWMAModel
from src.baseline_garch import GARCHModel
from src.data import MANIFEST_PATH, ROOT, load_processed
from src.target import HORIZON, forward_realized_variance
from src.validation import (
    BURN_IN,
    NAIVE_SEED,
    NAIVE_TEST_FRACTION,
    forecast_dates,
    information_set_end,
    naive_random_split,
    run_split,
)

NAIVE_BASELINES_PATH = ROOT / "docs" / "phase2_naive_baselines.json"


def _aligned(realized: pd.Series, forecast: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not realized.index.equals(forecast.index):
        raise ValueError("realized and forecast must share the same index.")
    rv, f = realized.to_numpy(dtype=float), forecast.to_numpy(dtype=float)
    if np.isnan(rv).any() or np.isnan(f).any():
        raise ValueError("NaN in realized or forecast values.")
    if (f <= 0).any():
        raise ValueError("Forecasts must be strictly positive for QLIKE.")
    return rv, f


def qlike_losses(realized: pd.Series, forecast: pd.Series) -> pd.Series:
    rv, f = _aligned(realized, forecast)
    return pd.Series(np.log(f) + rv / f, index=forecast.index, name="qlike")


def squared_errors(realized: pd.Series, forecast: pd.Series) -> pd.Series:
    rv, f = _aligned(realized, forecast)
    return pd.Series((rv - f) ** 2, index=forecast.index, name="squared_error")


def score(realized: pd.Series, forecast: pd.Series) -> dict:
    return {
        "qlike": float(qlike_losses(realized, forecast).mean()),
        "rmse": float(np.sqrt(squared_errors(realized, forecast).mean())),
        "n": len(forecast),
    }


def record_naive_baselines(path: Path = NAIVE_BASELINES_PATH) -> dict:
    """Score EWMA and GARCH(1,1) under the naive random split and write the record."""
    data = load_processed()
    target = forward_realized_variance(data["log_return"], HORIZON)
    dates = forecast_dates(target)
    split = naive_random_split(dates)
    realized = target.loc[split.test]

    models = {}
    for model in (EWMAModel(), GARCHModel()):
        forecasts = run_split(model, data, split)
        models[model.name] = {**score(realized, forecasts), "parameters": model.describe()}

    manifest = json.loads(MANIFEST_PATH.read_text())
    record = {
        "purpose": (
            "Phase 2 record of the simple baselines under the NAIVE random split, made "
            "before the complex model exists. Contrast only, not the project's result "
            "(PROJECT.md section 9, DECISIONS.md ADR-008)."
        ),
        "generated_by": "python -m src.evaluate naive-baselines",
        "processed_file_sha256": manifest["processed_file_sha256"],
        "target": {"name": target.name, "horizon_days": HORIZON, "units": "5-day variance"},
        "split": {
            "scheme": "random split of forecast dates, time order ignored",
            "test_fraction": NAIVE_TEST_FRACTION,
            "seed": NAIVE_SEED,
            "burn_in_rows": BURN_IN,
            "eligible_dates": {
                "first": dates[0].strftime("%Y-%m-%d"),
                "last": dates[-1].strftime("%Y-%m-%d"),
                "n": len(dates),
            },
            "n_train": len(split.train),
            "n_test": len(split.test),
            "estimation_data_end": information_set_end(data.index, split.train).strftime(
                "%Y-%m-%d"
            ),
        },
        "metrics": {
            "qlike": "mean of log(F) + RV/F; lower is better",
            "rmse": "root mean squared error of 5-day variance; lower is better",
        },
        "models": models,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("naive-baselines", help="record Phase 2 naive-split baseline scores")
    args = parser.parse_args()

    if args.command == "naive-baselines":
        record = record_naive_baselines()
        for name, result in record["models"].items():
            print(f"{name:12s} QLIKE {result['qlike']:.4f}  RMSE {result['rmse']:.3e}  n={result['n']}")
        print(f"-> {NAIVE_BASELINES_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
