"""
Forecast evaluation (see DECISIONS.md ADR-002).

Losses are on the variance scale against the 5-day forward realized
variance:

    QLIKE(RV, F) = log(F) + RV / F     (primary)
    MSE(RV, F)   = (RV - F)^2          (secondary, reported as RMSE)

Both rank forecasts consistently when RV is a noisy but conditionally
unbiased proxy (Patton, 2011).

Commands:

- ``python -m src.evaluate naive-baselines`` records the Phase 2 naive
  random-split numbers for the simple baselines in
  ``docs/phase2_naive_baselines.json``.
- ``python -m src.evaluate validate`` scores all three models under the
  naive random split and under walk-forward validation (Phase 4). It writes
  ``docs/phase4_validation_results.json`` and the per-date forecasts
  (``data/processed/forecasts.csv``), and fails unless the baselines' naive
  scores reproduce the Phase 2 record exactly.

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
from src.complex_model import GBMModel
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
    run_walk_forward,
    walk_forward_folds,
)

NAIVE_BASELINES_PATH = ROOT / "docs" / "phase2_naive_baselines.json"
VALIDATION_RESULTS_PATH = ROOT / "docs" / "phase4_validation_results.json"
FORECASTS_PATH = ROOT / "data" / "processed" / "forecasts.csv"


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


def _models() -> list:
    return [EWMAModel(), GARCHModel(), GBMModel()]


def _date_range(dates: pd.DatetimeIndex) -> dict:
    return {
        "first": dates[0].strftime("%Y-%m-%d"),
        "last": dates[-1].strftime("%Y-%m-%d"),
        "n": len(dates),
    }


def run_validation(
    results_path: Path = VALIDATION_RESULTS_PATH, forecasts_path: Path = FORECASTS_PATH
) -> dict:
    """Score every model under both schemes and write the Phase 4 record."""
    data = load_processed()
    target = forward_realized_variance(data["log_return"], HORIZON)
    dates = forecast_dates(target)
    naive = naive_random_split(dates)
    folds = walk_forward_folds(data.index, dates)

    frames = []
    naive_results, wf_results = {}, {}
    for model in _models():
        forecasts = run_split(model, data, naive)
        naive_results[model.name] = {
            **score(target.loc[forecasts.index], forecasts),
            "parameters": model.describe(),
        }
        frames.append(
            pd.DataFrame({"scheme": "naive_random_split", "model": model.name, "forecast": forecasts})
        )
    for model in _models():
        forecasts, fits = run_walk_forward(model, data, folds)
        wf_results[model.name] = {**score(target.loc[forecasts.index], forecasts), "folds": fits}
        frames.append(
            pd.DataFrame({"scheme": "walk_forward", "model": model.name, "forecast": forecasts})
        )

    # The Phase 2 record must reproduce exactly, or a baseline changed after it was made.
    for name, recorded in json.loads(NAIVE_BASELINES_PATH.read_text())["models"].items():
        now = naive_results[name]
        if any(now[k] != recorded[k] for k in ("qlike", "rmse", "n")):
            raise RuntimeError(f"{name}: naive-split score no longer matches the Phase 2 record.")

    table = pd.concat(frames).rename_axis("date").reset_index()
    table["realized"] = target.reindex(table["date"]).to_numpy()
    forecasts_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(forecasts_path, index=False, lineterminator="\n")

    record = {
        "purpose": (
            "Phase 4: all three models under the naive random split (contrast only) and "
            "under walk-forward validation (the reported scheme). The comparison, "
            "significance tests and verdict are Phase 5 (docs/results_comparison.md)."
        ),
        "generated_by": "python -m src.evaluate validate",
        "processed_file_sha256": json.loads(MANIFEST_PATH.read_text())["processed_file_sha256"],
        "phase2_naive_baselines_reproduced": True,
        "naive_random_split": {
            "test_dates": _date_range(naive.test),
            "estimation_data_end": information_set_end(data.index, naive.train).strftime(
                "%Y-%m-%d"
            ),
            "models": naive_results,
        },
        "walk_forward": {
            "scheme": "expanding window, refit each calendar year, 5-row purge (ADR-003)",
            "n_folds": len(folds),
            "test_dates": _date_range(pd.DatetimeIndex(np.concatenate([f.test for f in folds]))),
            "models": wf_results,
        },
        "forecasts_file": forecasts_path.relative_to(ROOT).as_posix(),
    }
    results_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("naive-baselines", help="record Phase 2 naive-split baseline scores")
    sub.add_parser("validate", help="naive and walk-forward scores for all models (Phase 4)")
    args = parser.parse_args()

    if args.command == "naive-baselines":
        record = record_naive_baselines()
        for name, result in record["models"].items():
            print(f"{name:12s} QLIKE {result['qlike']:.4f}  RMSE {result['rmse']:.3e}  n={result['n']}")
        print(f"-> {NAIVE_BASELINES_PATH.relative_to(ROOT).as_posix()}")
    elif args.command == "validate":
        record = run_validation()
        for scheme in ("naive_random_split", "walk_forward"):
            print(scheme)
            for name, r in record[scheme]["models"].items():
                print(f"  {name:12s} QLIKE {r['qlike']:.4f}  RMSE {r['rmse']:.3e}  n={r['n']}")
        print(f"-> {VALIDATION_RESULTS_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
