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

- ``python -m src.evaluate compare`` (Phase 5) reads those forecasts and
  writes ``docs/phase5_comparison.json``. That file holds the
  naive-vs-walk-forward table, Diebold–Mariano tests with Newey–West
  errors, per-year loss differences, and the diagnostics behind
  ``docs/results_comparison.md``. The test design is pre-registered in
  DECISIONS.md ADR-005.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_ewma import EWMAModel
from src.baseline_garch import GARCHModel
from src.complex_model import GBMModel, build_features
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

COMPARISON_PATH = ROOT / "docs" / "phase5_comparison.json"
DM_SENSITIVITY_LAG = 63  # about one quarter of trading days (ADR-005)
MODEL_NAMES = ("EWMA", "GARCH(1,1)", "LightGBM")
# Years whose realized variance exceeded every training target of their fold.
# Used only for post-hoc, descriptive decompositions, never for the verdict.
EXTRAPOLATION_YEARS = (2008, 2020)
VARIANCE_FEATURES = (
    "rv_1d", "rv_5d", "rv_22d", "rv_66d", "semivar_down_5d", "semivar_down_22d",
    "parkinson_1d", "parkinson_5d", "parkinson_22d", "overnight_1d", "overnight_5d",
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


def newey_west_lag(n: int, horizon: int = HORIZON) -> int:
    """Rule-of-thumb HAC lag ``floor(4 (n/100)^(2/9))``, at least ``horizon - 1`` (ADR-005)."""
    return max(horizon - 1, math.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Bartlett-kernel long-run variance of ``x`` (variance of its sum divided by n)."""
    x = np.asarray(x, dtype=float)
    u = x - x.mean()
    n = len(u)
    lrv = u @ u / n
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return float(lrv)


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lag: int) -> dict:
    """
    DM test of equal expected loss on ``d_t = loss_a - loss_b``. Negative mean
    and statistic favor ``a``. Two-sided p-value from the normal distribution.
    """
    if not loss_a.index.equals(loss_b.index):
        raise ValueError("Losses must share the same index.")
    d = (loss_a - loss_b).to_numpy(dtype=float)
    lrv = newey_west_variance(d, lag)
    if lrv <= 0:
        raise ValueError("Loss differential has zero long-run variance.")
    se = math.sqrt(lrv / len(d))
    stat = d.mean() / se
    return {
        "mean_diff": float(d.mean()),
        "se": se,
        "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
        "dm_stat": float(stat),
        "p_value": math.erfc(abs(stat) / math.sqrt(2)),
        "lag": lag,
        "n": len(d),
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


def _load_forecasts(path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.evaluate validate` first.")
    table = pd.read_csv(path, parse_dates=["date"])
    return {
        key: group.set_index("date")[["forecast", "realized"]].sort_index()
        for key, group in table.groupby(["scheme", "model"])
    }


def _dm_block(frames: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> dict:
    """DM tests for every model pair, both metrics, primary and sensitivity lag."""
    lags = {"primary": newey_west_lag(len(dates)), "sensitivity": DM_SENSITIVITY_LAG}
    losses = {
        metric: {
            m: fn(f.loc[dates, "realized"], f.loc[dates, "forecast"]) for m, f in frames.items()
        }
        for metric, fn in (("qlike", qlike_losses), ("mse", squared_errors))
    }
    pairs = [("LightGBM", "GARCH(1,1)"), ("LightGBM", "EWMA"), ("GARCH(1,1)", "EWMA")]
    return {
        metric: {
            f"{a} vs {b}": {
                name: diebold_mariano(by_model[a], by_model[b], lag) for name, lag in lags.items()
            }
            for a, b in pairs
        }
        for metric, by_model in losses.items()
    }


def _per_year(frames: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for year, idx in frames["GARCH(1,1)"].groupby(frames["GARCH(1,1)"].index.year).groups.items():
        row = {"n": len(idx)}
        for m, f in frames.items():
            sub = f.loc[idx]
            row[m] = score(sub["realized"], sub["forecast"])
        row["qlike_diff_lgbm_minus_garch"] = row["LightGBM"]["qlike"] - row["GARCH(1,1)"]["qlike"]
        row["mse_diff_lgbm_minus_garch"] = (
            row["LightGBM"]["rmse"] ** 2 - row["GARCH(1,1)"]["rmse"] ** 2
        )
        row["max_realized"] = float(frames["GARCH(1,1)"].loc[idx, "realized"].max())
        out[int(year)] = row
    return out


def _extrapolation(frames: dict[str, pd.DataFrame], data: pd.DataFrame, target: pd.Series) -> dict:
    """Per fold: the largest training target versus the test year's realized and forecast extremes."""
    folds = walk_forward_folds(data.index, forecast_dates(target))
    rows = []
    for fold in folds:
        cap = float(target.loc[fold.train].max())
        realized = target.loc[fold.test]
        row = {
            "year": int(fold.test[0].year),
            "max_training_rv": cap,
            "max_realized_rv": float(realized.max()),
            "n_realized_above_training_max": int((realized > cap).sum()),
        }
        for m, f in frames.items():
            row[f"max_forecast_{m}"] = float(f.loc[fold.test, "forecast"].max())
        row["n_lightgbm_forecasts_above_training_max"] = int(
            (frames["LightGBM"].loc[fold.test, "forecast"] > cap).sum()
        )
        rows.append(row)
    return {"folds": rows}


def _contributions(diff: pd.Series) -> dict:
    """Split a mean loss difference into per-year contributions (they sum to the mean)."""
    by_year = diff.groupby(diff.index.year).sum() / len(diff)
    excluded = diff[~diff.index.year.isin(EXTRAPOLATION_YEARS)]
    return {
        "mean": float(diff.mean()),
        "by_year": {int(y): float(v) for y, v in by_year.items()},
        "posthoc_mean_excluding_extrapolation_years": float(excluded.mean()),
        "extrapolation_years": list(EXTRAPOLATION_YEARS),
    }


def _ceiling_probe(data: pd.DataFrame, target: pd.Series, year: int = 2008) -> dict:
    """
    Refit the walk-forward LightGBM of one fold and probe how its forecast
    responds when every variance feature on the worst realized week is scaled up.
    """
    fold = next(f for f in walk_forward_folds(data.index, forecast_dates(target))
                if f.test[0].year == year)
    history = data.loc[: information_set_end(data.index, fold.train)]
    model = GBMModel().fit(history, fold.train)
    features = build_features(data)
    worst = target.loc[fold.test].idxmax()
    row = features.loc[[worst]]
    scales = [1, 2, 10, 100]
    probes = pd.concat(
        [row.assign(**{c: row[c] * k for c in VARIANCE_FEATURES}) for k in scales]
    )
    exceedance = {
        c: {
            "training_max": float(features.loc[fold.train, c].max()),
            "test_max": float(features.loc[fold.test, c].max()),
            "n_test_dates_above_training_max": int(
                (features.loc[fold.test, c] > features.loc[fold.train, c].max()).sum()
            ),
        }
        for c in ("rv_5d", "rv_22d", "parkinson_5d")
    }
    return {
        "fold_year": year,
        "worst_week_date": worst.strftime("%Y-%m-%d"),
        "worst_week_realized": float(target.loc[worst]),
        "forecast_with_variance_features_scaled": dict(
            zip([f"x{k}" for k in scales], map(float, model.estimator.predict(probes)))
        ),
        "max_forecast_over_full_sample": float(model.predict(data).max()),
        "max_training_target": float(target.loc[fold.train].max()),
        "feature_exceedance": exceedance,
    }


def _error_concentration(frames: dict[str, pd.DataFrame], top_fraction: float = 0.01) -> dict:
    """Share of each model's total squared error and QLIKE excess coming from the top RV weeks."""
    realized = frames["GARCH(1,1)"]["realized"]
    top = realized >= realized.quantile(1 - top_fraction)
    out = {"top_fraction": top_fraction, "n_top": int(top.sum()), "models": {}}
    for m, f in frames.items():
        se = squared_errors(f["realized"], f["forecast"])
        ql = qlike_losses(f["realized"], f["forecast"])
        out["models"][m] = {
            "share_of_squared_error_in_top": float(se[top].sum() / se.sum()),
            "rmse_excluding_top": float(np.sqrt(se[~top].mean())),
            "qlike_excluding_top": float(ql[~top].mean()),
            "qlike_top_only": float(ql[top].mean()),
            "mean_forecast_over_mean_realized": float(f["forecast"].mean() / f["realized"].mean()),
            "median_forecast_over_median_realized": float(
                f["forecast"].median() / f["realized"].median()
            ),
        }
    return out


def run_comparison(forecasts_path: Path = FORECASTS_PATH, out_path: Path = COMPARISON_PATH) -> dict:
    """Phase 5: comparison table, DM tests and diagnostics from the Phase 4 forecasts."""
    fc = _load_forecasts(forecasts_path)
    naive = {m: fc[("naive_random_split", m)] for m in MODEL_NAMES}
    wf = {m: fc[("walk_forward", m)] for m in MODEL_NAMES}
    wf_dates = wf["GARCH(1,1)"].index
    naive_dates = naive["GARCH(1,1)"].index
    common = naive_dates[naive_dates >= wf_dates.min()]
    if not common.isin(wf_dates).all():
        raise ValueError("Naive test dates from the walk-forward start must all be walk-forward dates.")

    def _score(f: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
        return score(f.loc[dates, "realized"], f.loc[dates, "forecast"])

    table = {
        m: {
            "naive_all": _score(naive[m], naive_dates),
            "naive_same_dates": _score(naive[m], common),
            "walk_forward_same_dates": _score(wf[m], common),
            "walk_forward_all": _score(wf[m], wf_dates),
        }
        for m in MODEL_NAMES
    }
    scheme_effect = {}
    for m in MODEL_NAMES:
        if np.array_equal(naive[m].loc[common, "forecast"], wf[m].loc[common, "forecast"]):
            # EWMA estimates nothing, so both schemes produce the same forecast.
            scheme_effect[m] = {"identical_forecasts": True}
            continue
        scheme_effect[m] = {
            metric: diebold_mariano(
                fn(naive[m].loc[common, "realized"], naive[m].loc[common, "forecast"]),
                fn(wf[m].loc[common, "realized"], wf[m].loc[common, "forecast"]),
                newey_west_lag(len(common)),
            )
            for metric, fn in (("qlike", qlike_losses), ("mse", squared_errors))
        }

    def _qlike_diff(a: pd.DataFrame, b: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        return qlike_losses(a.loc[dates, "realized"], a.loc[dates, "forecast"]) - qlike_losses(
            b.loc[dates, "realized"], b.loc[dates, "forecast"]
        )

    crisis = pd.DatetimeIndex(common[common.year.isin(EXTRAPOLATION_YEARS)])
    naive_in_crisis = {
        "dates": _date_range(crisis),
        "mean_realized": float(wf["LightGBM"].loc[crisis, "realized"].mean()),
        "mean_forecast": {
            "naive LightGBM": float(naive["LightGBM"].loc[crisis, "forecast"].mean()),
            "walk-forward LightGBM": float(wf["LightGBM"].loc[crisis, "forecast"].mean()),
            "naive GARCH(1,1)": float(naive["GARCH(1,1)"].loc[crisis, "forecast"].mean()),
            "walk-forward GARCH(1,1)": float(wf["GARCH(1,1)"].loc[crisis, "forecast"].mean()),
        },
    }

    data = load_processed()
    target = forward_realized_variance(data["log_return"], HORIZON)
    record = {
        "purpose": "Phase 5 comparison; test design pre-registered in DECISIONS.md ADR-005.",
        "generated_by": "python -m src.evaluate compare",
        "source": forecasts_path.relative_to(ROOT).as_posix(),
        "dates": {
            "naive_all": _date_range(naive_dates),
            "same_dates": _date_range(common),
            "walk_forward_all": _date_range(wf_dates),
        },
        "table": table,
        "scheme_effect_naive_minus_walk_forward_same_dates": scheme_effect,
        "diebold_mariano": {
            "walk_forward": _dm_block(wf, wf_dates),
            "naive_random_split": _dm_block(naive, naive_dates),
        },
        "walk_forward_per_year": _per_year(wf),
        "walk_forward_extrapolation": _extrapolation(wf, data, target),
        "walk_forward_error_concentration": _error_concentration(wf),
        "walk_forward_qlike_diff_lgbm_minus_garch": _contributions(
            _qlike_diff(wf["LightGBM"], wf["GARCH(1,1)"], wf_dates)
        ),
        "scheme_effect_qlike_contributions": {
            m: _contributions(_qlike_diff(naive[m], wf[m], common))
            for m in ("GARCH(1,1)", "LightGBM")
        },
        "naive_test_dates_in_extrapolation_years": naive_in_crisis,
        "lightgbm_ceiling_probe": _ceiling_probe(data, target),
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _print_comparison(record: dict) -> None:
    print("| Model | Naive (all) | Naive (2000+) | Walk-forward (same dates) | Walk-forward (all) |")
    print("|---|---|---|---|---|")
    for metric, fmt in (("qlike", "{:.4f}"), ("rmse", "{:.3e}")):
        for m, row in record["table"].items():
            cells = [fmt.format(row[k][metric]) for k in row]
            print(f"| {m} {metric.upper()} | " + " | ".join(cells) + " |")
    for scheme, block in record["diebold_mariano"].items():
        print(f"\nDM tests, {scheme} (negative favors the first model)")
        for metric, pairs in block.items():
            for pair, res in pairs.items():
                p, s = res["primary"], res["sensitivity"]
                print(
                    f"  {metric:5s} {pair:26s} diff {p['mean_diff']:+.3e}  "
                    f"DM {p['dm_stat']:+.2f} p={p['p_value']:.4f} (lag {p['lag']}) | "
                    f"DM {s['dm_stat']:+.2f} p={s['p_value']:.4f} (lag {s['lag']})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("naive-baselines", help="record Phase 2 naive-split baseline scores")
    sub.add_parser("validate", help="naive and walk-forward scores for all models (Phase 4)")
    sub.add_parser("compare", help="comparison table, DM tests and diagnostics (Phase 5)")
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
    elif args.command == "compare":
        record = run_comparison()
        _print_comparison(record)
        print(f"-> {COMPARISON_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
