"""
Daily price data for the study instrument (see DECISIONS.md ADR-001).

Pipeline: download once from yfinance -> cache the raw frame in data/raw/ ->
clean -> add daily log returns -> write data/processed/ and the dataset
manifest. Later runs reuse the raw cache so the sample does not silently
change between runs; pass ``--refresh`` to re-download.

Timing convention used throughout the project: row ``t`` is the close of
trading day ``t``, and ``log_return`` at ``t`` is ``ln(P_t / P_{t-1})``,
which is known at the close of ``t``.

Run: ``python -m src.data [--refresh]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

TICKER = "SPY"
START = "1993-01-29"  # SPY's first trading day
END = "2025-12-31"  # inclusive; fixed so the sample does not drift between runs

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / f"{TICKER}_daily_raw.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / f"{TICKER}_daily.csv"
MANIFEST_PATH = ROOT / "data" / "dataset_manifest.json"

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class CleaningReport:
    rows_in: int
    duplicate_dates_dropped: int
    missing_close_dropped: int
    rows_out: int


def download_prices(ticker: str = TICKER, start: str = START, end: str = END) -> pd.DataFrame:
    """Download split- and dividend-adjusted daily OHLCV from yfinance."""
    import yfinance as yf

    # yfinance treats `end` as exclusive; add a day so END itself is included.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        actions=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} ({start} to {end}).")
    return normalize_yfinance_frame(raw)


def normalize_yfinance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's (field, ticker) column MultiIndex and lower-case the field names."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).lower() for c in frame.columns]
    frame.index.name = "date"
    return frame


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Validate and clean a daily OHLCV frame.

    - Index is converted to tz-naive dates and sorted ascending.
    - Duplicate dates keep the last record.
    - Rows with a missing close are dropped, never forward-filled: a filled
      price would create an artificial zero return and bias volatility down.
    - A non-positive close raises, since log returns would be undefined.
    """
    missing = [c for c in PRICE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Price frame is missing columns: {missing}")
    if raw.empty:
        raise ValueError("Price frame is empty.")

    frame = raw[PRICE_COLUMNS].copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = "date"
    # Stable sort: among duplicate dates, original order decides which record is "last".
    frame = frame.sort_index(kind="stable")
    rows_in = len(frame)

    duplicated = frame.index.duplicated(keep="last")
    frame = frame[~duplicated]

    missing_close = frame["close"].isna()
    frame = frame[~missing_close]

    if (frame["close"] <= 0).any():
        bad = frame.index[frame["close"] <= 0][:5].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Non-positive close prices found, e.g. on {bad}")

    report = CleaningReport(
        rows_in=rows_in,
        duplicate_dates_dropped=int(duplicated.sum()),
        missing_close_dropped=int(missing_close.sum()),
        rows_out=len(frame),
    )
    return frame, report


def add_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add ``log_return = ln(close_t / close_{t-1})`` and drop the first row, which has none."""
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be sorted by date before computing returns.")
    out = prices.copy()
    out["log_return"] = np.log(out["close"]).diff()
    return out.iloc[1:]


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed dataset written by ``build_dataset``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def build_dataset(refresh: bool = False) -> pd.DataFrame:
    """Download (or reuse the cached raw file), clean, add returns, and write outputs + manifest."""
    previous = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    retrieved_at = previous.get("retrieved_at_utc")

    if refresh or not RAW_PATH.exists():
        raw = download_prices()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(RAW_PATH)
        retrieved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        raw = pd.read_csv(RAW_PATH, index_col="date", parse_dates=["date"])

    prices, report = clean_prices(raw)
    dataset = add_log_returns(prices)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Fixed line terminator so the manifest's file hash is the same on every OS.
    dataset.to_csv(PROCESSED_PATH, lineterminator="\n")

    _write_manifest(dataset, report, retrieved_at)
    return dataset


def _write_manifest(dataset: pd.DataFrame, report: CleaningReport, retrieved_at: str | None) -> None:
    import yfinance as yf

    r = dataset["log_return"]
    manifest = {
        "source": "yfinance",
        "yfinance_version": yf.__version__,
        "instrument": TICKER,
        "instrument_description": "SPDR S&P 500 ETF Trust (NYSE Arca)",
        "frequency": "daily (close-to-close)",
        "price_adjustment": "auto_adjust=True: OHLC adjusted for splits and dividends",
        "requested_date_range": {"start": START, "end_inclusive": END},
        "date_range": {
            "first_return": dataset.index[0].strftime("%Y-%m-%d"),
            "last_return": dataset.index[-1].strftime("%Y-%m-%d"),
        },
        "n_return_rows": len(dataset),
        "cleaning": asdict(report),
        "retrieved_at_utc": retrieved_at,
        # Adjusted price *levels* shift whenever a new dividend is paid, but
        # log returns do not; these let a re-download be checked for drift.
        "log_return_checks": {
            "sum": float(f"{r.sum():.10g}"),
            "sum_of_squares": float(f"{(r**2).sum():.10g}"),
        },
        "processed_file_sha256": hashlib.sha256(PROCESSED_PATH.read_bytes()).hexdigest(),
        "files": {
            "raw": RAW_PATH.relative_to(ROOT).as_posix(),
            "processed": PROCESSED_PATH.relative_to(ROOT).as_posix(),
        },
        "rationale": "See DECISIONS.md ADR-001.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-download from yfinance")
    args = parser.parse_args()
    dataset = build_dataset(refresh=args.refresh)
    print(
        f"{TICKER}: {len(dataset)} daily returns, "
        f"{dataset.index[0].date()} to {dataset.index[-1].date()} -> {PROCESSED_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
