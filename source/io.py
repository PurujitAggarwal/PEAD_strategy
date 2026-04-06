"""
io.py
-----
Data loading utilities for universe, price bars, and earnings files.
"""

from pathlib import Path

import pandas as pd


def load_universe(file_path) -> set:
    """
    Loads a ticker universe from an Excel file.
    Expects a column named 'Ticker'; returns a set of normalised ticker strings.
    """
    file_path = Path(file_path)
    df = pd.read_excel(file_path, sheet_name=0)

    if "Ticker" not in df.columns:
        raise ValueError(
            f"'Ticker' column not found. Available columns: {list(df.columns)}"
        )

    tickers = (
        df["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return set(tickers[tickers != ""])


def filter_universe(universe: set, bars: pd.DataFrame, liquidity: float = 2_000_000) -> set:
    """
    Filters the universe to tickers with a median daily dollar volume above
    the liquidity threshold (default $2M/day).
    """
    df = bars[bars["ticker"].isin(universe)].copy()
    if df.empty:
        return set()

    df["date"]       = df["timestamp_utc"].dt.floor("D")
    df["dollar_vol"] = df["close"] * df["volume"]

    daily = df.groupby(["ticker", "date"], as_index=False)["dollar_vol"].sum()
    med   = daily.groupby("ticker")["dollar_vol"].median()

    return set(med[med >= liquidity].index)


def load_earnings(file_path) -> pd.DataFrame:
    """
    Loads earnings data from a CSV file.
    Required columns: ticker, earnings_datetime_utc, eps_actual, eps_opinion.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    required = {"ticker", "earnings_datetime_utc", "eps_actual", "eps_opinion"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in earnings file: {missing}")

    df["ticker"]                 = df["ticker"].astype(str).str.strip().str.upper()
    df["earnings_datetime_utc"]  = pd.to_datetime(df["earnings_datetime_utc"], utc=True)
    df["eps_actual"]             = pd.to_numeric(df["eps_actual"],  errors="coerce")
    df["eps_opinion"]            = pd.to_numeric(df["eps_opinion"], errors="coerce")

    return df.dropna(subset=["ticker", "earnings_datetime_utc"]).reset_index(drop=True)


def load_bars(file_paths) -> pd.DataFrame:
    """
    Loads one or more price bar files (CSV or zstd-compressed CSV).
    Required columns: ticker, timestamp_utc, close.
    Optional column:  volume (defaults to 0.0 if absent).
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    frames = []
    for file_path in file_paths:
        file_path = Path(file_path)
        suffixes  = "".join(file_path.suffixes).lower()

        df = (
            pd.read_csv(file_path, compression="zstd")
            if suffixes.endswith(".csv.zst")
            else pd.read_csv(file_path)
        )
        df.columns = df.columns.str.strip().str.lower()

        required = {"ticker", "timestamp_utc", "close"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Bars file {file_path.name} is missing columns: {missing}"
            )

        df["ticker"]       = df["ticker"].astype(str).str.strip().str.upper()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["close"]        = pd.to_numeric(df["close"],  errors="coerce")
        df["volume"]       = (
            pd.to_numeric(df["volume"], errors="coerce")
            if "volume" in df.columns
            else 0.0
        )

        frames.append(df[["ticker", "timestamp_utc", "close", "volume"]])

    if not frames:
        raise ValueError("No bar files loaded. Check file paths and extensions.")

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.dropna(subset=["ticker", "timestamp_utc", "close"])
    return (
        bars
        .sort_values(["ticker", "timestamp_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
