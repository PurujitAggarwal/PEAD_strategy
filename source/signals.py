"""
signals.py
----------
Converts the features DataFrame into a set of directional trade signals by
applying the configured entry filters.
"""

import pandas as pd
from source.configuration import parameters


def create_signals(features: pd.DataFrame) -> pd.DataFrame:
    """
    Applies EPS surprise, after-hours move, and volume ratio filters to features
    and returns a DataFrame of actionable trade signals with direction assigned.

    Signals are long when EPS surprise is positive, short when negative.
    """
    required = {"ticker", "eps_surprise", "after_hours_move",
                "entry_time_utc", "entry_price", "volume_ratio"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Features missing columns: {missing}. Got: {list(features.columns)}")

    eps_th  = float(parameters.get("eps_surprise_threshold", 0.05))
    move_th = float(parameters.get("ah_move_threshold",      0.02))
    vol_th  = float(parameters.get("volume_ratio_threshold", 0.08))

    df = features.copy()

    # Drop rows with missing values in any required column
    drop_cols = list(required)
    if "risk_on" in df.columns:
        drop_cols.append("risk_on")
    df = df.dropna(subset=drop_cols)

    # Market regime filter (only applied when regime data is present and populated)
    if "risk_on" in df.columns and df["risk_on"].any():
        df = df[df["risk_on"]].copy()

    # Entry filters
    df = df[df["volume_ratio"]          >= vol_th]
    df = df[df["eps_surprise"].abs()    >= eps_th]
    df = df[df["after_hours_move"].abs() >= move_th]

    # Direction and sizing signal
    df["direction"]    = df["eps_surprise"].apply(lambda x: "long" if x > 0 else "short")
    df["abs_surprise"] = df["eps_surprise"].abs()

    return (
        df[[
            "ticker",
            "direction",
            "entry_time_utc",
            "entry_price",
            "eps_surprise",
            "after_hours_move",
            "abs_surprise",
        ]]
        .rename(columns={"after_hours_move": "ah_move"})
        .reset_index(drop=True)
    )
