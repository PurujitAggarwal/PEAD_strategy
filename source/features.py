"""
features.py
-----------
Builds the feature DataFrame used for signal generation.

For each earnings event the following are computed:
  - EPS surprise (actual vs consensus estimate)
  - After-hours price move (entry price vs last regular-session close)
  - Volume ratio (post-earnings vs median regular-session bar volume)
  - Market regime flag (SPY above/below rolling SMA)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from source.configuration import parameters
from source.sessions import is_regular_session


# ── EPS surprise ──────────────────────────────────────────────────────────────

def calculate_eps_surprise(eps_actual, eps_opinion) -> Optional[float]:
    """Returns (actual - estimate) / |estimate|, or None if inputs are invalid."""
    if eps_opinion is None or pd.isna(eps_opinion) or eps_opinion == 0:
        return None
    if eps_actual is None or pd.isna(eps_actual):
        return None
    return (float(eps_actual) - float(eps_opinion)) / abs(float(eps_opinion))


# ── Bar data container ────────────────────────────────────────────────────────

@dataclass
class TickerBars:
    t:            np.ndarray   # int64 nanoseconds since epoch
    close:        np.ndarray   # float64 close prices
    vol:          np.ndarray   # float64 bar volumes
    regular_mask: np.ndarray   # bool — True during regular session (14:30–21:00 UTC)


def _prep_bars_by_ticker(bars: pd.DataFrame) -> Dict[str, TickerBars]:
    reg  = is_regular_session(bars["timestamp_utc"])
    bars = bars.copy()
    bars["regular"] = reg.values

    out: Dict[str, TickerBars] = {}
    for ticker, df in bars.groupby("ticker", sort=False):
        ts = df["timestamp_utc"].astype("int64").to_numpy()
        out[ticker] = TickerBars(
            t            = ts,
            close        = df["close"].to_numpy(dtype="float64"),
            vol          = df["volume"].to_numpy(dtype="float64"),
            regular_mask = df["regular"].to_numpy(dtype=bool),
        )
    return out


# ── Vectorised bar lookups ────────────────────────────────────────────────────

def _last_regular_close(tb: TickerBars, ert_ns: int) -> Optional[float]:
    """Returns the most recent regular-session close price strictly before ert_ns."""
    idx = np.searchsorted(tb.t, ert_ns, side="left") - 1
    while idx >= 0 and not tb.regular_mask[idx]:
        idx -= 1
    return float(tb.close[idx]) if idx >= 0 else None


def _first_bar_at_or_after(tb: TickerBars, t_ns: int) -> Optional[int]:
    """Returns the index of the first bar at or after t_ns, or None if beyond end."""
    j = np.searchsorted(tb.t, t_ns, side="left")
    return int(j) if j < len(tb.t) else None


def _sum_volume_between(tb: TickerBars, start_ns: int, end_ns: int) -> Tuple[float, int]:
    """Returns (total_volume, bar_count) in the half-open interval [start_ns, end_ns)."""
    a = np.searchsorted(tb.t, start_ns, side="left")
    b = np.searchsorted(tb.t, end_ns,   side="right")
    if b <= a:
        return 0.0, 0
    return float(tb.vol[a:b].sum()), int(b - a)


# ── Market regime ─────────────────────────────────────────────────────────────

def _build_market_regime(
    bars:         pd.DataFrame,
    benchmark:    str = "SPY",
    window_days:  int = 100,
) -> pd.DataFrame:
    """
    Constructs a daily risk-on / risk-off flag.
    risk_on = True when the benchmark daily close is above its rolling SMA.
    Returns a DataFrame with columns [date, risk_on].
    """
    b = bars[bars["ticker"] == benchmark].copy()
    if b.empty:
        return pd.DataFrame(columns=["date", "risk_on"])

    b = b.sort_values("timestamp_utc")
    b["date"] = b["timestamp_utc"].dt.floor("D")

    daily = b.groupby("date", as_index=False)["close"].last()
    daily["sma"]    = daily["close"].rolling(window_days, min_periods=window_days).mean()
    # Default to risk_on=True while SMA is warming up to avoid discarding early history
    daily["risk_on"] = np.where(daily["sma"].isna(), True, daily["close"] > daily["sma"])

    return daily[["date", "risk_on"]]


# ── Main feature builder ──────────────────────────────────────────────────────

def create_features(
    earnings: pd.DataFrame,
    bars:     pd.DataFrame,
    universe: set,
) -> pd.DataFrame:
    """
    Joins earnings events with price and volume data to produce one row per event
    with all features needed by create_signals().
    """
    delay = int(parameters.get("entry_delay", 0))

    bars_by_ticker = _prep_bars_by_ticker(bars)

    # Precompute per-ticker median regular-session bar volume once
    med_reg_vol: Dict[str, float] = {}
    for tkr, tb in bars_by_ticker.items():
        reg_vol = tb.vol[tb.regular_mask]
        med_reg_vol[tkr] = float(np.nanmedian(reg_vol)) if reg_vol.size else np.nan

    # Build market regime lookup {date -> bool}
    benchmark = str(parameters.get("benchmark_ticker", "SPY")).upper()
    ma_days   = int(parameters.get("market_ma_days", 100))
    regime_df = _build_market_regime(bars, benchmark=benchmark, window_days=ma_days)
    regime_map = dict(zip(regime_df["date"], regime_df["risk_on"])) if not regime_df.empty else None

    rows = []

    for er in earnings.itertuples(index=False):
        ticker = str(er.ticker).strip().upper()
        if ticker not in universe:
            continue

        tb = bars_by_ticker.get(ticker)
        if tb is None:
            continue

        ert    = pd.Timestamp(er.earnings_datetime_utc)
        ert_ns = int(ert.value)

        last_close = _last_regular_close(tb, ert_ns)

        entry_ns   = int((ert + pd.Timedelta(minutes=delay)).value)
        j          = _first_bar_at_or_after(tb, entry_ns)
        entry_time = pd.to_datetime(tb.t[j], utc=True) if j is not None else None
        entry_price = float(tb.close[j])               if j is not None else None

        # Volume ratio: post-earnings volume vs expected based on median bar volume
        vol_ratio = None
        if j is not None:
            after_vol, n_bars = _sum_volume_between(tb, ert_ns, int(tb.t[j]))
            med = med_reg_vol.get(ticker, np.nan)
            if n_bars > 0 and not np.isnan(med) and med > 0:
                expected = med * n_bars
                if expected > 0:
                    vol_ratio = float(after_vol / expected)

        surprise = calculate_eps_surprise(er.eps_actual, er.eps_opinion)

        ah_move = None
        if last_close is not None and entry_price is not None:
            ah_move = (entry_price / last_close) - 1.0

        risk_on = bool(regime_map.get(ert.floor("D"), True)) if regime_map is not None else True

        rows.append({
            "ticker":               ticker,
            "earnings_datetime_utc": ert,
            "eps_actual":           er.eps_actual,
            "eps_opinion":          er.eps_opinion,
            "eps_surprise":         surprise,
            "last_regular_close":   last_close,
            "entry_time_utc":       entry_time,
            "entry_price":          entry_price,
            "after_hours_move":     ah_move,
            "volume_ratio":         vol_ratio,
            "risk_on":              risk_on,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "earnings_datetime_utc", "eps_actual", "eps_opinion",
            "eps_surprise", "last_regular_close", "entry_time_utc",
            "entry_price", "after_hours_move", "volume_ratio", "risk_on",
        ])
    return df
