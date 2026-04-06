import numpy as np
import pandas as pd
from source.configuration import parameters


def build_bars_map(bars: pd.DataFrame) -> dict:
    """
    Pre-index bar data by ticker for O(log n) timestamp lookups during backtesting.
    Returns {ticker: (timestamps_ns_array, close_array)}.
    """
    bars_sorted = (
        bars
        .sort_values(["ticker", "timestamp_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    out = {}
    for tkr, df in bars_sorted.groupby("ticker", sort=False):
        t = df["timestamp_utc"].astype("int64").to_numpy()
        close = df["close"].to_numpy(dtype="float64")
        out[tkr] = (t, close)
    return out


def backtest(signals: pd.DataFrame, bars_map: dict) -> pd.DataFrame:
    """
    Event-driven backtest over a set of entry signals.

    For each signal the full price path within the holding window is computed
    vectorially, then exit logic (stop-loss, take-profit, trailing stop, time)
    is applied.  Both raw returns and volatility-scaled returns are recorded.

    Returns a DataFrame with one row per trade.
    """
    # ── Parameters ────────────────────────────────────────────────
    max_holding     = int(parameters["max_holding"])
    stop_loss       = parameters.get("stop_loss", None)
    take_profit     = parameters.get("take_profit", None)
    use_trailing    = bool(parameters.get("use_trailing_stop", True))
    trail_activation = float(parameters.get("trail_activation", 0.03))
    trail_amount     = float(parameters.get("trail_amount", 0.02))
    target_risk      = float(parameters.get("target_risk", 0.01))

    results = []
    signals_sorted = signals.sort_values("entry_time_utc").reset_index(drop=True)

    for s in signals_sorted.itertuples(index=False):
        ticker      = s.ticker
        direction   = s.direction
        entry_time  = pd.to_datetime(s.entry_time_utc, utc=True)
        entry_ns    = int(entry_time.value)
        entry_price = float(s.entry_price)

        series = bars_map.get(ticker)
        if series is None:
            continue

        t_arr, p_arr = series
        end_ns = int((entry_time + pd.Timedelta(minutes=max_holding)).value)

        a = np.searchsorted(t_arr, entry_ns, side="left")
        b = np.searchsorted(t_arr, end_ns,   side="right")
        if b <= a:
            continue

        t_win = t_arr[a:b]
        p_win = p_arr[a:b]

        # ── Realised volatility of the holding window (for scaling) ──
        if len(p_win) > 2:
            vol = np.std(np.diff(np.log(p_win)))
        else:
            vol = 0.02  # fallback for very thin windows

        exit_idx = len(p_win) - 1
        reason   = "time"

        # ── Return path and running drawdown from peak ────────────
        if direction == "long":
            rets_path = (p_win / entry_price) - 1.0
            peaks     = np.maximum.accumulate(p_win)
            dd_from_peak = (peaks - p_win) / peaks
        else:
            rets_path = (entry_price / p_win) - 1.0
            peaks     = np.minimum.accumulate(p_win)
            dd_from_peak = (p_win - peaks) / peaks

        # ── Exit conditions ───────────────────────────────────────
        exits = []

        if stop_loss is not None:
            sl_hits = np.where(rets_path <= -float(stop_loss))[0]
            if sl_hits.size:
                exits.append((int(sl_hits[0]), "stop_loss"))

        if take_profit is not None:
            tp_hits = np.where(rets_path >= float(take_profit))[0]
            if tp_hits.size:
                exits.append((int(tp_hits[0]), "take_profit"))

        if use_trailing:
            activated = rets_path >= trail_activation
            if np.any(activated):
                trail_hits = np.where(activated & (dd_from_peak >= trail_amount))[0]
                if trail_hits.size:
                    exits.append((int(trail_hits[0]), "trailing_stop"))

        if exits:
            exits.sort(key=lambda x: x[0])
            exit_idx, reason = exits[0]

        exit_price = float(p_win[exit_idx])
        exit_time  = pd.to_datetime(t_win[exit_idx], utc=True)

        # ── Raw return ────────────────────────────────────────────
        if direction == "long":
            raw_return = (exit_price / entry_price) - 1.0
        else:
            raw_return = (entry_price / exit_price) - 1.0

        # ── Volatility-scaled return ──────────────────────────────
        # Scales each trade so that one unit of holding-period vol corresponds
        # to target_risk, making returns comparable across high- and low-vol names.
        vol_scalar    = target_risk / max(vol, 0.0001)
        scaled_return = raw_return * vol_scalar

        results.append({
            "ticker":         ticker,
            "direction":      direction,
            "entry_time_utc": entry_time,
            "entry_price":    entry_price,
            "exit_time_utc":  exit_time,
            "exit_price":     exit_price,
            "exit_reason":    reason,
            "return":         float(raw_return),
            "scaled_return":  float(scaled_return),
        })

    return pd.DataFrame(results)
