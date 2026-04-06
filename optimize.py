"""
optimize.py
-----------
Grid search over EPS surprise threshold, stop-loss, and volume ratio threshold.

Usage:
    python optimize.py
"""

from pathlib import Path

import pandas as pd

from source.configuration import parameters
from source.io import load_earnings, load_bars, load_universe, filter_universe
from source.features import create_features
from source.signals import create_signals
from source.backtest import backtest, build_bars_map
from source.metrics import produce_results


def run_grid_search():
    data_dir = Path(__file__).resolve().parent / "data"

    universe_path = data_dir / "universe.xlsx"
    bars_paths    = sorted(data_dir.glob("bars*.csv*"))
    earnings_path = data_dir / "earnings.csv"

    # --- Load data once (expensive) ---
    universe = load_universe(universe_path)
    raw_bars = load_bars(bars_paths)
    earnings = load_earnings(earnings_path)

    active_universe = filter_universe(universe, raw_bars)
    bars_map        = build_bars_map(raw_bars)

    print("Pre-calculating features ...")
    base_features = create_features(earnings, raw_bars, active_universe)

    # --- Search space ---
    surprises  = [0.03, 0.05, 0.08, 0.10]
    stops      = [0.01, 0.02, 0.03, 0.05, None]
    vol_ratios = [0.05, 0.10, 0.15]

    total = len(surprises) * len(stops) * len(vol_ratios)
    print(f"Starting grid search: {total} combinations ...")

    results_grid = []

    for s_th in surprises:
        for sl in stops:
            for v_th in vol_ratios:
                parameters["eps_surprise_threshold"] = s_th
                parameters["stop_loss"]              = sl
                parameters["volume_ratio_threshold"] = v_th

                sigs    = create_signals(base_features)
                trades  = backtest(sigs, bars_map)
                metrics = produce_results(trades)

                results_grid.append({
                    "eps_th":    s_th,
                    "stop_loss": sl,
                    "vol_th":    v_th,
                    "sharpe":    metrics["sharpe_ratio"],
                    "trades":    metrics["number_of_trades"],
                    "win_rate":  metrics["win_rate"],
                    "max_dd":    metrics["max_drawdown"],
                })

    # --- Results ---
    df = pd.DataFrame(results_grid)
    df = df[df["trades"] > 10]  # discard noise from very sparse parameter sets

    if df.empty:
        print("No combinations produced more than 10 trades.")
        return

    best = df.sort_values("sharpe", ascending=False).iloc[0]
    print("\n--- BEST SHARPE FOUND ---")
    print(best.to_string())

    print("\n--- TOP 10 BY SHARPE ---")
    print(df.sort_values("sharpe", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    run_grid_search()
