"""
main.py
-------
Entry point for running a single backtest or parameter sweep.

Usage:
    python main.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from source.configuration import parameters
from source.features import create_features
from source.metrics import produce_results
from source.io import load_universe, load_bars, load_earnings, filter_universe
from source.signals import create_signals
from source.backtest import backtest, build_bars_map


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "data"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


# ── Caching helper ────────────────────────────────────────────────────────────
def load_or_build(path: Path, builder, fmt: str = "parquet", force: bool = False):
    """Loads a cached DataFrame if it exists, otherwise builds and caches it."""
    if not force and path.exists():
        return pd.read_parquet(path) if fmt == "parquet" else pd.read_pickle(path)
    obj = builder()
    obj.to_parquet(path, index=False) if fmt == "parquet" else obj.to_pickle(path)
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────
def run_main():
    # --- Load & prepare data ---
    universe_raw = load_universe(DATA / "All Sessions Shares.xlsx")
    print(f"Universe (raw):               {len(universe_raw)} tickers")

    bar_files = sorted(DATA.glob("bars*.csv*"))
    bars = load_or_build(
        CACHE / "bars.parquet",
        builder=lambda: load_bars(bar_files),
        fmt="parquet",
        force=False,
    )
    print(f"Bars:                         {len(bars):,} rows | {bars['ticker'].nunique()} tickers")

    universe = filter_universe(universe_raw, bars)
    print(f"Universe (after liquidity filter): {len(universe)} tickers")

    earnings = load_earnings(DATA / "eps_backtest_2021_2025.csv")

    features = load_or_build(
        CACHE / "features.parquet",
        builder=lambda: create_features(earnings, bars, universe),
        fmt="parquet",
        force=False,
    )
    print(f"Features:                     {len(features):,} rows")

    print("Building bars map ...")
    bars_map = build_bars_map(bars)

    # --- Run ---
    DO_OPTIMIZE = True
    best_trades = None
    best_sharpe = -np.inf

    if not DO_OPTIMIZE:
        print("\n--- SINGLE BACKTEST ---")
        parameters["eps_surprise_threshold"] = 0.03
        parameters["rank_cutoff"]            = 0.70
        parameters["stop_loss"]              = None

        signals     = create_signals(features)
        best_trades = backtest(signals, bars_map)

    else:
        print("\n--- PARAMETER OPTIMISATION ---")
        surprise_range = [0.01, 0.02, 0.03]
        rank_range     = [0.50, 0.60, 0.70]
        results_list   = []

        for s_th in surprise_range:
            for r_ct in rank_range:
                parameters["eps_surprise_threshold"] = s_th
                parameters["rank_cutoff"]            = r_ct
                parameters["stop_loss"]              = None

                sigs = create_signals(features)
                trds = backtest(sigs, bars_map)

                if len(trds) == 0:
                    continue

                stats          = produce_results(trds)
                current_sharpe = stats.get("sharpe_ratio", 0)

                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    best_trades = trds.copy()

                results_list.append({
                    "EPS_Thresh": s_th,
                    "Rank_Cut":   r_ct,
                    "Sharpe":     current_sharpe,
                    "Trades":     stats.get("number_of_trades", 0),
                    "WinRate":    stats.get("win_rate",         0),
                    "MaxDD":      stats.get("max_drawdown",     0),
                })
                print(f"  EPS {s_th:.2f} | Rank {r_ct:.2f}  ->  Sharpe {current_sharpe:.3f}")

        df_results = pd.DataFrame(results_list)
        if not df_results.empty:
            print("\n--- TOP 5 BY SHARPE ---")
            print(df_results.sort_values("Sharpe", ascending=False).head(5).to_string(index=False))

    # --- Report ---
    if best_trades is not None:
        stats = produce_results(best_trades)
        print("\n" + "=" * 35)
        print("   BEST CONFIGURATION REPORT")
        print("=" * 35)
        print(f"Total Trades:      {stats['number_of_trades']}")
        print(f"Win Rate:          {stats['win_rate']:.2%}")
        print(f"Sharpe Ratio:      {stats['sharpe_ratio']:.3f}")
        print("-" * 35)
        print(f"Payoff Ratio:      {stats['payoff_ratio']:.2f}x")
        print(f"Avg Win:           {stats['average_win']:.4f}")
        print(f"Avg Loss:          {stats['average_loss']:.4f}")
        print(f"Expectancy:        {stats['expected']:.4f}")
        print("-" * 35)
        print(f"Max Drawdown:      {stats['max_drawdown']:.2%}")
        print(f"Max DD Duration:   {stats['max_drawdown_duration']} trades")
        print(f"Final Equity:      {stats['final_equity']:.3f}x")
        print("=" * 35)


if __name__ == "__main__":
    run_main()
