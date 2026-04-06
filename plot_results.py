"""
plot_results.py
---------------
Runs the strategy with current configuration and produces a full suite of
diagnostic plots.

Usage:
    python plot_results.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from source.configuration import parameters
from source.io import load_universe, load_bars, load_earnings, filter_universe
from source.features import create_features
from source.signals import create_signals
from source.backtest import backtest, build_bars_map


ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "data"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def load_or_build(path: Path, builder, fmt: str = "parquet", force: bool = False):
    """Loads a cached DataFrame if it exists, otherwise builds and caches it."""
    if not force and path.exists():
        return pd.read_parquet(path) if fmt == "parquet" else pd.read_pickle(path)
    obj = builder()
    obj.to_parquet(path, index=False) if fmt == "parquet" else obj.to_pickle(path)
    return obj


def main():
    # --- Load data ---
    universe_raw = load_universe(DATA / "All Sessions Shares.xlsx")
    bar_files    = sorted(DATA.glob("bars*.csv*"))

    bars     = load_or_build(CACHE / "bars.parquet",     lambda: load_bars(bar_files),                          fmt="parquet")
    universe = filter_universe(universe_raw, bars)
    earnings = load_earnings(DATA / "eps_backtest_2021_2025.csv")
    features = load_or_build(CACHE / "features.parquet", lambda: create_features(earnings, bars, universe),     fmt="parquet")
    bars_map = build_bars_map(bars)

    # --- Run strategy ---
    signals = create_signals(features)
    trades  = backtest(signals, bars_map)

    if trades is None or trades.empty:
        raise SystemExit("No trades produced — nothing to plot.")

    # ── Build daily equity from risk-scaled trade returns ─────────
    # Raw returns reflect actual trade economics (used for equity curve & drawdown).
    # Scaled returns incorporate vol normalisation (used for Sharpe display).
    risk_per_trade = float(parameters.get("risk_per_trade", 0.02))
    raw_rets       = trades["return"].astype(float).copy()

    # Optional: weight by abs_surprise (clips to avoid outsized individual trades)
    if "abs_surprise" in trades.columns and trades["abs_surprise"].notna().any():
        w        = trades["abs_surprise"].astype(float)
        w        = (w / w.mean()).clip(0.5, 1.5)
        raw_rets = raw_rets * w

    raw_rets     = raw_rets * risk_per_trade
    trade_equity = (1.0 + raw_rets.fillna(0)).cumprod()

    equity               = trade_equity.copy()
    equity.index         = pd.to_datetime(trades["exit_time_utc"], utc=True)
    equity               = equity.sort_index()
    daily_equity         = equity.resample("1D").last().ffill()
    daily_returns        = daily_equity.pct_change().dropna()

    # Drawdown
    peak = daily_equity.cummax()
    dd   = (daily_equity - peak) / peak

    # Rolling 1Y Sharpe
    rolling_sharpe = (
        daily_returns.rolling(252).mean() / daily_returns.rolling(252).std()
    ) * (252 ** 0.5)

    # Year-by-year returns
    yearly_returns = daily_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    # Long / short split
    longs  = (trades.loc[trades["direction"] == "long",  "return"].astype(float)
              if "direction" in trades.columns else pd.Series(dtype=float))
    shorts = (trades.loc[trades["direction"] == "short", "return"].astype(float)
              if "direction" in trades.columns else pd.Series(dtype=float))

    # ── Plot 1: Daily equity curve ────────────────────────────────
    plt.figure()
    plt.plot(daily_equity.index, daily_equity.values)
    plt.title("Daily Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.tight_layout()
    plt.show()

    # ── Plot 2: Underwater (drawdown) ─────────────────────────────
    plt.figure()
    plt.fill_between(dd.index, dd.values, 0, alpha=0.7)
    plt.title("Underwater Plot (Drawdown)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

    # ── Plot 3: Histogram of daily returns ────────────────────────
    plt.figure()
    plt.hist(daily_returns.values, bins=40)
    plt.title("Histogram of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ── Plot 4: Rolling 1Y Sharpe ─────────────────────────────────
    plt.figure()
    plt.plot(rolling_sharpe.index, rolling_sharpe.values)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Rolling 1Y Sharpe (annualised)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    plt.show()

    # ── Plot 5: Cumulative PnL by trade number ────────────────────
    plt.figure()
    plt.plot(trade_equity.values)
    plt.title("Cumulative PnL by Trade Number")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (normalised)")
    plt.tight_layout()
    plt.show()

    # ── Plot 6: Return vs realised volatility scatter ─────────────
    window         = 21
    rolling_ret    = (1 + daily_returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)
    rolling_vol    = daily_returns.rolling(window).std() * (252 ** 0.5)
    rv             = pd.concat([rolling_vol.rename("vol"), rolling_ret.rename("ret")], axis=1).dropna()

    plt.figure()
    plt.scatter(rv["vol"].values, rv["ret"].values, alpha=0.5)
    plt.title(f"Return vs Volatility Scatter (rolling {window}D)")
    plt.xlabel("Realised Volatility (annualised)")
    plt.ylabel(f"Realised Return over {window}D")
    plt.tight_layout()
    plt.show()

    # ── Plot 7: Long vs Short return distributions ────────────────
    if not longs.empty or not shorts.empty:
        plt.figure()
        if not longs.empty:
            plt.hist(longs.values,  bins=20, alpha=0.6, label="Long")
        if not shorts.empty:
            plt.hist(shorts.values, bins=20, alpha=0.6, label="Short")
        plt.title("Trade Return Distribution: Long vs Short")
        plt.xlabel("Trade Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ── Plot 8: Trade return vs EPS surprise magnitude ────────────
    if "abs_surprise" in trades.columns:
        plt.figure()
        plt.scatter(
            trades["abs_surprise"].astype(float),
            trades["return"].astype(float),
            alpha=0.5,
        )
        plt.title("Trade Return vs |EPS Surprise|")
        plt.xlabel("|EPS Surprise|")
        plt.ylabel("Trade Return")
        plt.tight_layout()
        plt.show()

    # ── Plot 9: Year-by-year returns ──────────────────────────────
    plt.figure()
    plt.bar(yearly_returns.index.year, yearly_returns.values)
    plt.title("Year-by-Year Returns (daily compounded)")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.show()

    # ── Summary stats ─────────────────────────────────────────────
    ann_sharpe = (
        (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        if daily_returns.std() != 0 else None
    )
    print("\nSummary stats:")
    print(f"  Trades:                  {len(trades)}")
    print(f"  Annualised Sharpe:       {ann_sharpe:.3f}" if ann_sharpe else "  Annualised Sharpe:       N/A")
    print(f"  Final equity (daily):    {daily_equity.iloc[-1]:.4f}")
    print(f"  Max drawdown (daily):    {dd.min():.2%}")


if __name__ == "__main__":
    main()
