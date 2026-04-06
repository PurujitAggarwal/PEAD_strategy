import pandas as pd
import numpy as np


def equity_curve(returns: pd.Series) -> pd.Series:
    """Computes a cumulative product equity curve from a series of trade returns."""
    return (1 + returns.fillna(0).astype(float)).cumprod()


def drawdowns(equity: pd.Series):
    """
    Computes the drawdown series, maximum drawdown, and maximum drawdown duration
    (in number of consecutive underwater trades) from an equity curve.
    """
    equity = equity.astype(float)
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    max_dd = float(dd.min())

    max_dur = 0
    cur     = 0
    for x in dd:
        if x < 0:
            cur     += 1
            max_dur  = max(max_dur, cur)
        else:
            cur = 0

    return dd, max_dd, int(max_dur)


def calculate_trades(returns: pd.Series) -> dict:
    """Per-trade statistics: win rate, average win/loss, payoff ratio, expectancy."""
    returns  = returns.dropna().astype(float)
    wins     = returns[returns > 0]
    losses   = returns[returns <= 0]

    win_rate  = len(wins) / len(returns) if len(returns) else 0.0
    avg_win   = float(wins.mean())   if len(wins)   else 0.0
    avg_loss  = float(losses.mean()) if len(losses) else 0.0
    payoff    = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
    expected  = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "win_rate":     win_rate,
        "average_win":  avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": payoff,
        "expected":     expected,
    }


def produce_results(trades: pd.DataFrame) -> dict:
    """
    Computes a full performance report for a trades DataFrame produced by backtest().

    Sharpe ratio uses volatility-scaled returns (the ``scaled_return`` column) so
    that position sizing is reflected in the risk-adjusted metric.  All other
    statistics (win rate, average win/loss, equity curve, drawdown) use raw returns
    so they reflect actual trade economics without scaling artefacts.
    """
    if trades is None or trades.empty:
        return {
            "number_of_trades":    0,
            "max_drawdown":        0.0,
            "max_drawdown_duration": 0,
            "sharpe_ratio":        0.0,
            "win_rate":            0.0,
            "average_win":         0.0,
            "average_loss":        0.0,
            "payoff_ratio":        0.0,
            "expected":            0.0,
            "final_equity":        1.0,
        }

    exit_times      = pd.to_datetime(trades["exit_time_utc"], utc=True).sort_values()
    years           = (exit_times.iloc[-1] - exit_times.iloc[0]).days / 365.25
    trades_per_year = len(trades) / years if years > 0 else 0

    # Sharpe uses scaled returns (vol-normalised, reflects position sizing)
    scaled_rets = trades["scaled_return"].astype(float).copy()
    if len(scaled_rets) > 1 and scaled_rets.std() != 0 and trades_per_year > 0:
        sharpe = float((scaled_rets.mean() / scaled_rets.std()) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Equity curve and drawdown use raw returns (reflects actual trade economics)
    raw_rets = trades["return"].astype(float).copy()
    equity   = equity_curve(raw_rets)
    equity.index = pd.to_datetime(trades["exit_time_utc"], utc=True)
    equity   = equity.sort_index()

    _, max_dd, max_dd_dur = drawdowns(equity)
    stats = calculate_trades(raw_rets)

    return {
        "number_of_trades":      int(len(raw_rets)),
        "max_drawdown":          max_dd,
        "max_drawdown_duration": max_dd_dur,
        "sharpe_ratio":          sharpe,
        "win_rate":              stats["win_rate"],
        "average_win":           stats["average_win"],
        "average_loss":          stats["average_loss"],
        "payoff_ratio":          stats["payoff_ratio"],
        "expected":              stats["expected"],
        "final_equity":          float(equity.iloc[-1]),
    }
