parameters = {
    # ── Signal filters ────────────────────────────────────────────
    "eps_surprise_threshold": 0.08,   # minimum |EPS surprise| to enter
    "ah_move_threshold":      0.005,  # minimum |after-hours move| to enter
    "volume_ratio_threshold": 0.02,   # minimum volume ratio vs median session

    # ── Entry ─────────────────────────────────────────────────────
    "entry_delay":   0,       # minutes after earnings release to enter
    "max_holding":   4500,    # maximum holding period in minutes (~3 trading days)

    # ── Exit ──────────────────────────────────────────────────────
    "stop_loss":    None,     # fractional stop-loss (e.g. 0.03 = 3%); None = off
    "take_profit":  None,     # fractional take-profit; None = off

    # ── Trailing stop ─────────────────────────────────────────────
    "use_trailing_stop":   True,  # enable trailing stop
    "trail_activation":    0.03,  # profit threshold before trailing activates
    "trail_amount":        0.02,  # drawdown from peak that triggers exit

    # ── Sizing & ranking ──────────────────────────────────────────
    "risk_per_trade": 0.02,   # fraction of equity risked per trade
    "rank_cutoff":    0.90,   # retain top (1 - rank_cutoff) by abs surprise
    "ah_confirm_long": True,  # require after-hours move > 0 for long entries

    # ── Volatility scaling ────────────────────────────────────────
    "target_risk":    0.01,   # target per-trade risk used in vol scaling

    # ── Market regime ─────────────────────────────────────────────
    "benchmark_ticker": "SPY",
    "market_ma_days":   100,  # SMA window for risk-on/risk-off filter
    "past_days":        40,   # reserved for future feature lookback use
}
