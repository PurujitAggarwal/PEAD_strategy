# PEAD After-Hours Momentum Strategy

A systematic backtesting framework implementing a Post-Earnings Announcement Drift (PEAD) strategy. The strategy enters positions in the after-hours session immediately following earnings releases, exploiting the empirically documented tendency for prices to continue drifting in the direction of an EPS surprise over the subsequent trading days.

---

## Strategy Overview

### Entry

A trade signal is generated when all three of the following filters are satisfied:

1. **EPS surprise** — actual EPS deviates from the consensus estimate by at least 8% in absolute terms, calculated as `(actual − estimate) / |estimate|`
2. **After-hours move** — the price at entry has moved at least 0.5% relative to the prior regular-session close
3. **Volume ratio** — post-earnings volume exceeds the ticker's median regular-session bar volume by at least 2%

Direction is long on a positive EPS surprise and short on a negative one.

### Exit

Exits are evaluated in the following priority order at each bar within the holding window:

1. **Stop-loss** — exits if the position moves against entry by a configurable fraction (disabled by default)
2. **Take-profit** — exits if the position gains a configurable fraction (disabled by default)
3. **Trailing stop** — activates once a position gains 3% from entry; exits if the price subsequently retraces 2% from its intra-trade peak
4. **Time exit** — closes the position at the end of the maximum holding window, which defaults to approximately 3 trading days

### Market Regime Filter

Trades are only entered when SPY is above its 100-day simple moving average. When SPY is below this level (`risk_on = False`), all signals are suppressed, reducing drawdown exposure during sustained bear market regimes.

### Position Sizing

Each trade is volatility-scaled. The raw return is divided by the realised intra-window volatility and re-scaled to a target risk per trade. This normalises the economic contribution of each trade across high- and low-volatility names, producing a more stable Sharpe ratio estimate.

---

## Universe

100 US-listed equities covering a broad cross-section of the market, including large-cap technology (AAPL, MSFT, NVDA, GOOGL, META, AMZN), financials (JPM, GS, BAC, C, WFC), healthcare (JNJ, LLY, PFE, MRK, UNH), energy (XOM, CVX), high-beta and growth names (TSLA, PLTR, MSTR, COIN, HOOD, RDDT), and international ADRs (TSM, ASML, NVO, BABA, NIO).

The universe is filtered at runtime to tickers with a median daily dollar volume above $2M, ensuring minimum liquidity for realistic execution.

---

## Repository Structure

```
.
├── main.py              # Entry point: single backtest or parameter sweep
├── optimize.py          # Grid search over signal and exit parameters
├── plot_results.py      # Full suite of diagnostic plots
├── source/
│   ├── configuration.py # All parameters in one place
│   ├── backtest.py      # Vectorised event-driven backtesting engine
│   ├── features.py      # Feature engineering pipeline
│   ├── signals.py       # Signal generation and filter application
│   ├── metrics.py       # Performance reporting
│   ├── io.py            # Data loading
│   └── sessions.py      # Regular trading session identification
└── data/                # Not included — see Data Requirements below
```

---

## Key Design Decisions

**Pipeline separation.** Features, signals, and backtesting are kept as distinct stages. `create_features()` runs once before any parameter search loop, avoiding redundant computation on the expensive data join even across many grid search combinations.

**Vectorised exit logic.** Rather than iterating bar-by-bar, `backtest.py` computes the full return path across the holding window in a single NumPy pass, then applies all exit conditions with `np.where`. This is significantly faster for large universes over multi-year periods.

**O(log n) bar lookups.** `np.searchsorted` is used throughout `features.py` and `backtest.py` to locate relevant timestamps, avoiding linear scans through tick data.

**Scaled vs raw returns.** `backtest.py` records both `return` (raw trade economics) and `scaled_return` (vol-normalised). `metrics.py` uses `scaled_return` for the Sharpe ratio so that position sizing is reflected in the risk-adjusted metric, and uses `return` for equity curve, drawdown, and win/loss statistics to keep those figures economically interpretable.

**Parquet caching.** Bars and features DataFrames are cached to `data/cache/` on first build. Subsequent runs load from parquet rather than re-processing compressed CSVs, cutting startup time from minutes to seconds.

---

## Configuration

All parameters live in `source/configuration.py`. The key ones:

- `eps_surprise_threshold` (default `0.08`) — minimum |EPS surprise| to enter
- `ah_move_threshold` (default `0.005`) — minimum |after-hours move| to enter
- `volume_ratio_threshold` (default `0.02`) — minimum volume ratio vs median session bar
- `max_holding` (default `4500`) — maximum holding period in minutes (~3 trading days)
- `stop_loss` (default `None`) — fractional stop-loss; `None` = disabled
- `take_profit` (default `None`) — fractional take-profit; `None` = disabled
- `use_trailing_stop` (default `True`) — enable trailing stop
- `trail_activation` (default `0.03`) — profit level before trailing activates
- `trail_amount` (default `0.02`) — drawdown from peak that triggers trailing exit
- `risk_per_trade` (default `0.02`) — fraction of equity risked per trade
- `target_risk` (default `0.01`) — target per-trade risk used in volatility scaling
- `benchmark_ticker` (default `"SPY"`) — regime filter benchmark
- `market_ma_days` (default `100`) — SMA window for regime filter

---

## Usage

```bash
# Install dependencies
pip install pandas numpy matplotlib openpyxl pyarrow zstandard

# Run a single backtest or parameter sweep (controlled by DO_OPTIMIZE flag in main.py)
python main.py

# Grid search over EPS threshold, stop-loss, and volume ratio
python optimize.py

# Generate diagnostic plots
python plot_results.py
```

Place data files in a `data/` directory in the project root before running.

---

## Data Requirements

The `data/` directory is excluded from this repository. To run the backtest you will need three files:

**`All Sessions Shares.xlsx`** — ticker universe definition with a column named `Ticker`

**`bars_YYYY.csv.zst`** — minute-bar price data in zstd-compressed CSV format, with columns `ticker`, `timestamp_utc`, `close`, and `volume`. Multiple files are supported and will be concatenated automatically.

**`eps_backtest_2021_2025.csv`** — earnings releases with columns `ticker`, `earnings_datetime_utc`, `eps_actual`, and `eps_opinion`

Minute bar data can be sourced from [Polygon.io](https://polygon.io) or [Databento](https://databento.com). Earnings consensus estimates can be sourced from Refinitiv, FactSet, or [Intrinio](https://intrinio.com).

---

## Limitations

This is a research backtest and results should be interpreted carefully.

No transaction costs or slippage are modelled. After-hours spreads for the names in this universe are significantly wider than regular-session spreads, which would materially affect live performance.

The universe was selected based on well-known high-activity names rather than a point-in-time index, so survivorship and selection effects may inflate results.

Parameter selection on in-sample data will overstate out-of-sample performance. Walk-forward validation or a held-out test period is required before drawing conclusions about live viability.

The backtest period of 2021–2025 includes an unusually high-volatility earnings environment post-COVID. Performance in a lower-volatility regime may differ materially.
