######################
### GRID SEARCH VERSION ###
# 1. Keeps FFT/band-power diagnostics
# 2. Uses reusable backtest function
# 3. Sweeps key parameters
# 4. Prints ranked results
# 5. Plots best-performing run
######################

import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------
# Data / feature config
# ----------------------------
ticker = "MGM"
period = "5y"
price_lookback = 60
fft_window = 126
cost_per_trade = 0.001  # 10 bps per side

# ----------------------------
# Grid config
# ----------------------------
ENTRY_Z_GRID = [-1.0, -1.25, -1.5, -1.75]
EXIT_Z_GRID = [-0.5, -0.25, 0.0]
SLOPE_THRESHOLD_GRID = [0.0010, 0.00125, 0.0015, 0.0020]
STOP_LOSS_GRID = [-0.04, -0.05, -0.06, -0.08]
MAX_HOLD_GRID = [5, 7, 10, 15]

TOP_N_TO_PRINT = 20
MIN_TRADES_FILTER = 8  # helps ignore ultra-sparse solutions

# ----------------------------
# Download data
# ----------------------------
df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close", "Volume"]].dropna().copy()
df["Close"] = df["Close"].squeeze()
df["Volume"] = df["Volume"].squeeze()

close = df["Close"]

# ----------------------------
# Base features
# ----------------------------
df["log_close"] = np.log(close)
df["log_ret"] = df["log_close"].diff()

df["ma60"] = close.rolling(price_lookback).mean()
df["std60"] = close.rolling(price_lookback).std()
df["zscore60"] = (close - df["ma60"]) / df["std60"]

# fixed slope feature; threshold changes in the grid
slope_lookback = 10
df["ma60_slope_pct"] = df["ma60"].pct_change(slope_lookback) / slope_lookback

df["rv20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
df["resid"] = df["log_close"] - df["log_close"].rolling(price_lookback).mean()

# ----------------------------
# Rolling spectral diagnostics
# ----------------------------
dominant_periods = [np.nan] * len(df)
power_10_20 = [np.nan] * len(df)
power_20_40 = [np.nan] * len(df)
power_40_80 = [np.nan] * len(df)
power_total = [np.nan] * len(df)
power_low_ratio = [np.nan] * len(df)

resid_vals = df["resid"].to_numpy()

for i in range(fft_window, len(df)):
    x = resid_vals[i - fft_window:i].copy()

    if np.isnan(x).any():
        continue

    x = x - np.mean(x)
    win = np.hanning(fft_window)
    xw = x * win

    fft_vals = np.fft.rfft(xw)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(fft_window, d=1)

    if len(power) == 0:
        continue

    power[0] = 0.0

    periods = np.full_like(freqs, np.nan, dtype=float)
    valid = freqs > 0
    periods[valid] = 1.0 / freqs[valid]

    band_10_20 = (periods >= 10) & (periods < 20)
    band_20_40 = (periods >= 20) & (periods < 40)
    band_40_80 = (periods >= 40) & (periods <= 80)
    band_all = (periods >= 10) & (periods <= 80)

    p10_20 = np.nansum(power[band_10_20])
    p20_40 = np.nansum(power[band_20_40])
    p40_80 = np.nansum(power[band_40_80])
    pall = np.nansum(power[band_all])

    power_10_20[i] = p10_20
    power_20_40[i] = p20_40
    power_40_80[i] = p40_80
    power_total[i] = pall

    if pall > 0:
        power_low_ratio[i] = (p20_40 + p40_80) / pall

    if np.any(band_all):
        idx_local = np.argmax(power[band_all])
        idx_global = np.where(band_all)[0][idx_local]
        dominant_periods[i] = periods[idx_global]

df["dominant_period"] = dominant_periods
df["power_10_20"] = power_10_20
df["power_20_40"] = power_20_40
df["power_40_80"] = power_40_80
df["power_total"] = power_total
df["power_low_ratio"] = power_low_ratio

# optional diagnostic only
df["spectral_ok"] = (
    (df["power_total"] > df["power_total"].rolling(60).median()) &
    (df["power_low_ratio"] > 0.55)
)

# ----------------------------
# Helpers
# ----------------------------
def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1
    return float(dd.min())

def annualized_return(curve: pd.Series) -> float:
    n = len(curve)
    if n < 2:
        return np.nan
    years = n / 252
    if curve.iloc[-1] <= 0:
        return np.nan
    return float(curve.iloc[-1] ** (1 / years) - 1)

def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))

def sharpe(returns: pd.Series) -> float:
    vol = annualized_vol(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((returns.mean() * 252) / vol)

def build_trade_list(run_df: pd.DataFrame, stop_loss: float, max_hold_bars: int) -> pd.DataFrame:
    trades = []
    entry_idx = None
    entry_px = None
    entry_bar = None

    for i in range(len(run_df)):
        if run_df["executed_entry"].iloc[i]:
            entry_idx = run_df.index[i]
            entry_px = run_df["Close"].iloc[i]
            entry_bar = i

        elif run_df["executed_exit"].iloc[i] and entry_idx is not None:
            exit_idx = run_df.index[i]
            exit_px = run_df["Close"].iloc[i]

            gross_ret = exit_px / entry_px - 1.0
            net_ret = gross_ret - 2 * cost_per_trade
            bars_held = i - entry_bar
            cal_days = (exit_idx - entry_idx).days

            reason = "signal"
            open_ret = run_df["trade_return_open"].iloc[i]
            hold_bars = run_df["bars_in_trade"].iloc[i]

            if pd.notna(open_ret) and open_ret <= stop_loss:
                reason = "stop"
            elif pd.notna(hold_bars) and hold_bars >= max_hold_bars:
                reason = "time"

            trades.append({
                "entry_date": entry_idx,
                "exit_date": exit_idx,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "bars_held": bars_held,
                "days_held": cal_days,
                "exit_reason": reason
            })

            entry_idx = None
            entry_px = None
            entry_bar = None

    return pd.DataFrame(trades)

def run_backtest(base_df: pd.DataFrame,
                 entry_z: float,
                 exit_z: float,
                 slope_threshold_pct: float,
                 stop_loss: float,
                 max_hold_bars: int):
    run_df = base_df.copy()

    run_df["range_regime"] = run_df["ma60_slope_pct"].abs() < slope_threshold_pct

    run_df["entry_setup"] = (
        (run_df["zscore60"] < entry_z) &
        (run_df["range_regime"])
    )

    run_df["exit_setup"] = (
        (run_df["zscore60"] > exit_z) |
        (~run_df["range_regime"])
    )

    position = np.zeros(len(run_df), dtype=int)
    trade_flag = np.zeros(len(run_df), dtype=int)
    executed_entry = np.zeros(len(run_df), dtype=bool)
    executed_exit = np.zeros(len(run_df), dtype=bool)
    bars_in_trade_arr = np.full(len(run_df), np.nan)
    trade_return_arr = np.full(len(run_df), np.nan)

    in_pos = False
    entry_price = np.nan
    bars_in_trade = 0

    for i in range(1, len(run_df)):
        if not in_pos:
            if bool(run_df["entry_setup"].iloc[i - 1]):
                in_pos = True
                position[i] = 1
                trade_flag[i] = 1
                executed_entry[i] = True
                entry_price = run_df["Close"].iloc[i]
                bars_in_trade = 1
                bars_in_trade_arr[i] = bars_in_trade
                trade_return_arr[i] = 0.0
            else:
                position[i] = 0
        else:
            current_price = run_df["Close"].iloc[i]
            running_trade_return = current_price / entry_price - 1.0

            exit_signal_prev = bool(run_df["exit_setup"].iloc[i - 1])
            stop_triggered = running_trade_return <= stop_loss
            time_triggered = bars_in_trade >= max_hold_bars

            if exit_signal_prev or stop_triggered or time_triggered:
                in_pos = False
                position[i] = 0
                trade_flag[i] = 1
                executed_exit[i] = True
                bars_in_trade_arr[i] = bars_in_trade
                trade_return_arr[i] = running_trade_return
                entry_price = np.nan
                bars_in_trade = 0
            else:
                position[i] = 1
                bars_in_trade += 1
                bars_in_trade_arr[i] = bars_in_trade
                trade_return_arr[i] = running_trade_return

    run_df["position"] = position
    run_df["trade_flag"] = trade_flag
    run_df["executed_entry"] = executed_entry
    run_df["executed_exit"] = executed_exit
    run_df["bars_in_trade"] = bars_in_trade_arr
    run_df["trade_return_open"] = trade_return_arr

    run_df["asset_ret"] = run_df["Close"].pct_change()
    run_df["strategy_ret_gross"] = run_df["position"].shift(1).fillna(0) * run_df["asset_ret"]
    run_df["turnover"] = run_df["position"].diff().abs().fillna(0)
    run_df["strategy_ret_net"] = run_df["strategy_ret_gross"] - run_df["turnover"] * cost_per_trade

    run_df["equity_curve"] = (1 + run_df["strategy_ret_net"].fillna(0)).cumprod()
    run_df["buy_hold_curve"] = (1 + run_df["asset_ret"].fillna(0)).cumprod()

    trades_df = build_trade_list(run_df, stop_loss=stop_loss, max_hold_bars=max_hold_bars)

    stats = {
        "entry_z": entry_z,
        "exit_z": exit_z,
        "slope_threshold_pct": slope_threshold_pct,
        "stop_loss": stop_loss,
        "max_hold_bars": max_hold_bars,
        "total_return": run_df["equity_curve"].iloc[-1] - 1,
        "buy_hold_return": run_df["buy_hold_curve"].iloc[-1] - 1,
        "cagr": annualized_return(run_df["equity_curve"]),
        "buy_hold_cagr": annualized_return(run_df["buy_hold_curve"]),
        "ann_vol": annualized_vol(run_df["strategy_ret_net"].dropna()),
        "sharpe": sharpe(run_df["strategy_ret_net"].dropna()),
        "max_drawdown": max_drawdown(run_df["equity_curve"]),
        "exposure": run_df["position"].mean(),
        "trade_count": len(trades_df),
    }

    if len(trades_df) > 0:
        stats["win_rate"] = float((trades_df["net_ret"] > 0).mean())
        stats["avg_trade_return"] = float(trades_df["net_ret"].mean())
        stats["median_trade_return"] = float(trades_df["net_ret"].median())
        stats["avg_hold_bars"] = float(trades_df["bars_held"].mean())
    else:
        stats["win_rate"] = np.nan
        stats["avg_trade_return"] = np.nan
        stats["median_trade_return"] = np.nan
        stats["avg_hold_bars"] = np.nan

    return run_df, trades_df, stats

# ----------------------------
# Grid search
# ----------------------------
results = []

param_grid = list(itertools.product(
    ENTRY_Z_GRID,
    EXIT_Z_GRID,
    SLOPE_THRESHOLD_GRID,
    STOP_LOSS_GRID,
    MAX_HOLD_GRID
))

print(f"Running {len(param_grid)} parameter combinations...")

for idx, (entry_z, exit_z, slope_threshold_pct, stop_loss, max_hold_bars) in enumerate(param_grid, start=1):
    _, _, stats = run_backtest(
        df,
        entry_z=entry_z,
        exit_z=exit_z,
        slope_threshold_pct=slope_threshold_pct,
        stop_loss=stop_loss,
        max_hold_bars=max_hold_bars
    )
    results.append(stats)

    if idx % 50 == 0 or idx == len(param_grid):
        print(f"Completed {idx}/{len(param_grid)}")

results_df = pd.DataFrame(results)

# useful filters for readability
filtered_results = results_df[
    results_df["trade_count"] >= MIN_TRADES_FILTER
].copy()

filtered_results = filtered_results.sort_values(
    by=["sharpe", "total_return", "trade_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("\n=== Top Grid Search Results (filtered) ===")
print(filtered_results.head(TOP_N_TO_PRINT).to_string(index=False))

print("\n=== Parameter Robustness Summary ===")
for col in ["entry_z", "exit_z", "slope_threshold_pct", "stop_loss", "max_hold_bars"]:
    summary = (
        filtered_results.groupby(col)
        .agg(
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            mean_return=("total_return", "mean"),
            median_return=("total_return", "median"),
            mean_drawdown=("max_drawdown", "mean"),
            mean_trades=("trade_count", "mean"),
            count=("sharpe", "count")
        )
        .sort_values("mean_sharpe", ascending=False)
    )
    print(f"\n-- {col} --")
    print(summary.to_string())

if len(filtered_results) == 0:
    raise RuntimeError("No parameter combinations met the minimum trade-count filter.")

best = filtered_results.iloc[0].to_dict()

print("\n=== Best Parameter Set ===")
for k in ["entry_z", "exit_z", "slope_threshold_pct", "stop_loss", "max_hold_bars",
          "total_return", "cagr", "sharpe", "max_drawdown", "trade_count",
          "win_rate", "avg_trade_return", "median_trade_return", "avg_hold_bars", "exposure"]:
    v = best[k]
    if isinstance(v, (int, np.integer)):
        print(f"{k:20s}: {v}")
    else:
        print(f"{k:20s}: {v:.4f}")

# ----------------------------
# Re-run best model for detailed output
# ----------------------------
best_run_df, best_trades_df, best_stats = run_backtest(
    df,
    entry_z=float(best["entry_z"]),
    exit_z=float(best["exit_z"]),
    slope_threshold_pct=float(best["slope_threshold_pct"]),
    stop_loss=float(best["stop_loss"]),
    max_hold_bars=int(best["max_hold_bars"])
)

print("\n=== Best Run Recent Feature Snapshot ===")
cols = [
    "Close", "zscore60", "ma60_slope_pct", "range_regime",
    "dominant_period", "power_10_20", "power_20_40", "power_40_80",
    "power_low_ratio", "spectral_ok", "entry_setup", "exit_setup",
    "position", "bars_in_trade", "trade_return_open"
]
print(best_run_df[cols].tail(15).to_string())

if len(best_trades_df) > 0:
    print("\n=== Best Run Recent Trades ===")
    print(best_trades_df.tail(15).to_string(index=False))
    print("\n=== Best Run Exit Reason Counts ===")
    print(best_trades_df["exit_reason"].value_counts().to_string())
else:
    print("\nNo completed trades for the best run.")

# ----------------------------
# Plots for best run
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(best_run_df.index, best_run_df["Close"], label="MGM Close")
plt.plot(best_run_df.index, best_run_df["ma60"], label="60D Mean")

entry_points = best_run_df.index[best_run_df["executed_entry"]]
exit_points = best_run_df.index[best_run_df["executed_exit"]]

plt.scatter(entry_points, best_run_df.loc[entry_points, "Close"], marker="^", s=70, label="Executed Entry")
plt.scatter(exit_points, best_run_df.loc[exit_points, "Close"], marker="v", s=70, label="Executed Exit")

plt.title("Best Run: MGM Price, 60D Mean, and Executed Trades")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(best_run_df.index, best_run_df["zscore60"], label="60D Z-Score")
plt.axhline(float(best["entry_z"]), linestyle="--", label="Entry Threshold")
plt.axhline(float(best["exit_z"]), linestyle="--", label="Exit Threshold")
plt.axhline(0, linestyle=":")
plt.title("Best Run: Rolling 60D Z-Score")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(best_run_df.index, best_run_df["power_10_20"], label="Power 10-20d")
plt.plot(best_run_df.index, best_run_df["power_20_40"], label="Power 20-40d")
plt.plot(best_run_df.index, best_run_df["power_40_80"], label="Power 40-80d")
plt.title("Best Run: Rolling Spectral Band Powers")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(best_run_df.index, best_run_df["dominant_period"], label="Dominant Period")
plt.title("Best Run: Rolling Dominant Period (diagnostic only)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(best_run_df.index, best_run_df["equity_curve"], label="Strategy")
plt.plot(best_run_df.index, best_run_df["buy_hold_curve"], label="Buy & Hold")
plt.title("Best Run: Equity Curve")
plt.legend()
plt.show()
