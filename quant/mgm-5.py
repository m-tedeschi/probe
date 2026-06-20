######################
### WALK-FORWARD VERSION ###
# 1. Uses rolling train/test splits
# 2. Selects params on train only
# 3. Evaluates on next unseen test window
# 4. Aggregates out-of-sample performance
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
# Strategy parameter grid
# ----------------------------
ENTRY_Z_GRID = [-1.0, -1.25, -1.5, -1.75]
EXIT_Z_GRID = [-0.5, -0.25, 0.0]
SLOPE_THRESHOLD_GRID = [0.0010, 0.00125, 0.0015, 0.0020]
STOP_LOSS_GRID = [-0.04, -0.05, -0.06, -0.08]
MAX_HOLD_GRID = [5, 7, 10, 15]

MIN_TRAIN_TRADES = 6

# ----------------------------
# Walk-forward config
# ----------------------------
TRAIN_BARS = 504   # ~2 years
TEST_BARS = 126    # ~6 months
STEP_BARS = 126    # advance by one test window each fold

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

def choose_best_params(train_df: pd.DataFrame):
    param_grid = itertools.product(
        ENTRY_Z_GRID,
        EXIT_Z_GRID,
        SLOPE_THRESHOLD_GRID,
        STOP_LOSS_GRID,
        MAX_HOLD_GRID
    )

    candidates = []

    for entry_z, exit_z, slope_threshold_pct, stop_loss, max_hold_bars in param_grid:
        _, _, stats = run_backtest(
            train_df,
            entry_z=entry_z,
            exit_z=exit_z,
            slope_threshold_pct=slope_threshold_pct,
            stop_loss=stop_loss,
            max_hold_bars=max_hold_bars
        )

        if stats["trade_count"] >= MIN_TRAIN_TRADES and not pd.isna(stats["sharpe"]):
            candidates.append(stats)

    if not candidates:
        return None, None

    candidates_df = pd.DataFrame(candidates).sort_values(
        by=["sharpe", "total_return", "trade_count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best = candidates_df.iloc[0].to_dict()
    return best, candidates_df

# ----------------------------
# Walk-forward loop
# ----------------------------
fold_results = []
oos_returns = []
best_fold_run = None
best_fold_sharpe = -np.inf

start_idx = max(price_lookback, fft_window)

fold_num = 0
for train_start in range(start_idx, len(df) - TRAIN_BARS - TEST_BARS + 1, STEP_BARS):
    train_end = train_start + TRAIN_BARS
    test_end = train_end + TEST_BARS

    train_df = df.iloc[train_start:train_end].copy()
    test_df = df.iloc[train_end:test_end].copy()

    best_params, train_candidates_df = choose_best_params(train_df)

    if best_params is None:
        continue

    fold_num += 1

    test_run_df, test_trades_df, test_stats = run_backtest(
        test_df,
        entry_z=float(best_params["entry_z"]),
        exit_z=float(best_params["exit_z"]),
        slope_threshold_pct=float(best_params["slope_threshold_pct"]),
        stop_loss=float(best_params["stop_loss"]),
        max_hold_bars=int(best_params["max_hold_bars"])
    )

    buy_hold_test = (1 + test_run_df["asset_ret"].fillna(0)).cumprod().iloc[-1] - 1

    fold_summary = {
        "fold": fold_num,
        "train_start": train_df.index[0],
        "train_end": train_df.index[-1],
        "test_start": test_df.index[0],
        "test_end": test_df.index[-1],
        "entry_z": best_params["entry_z"],
        "exit_z": best_params["exit_z"],
        "slope_threshold_pct": best_params["slope_threshold_pct"],
        "stop_loss": best_params["stop_loss"],
        "max_hold_bars": best_params["max_hold_bars"],
        "train_sharpe": best_params["sharpe"],
        "train_return": best_params["total_return"],
        "test_sharpe": test_stats["sharpe"],
        "test_return": test_stats["total_return"],
        "test_buy_hold_return": buy_hold_test,
        "test_max_drawdown": test_stats["max_drawdown"],
        "test_trade_count": test_stats["trade_count"],
        "test_win_rate": test_stats["win_rate"],
    }
    fold_results.append(fold_summary)

    fold_ret = test_run_df[["strategy_ret_net", "asset_ret"]].copy()
    fold_ret["fold"] = fold_num
    oos_returns.append(fold_ret)

    if not pd.isna(test_stats["sharpe"]) and test_stats["sharpe"] > best_fold_sharpe:
        best_fold_sharpe = test_stats["sharpe"]
        best_fold_run = (test_run_df.copy(), test_trades_df.copy(), fold_summary)

# ----------------------------
# Aggregate OOS results
# ----------------------------
if not fold_results:
    raise RuntimeError("No valid walk-forward folds were produced.")

folds_df = pd.DataFrame(fold_results)

oos_df = pd.concat(oos_returns).sort_index()
oos_df = oos_df[~oos_df.index.duplicated(keep="first")].copy()

oos_df["strategy_curve"] = (1 + oos_df["strategy_ret_net"].fillna(0)).cumprod()
oos_df["buy_hold_curve"] = (1 + oos_df["asset_ret"].fillna(0)).cumprod()

aggregate = {
    "fold_count": len(folds_df),
    "oos_total_return": oos_df["strategy_curve"].iloc[-1] - 1,
    "oos_buy_hold_return": oos_df["buy_hold_curve"].iloc[-1] - 1,
    "oos_cagr": annualized_return(oos_df["strategy_curve"]),
    "oos_buy_hold_cagr": annualized_return(oos_df["buy_hold_curve"]),
    "oos_ann_vol": annualized_vol(oos_df["strategy_ret_net"].dropna()),
    "oos_sharpe": sharpe(oos_df["strategy_ret_net"].dropna()),
    "oos_max_drawdown": max_drawdown(oos_df["strategy_curve"]),
    "mean_fold_test_return": folds_df["test_return"].mean(),
    "median_fold_test_return": folds_df["test_return"].median(),
    "mean_fold_test_sharpe": folds_df["test_sharpe"].mean(),
    "median_fold_test_sharpe": folds_df["test_sharpe"].median(),
    "mean_fold_trade_count": folds_df["test_trade_count"].mean(),
}

print("\n=== Walk-Forward Fold Results ===")
print(folds_df.to_string(index=False))

print("\n=== Aggregate OOS Summary ===")
for k, v in aggregate.items():
    if isinstance(v, (int, np.integer)):
        print(f"{k:24s}: {v}")
    elif pd.isna(v):
        print(f"{k:24s}: NaN")
    else:
        print(f"{k:24s}: {v:.4f}")

print("\n=== Parameter Frequency Across Folds ===")
for col in ["entry_z", "exit_z", "slope_threshold_pct", "stop_loss", "max_hold_bars"]:
    print(f"\n-- {col} --")
    print(folds_df[col].value_counts().sort_index().to_string())

# ----------------------------
# Plot aggregate OOS curve
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(oos_df.index, oos_df["strategy_curve"], label="Walk-Forward OOS Strategy")
plt.plot(oos_df.index, oos_df["buy_hold_curve"], label="Walk-Forward OOS Buy & Hold")
plt.title("Walk-Forward Out-of-Sample Equity Curve")
plt.legend()
plt.show()

# ----------------------------
# Plot best OOS fold
# ----------------------------
best_run_df, best_trades_df, best_fold_summary = best_fold_run

print("\n=== Best OOS Fold Summary ===")
for k, v in best_fold_summary.items():
    if isinstance(v, pd.Timestamp):
        print(f"{k:24s}: {v.date()}")
    elif isinstance(v, (int, np.integer)):
        print(f"{k:24s}: {v}")
    elif pd.isna(v):
        print(f"{k:24s}: NaN")
    else:
        print(f"{k:24s}: {v:.4f}")

if len(best_trades_df) > 0:
    print("\n=== Best OOS Fold Trades ===")
    print(best_trades_df.to_string(index=False))
    print("\n=== Best OOS Fold Exit Reason Counts ===")
    print(best_trades_df["exit_reason"].value_counts().to_string())

plt.figure(figsize=(12, 5))
plt.plot(best_run_df.index, best_run_df["Close"], label="MGM Close")
plt.plot(best_run_df.index, best_run_df["ma60"], label="60D Mean")

entry_points = best_run_df.index[best_run_df["executed_entry"]]
exit_points = best_run_df.index[best_run_df["executed_exit"]]

plt.scatter(entry_points, best_run_df.loc[entry_points, "Close"], marker="^", s=70, label="Executed Entry")
plt.scatter(exit_points, best_run_df.loc[exit_points, "Close"], marker="v", s=70, label="Executed Exit")

plt.title("Best OOS Fold: MGM Price, 60D Mean, and Executed Trades")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(best_run_df.index, best_run_df["zscore60"], label="60D Z-Score")
plt.axhline(best_fold_summary["entry_z"], linestyle="--", label="Entry Threshold")
plt.axhline(best_fold_summary["exit_z"], linestyle="--", label="Exit Threshold")
plt.axhline(0, linestyle=":")
plt.title("Best OOS Fold: Rolling 60D Z-Score")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(best_run_df.index, best_run_df["equity_curve"], label="Best OOS Fold Strategy")
plt.plot(best_run_df.index, best_run_df["buy_hold_curve"], label="Best OOS Fold Buy & Hold")
plt.title("Best OOS Fold Equity Curve")
plt.legend()
plt.show()
