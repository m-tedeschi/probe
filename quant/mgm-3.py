######################
### NEXT VERSION ###
# 1. Keeps Hann-window FFT + band powers as diagnostics
# 2. Uses a normalized range-regime filter
# 3. Removes spectral_ok as a hard entry gate
# 4. Adds time stop + stop loss
# 5. Plots actual executed entries/exits
######################

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
ticker = "MGM"
price_lookback = 60
fft_window = 126

entry_z = -1.25
exit_z = -0.25

slope_lookback = 10
slope_threshold_pct = 0.0015   # 0.15% avg daily slope over last N days
max_hold_days = 10
stop_loss = -0.06              # -6%
cost_per_trade = 0.001         # 10 bps per side

# ----------------------------
# Download data
# ----------------------------
df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close", "Volume"]].dropna().copy()
df["Close"] = df["Close"].squeeze()
df["Volume"] = df["Volume"].squeeze()

close = df["Close"]

# ----------------------------
# Core features
# ----------------------------
df["log_close"] = np.log(close)
df["log_ret"] = df["log_close"].diff()

df["ma60"] = close.rolling(price_lookback).mean()
df["std60"] = close.rolling(price_lookback).std()
df["zscore60"] = (close - df["ma60"]) / df["std60"]

# normalized range-regime proxy:
# average daily % slope of the 60D mean over the last slope_lookback days
df["ma60_slope_pct"] = df["ma60"].pct_change(slope_lookback) / slope_lookback
df["range_regime"] = df["ma60_slope_pct"].abs() < slope_threshold_pct

# optional diagnostics
df["rv20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)

# detrended residual for spectral work
df["resid"] = df["log_close"] - df["log_close"].rolling(price_lookback).mean()

# ----------------------------
# Rolling spectral features (diagnostic only)
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

# keep spectral_ok as a diagnostic only
df["spectral_ok"] = (
    (df["power_total"] > df["power_total"].rolling(60).median()) &
    (df["power_low_ratio"] > 0.55)
)

# ----------------------------
# Setup logic
# ----------------------------
df["entry_setup"] = (
    (df["zscore60"] < entry_z) &
    (df["range_regime"])
)

df["exit_setup"] = (
    (df["zscore60"] > exit_z) |
    (~df["range_regime"])
)

# ----------------------------
# Stateful backtest with time stop + stop loss
# ----------------------------
position = np.zeros(len(df), dtype=int)
trade_flag = np.zeros(len(df), dtype=int)

executed_entry = np.zeros(len(df), dtype=bool)
executed_exit = np.zeros(len(df), dtype=bool)

days_in_trade_arr = np.full(len(df), np.nan)
trade_return_arr = np.full(len(df), np.nan)

in_pos = False
entry_price = np.nan
days_in_trade = 0

for i in range(1, len(df)):
    if not in_pos:
        if bool(df["entry_setup"].iloc[i - 1]):
            in_pos = True
            position[i] = 1
            trade_flag[i] = 1
            executed_entry[i] = True
            entry_price = df["Close"].iloc[i]
            days_in_trade = 1
            days_in_trade_arr[i] = days_in_trade
            trade_return_arr[i] = 0.0
        else:
            position[i] = 0
    else:
        current_price = df["Close"].iloc[i]
        running_trade_return = current_price / entry_price - 1.0

        exit_signal_prev = bool(df["exit_setup"].iloc[i - 1])
        stop_triggered = running_trade_return <= stop_loss
        time_triggered = days_in_trade >= max_hold_days

        if exit_signal_prev or stop_triggered or time_triggered:
            in_pos = False
            position[i] = 0
            trade_flag[i] = 1
            executed_exit[i] = True
            days_in_trade_arr[i] = days_in_trade
            trade_return_arr[i] = running_trade_return
            entry_price = np.nan
            days_in_trade = 0
        else:
            position[i] = 1
            days_in_trade += 1
            days_in_trade_arr[i] = days_in_trade
            trade_return_arr[i] = running_trade_return

df["position"] = position
df["trade_flag"] = trade_flag
df["executed_entry"] = executed_entry
df["executed_exit"] = executed_exit
df["days_in_trade"] = days_in_trade_arr
df["trade_return_open"] = trade_return_arr

# ----------------------------
# Returns
# ----------------------------
df["asset_ret"] = close.pct_change()
df["strategy_ret_gross"] = df["position"].shift(1).fillna(0) * df["asset_ret"]
df["turnover"] = df["position"].diff().abs().fillna(0)
df["strategy_ret_net"] = df["strategy_ret_gross"] - df["turnover"] * cost_per_trade

df["equity_curve"] = (1 + df["strategy_ret_net"].fillna(0)).cumprod()
df["buy_hold_curve"] = (1 + df["asset_ret"].fillna(0)).cumprod()

# ----------------------------
# Trade list
# ----------------------------
trades = []
entry_idx = None
entry_px = None

for i in range(len(df)):
    if df["executed_entry"].iloc[i]:
        entry_idx = df.index[i]
        entry_px = df["Close"].iloc[i]

    elif df["executed_exit"].iloc[i] and entry_idx is not None:
        exit_idx = df.index[i]
        exit_px = df["Close"].iloc[i]

        gross_ret = exit_px / entry_px - 1
        net_ret = gross_ret - 2 * cost_per_trade

        # reason tag
        reason = "signal"
        open_ret = df["trade_return_open"].iloc[i]
        hold_days = df["days_in_trade"].iloc[i]

        if pd.notna(open_ret) and open_ret <= stop_loss:
            reason = "stop"
        elif pd.notna(hold_days) and hold_days >= max_hold_days:
            reason = "time"

        trades.append({
            "entry_date": entry_idx,
            "exit_date": exit_idx,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "days_held": (exit_idx - entry_idx).days,
            "exit_reason": reason
        })

        entry_idx = None
        entry_px = None

trades_df = pd.DataFrame(trades)

# ----------------------------
# Performance stats
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
    return float(curve.iloc[-1] ** (1 / years) - 1)

def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))

def sharpe(returns: pd.Series) -> float:
    vol = annualized_vol(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((returns.mean() * 252) / vol)

stats = {
    "Total Return": df["equity_curve"].iloc[-1] - 1,
    "BuyHold Return": df["buy_hold_curve"].iloc[-1] - 1,
    "CAGR": annualized_return(df["equity_curve"]),
    "BuyHold CAGR": annualized_return(df["buy_hold_curve"]),
    "Ann Vol": annualized_vol(df["strategy_ret_net"].dropna()),
    "Sharpe": sharpe(df["strategy_ret_net"].dropna()),
    "Max Drawdown": max_drawdown(df["equity_curve"]),
    "Exposure": df["position"].mean(),
    "Trade Count": len(trades_df)
}

if len(trades_df) > 0:
    stats["Win Rate"] = float((trades_df["net_ret"] > 0).mean())
    stats["Avg Trade Return"] = float(trades_df["net_ret"].mean())
    stats["Median Trade Return"] = float(trades_df["net_ret"].median())
    stats["Avg Hold Days"] = float(trades_df["days_held"].mean())
else:
    stats["Win Rate"] = np.nan
    stats["Avg Trade Return"] = np.nan
    stats["Median Trade Return"] = np.nan
    stats["Avg Hold Days"] = np.nan

print("\n=== Strategy Stats ===")
for k, v in stats.items():
    if isinstance(v, (int, np.integer)):
        print(f"{k:20s}: {v}")
    elif pd.isna(v):
        print(f"{k:20s}: NaN")
    else:
        print(f"{k:20s}: {v:.4f}")

print("\n=== Recent Feature Snapshot ===")
cols = [
    "Close", "zscore60", "ma60_slope_pct", "range_regime",
    "dominant_period", "power_10_20", "power_20_40", "power_40_80",
    "power_low_ratio", "spectral_ok", "entry_setup", "exit_setup",
    "position", "days_in_trade", "trade_return_open"
]
print(df[cols].tail(15))

if len(trades_df) > 0:
    print("\n=== Recent Trades ===")
    print(trades_df.tail(10))
    print("\n=== Exit Reason Counts ===")
    print(trades_df["exit_reason"].value_counts())
else:
    print("\nNo completed trades found with current parameters.")

# ----------------------------
# Plots
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"], label="MGM Close")
plt.plot(df.index, df["ma60"], label="60D Mean")

entry_points = df.index[df["executed_entry"]]
exit_points = df.index[df["executed_exit"]]

plt.scatter(entry_points, df.loc[entry_points, "Close"], marker="^", s=70, label="Executed Entry")
plt.scatter(exit_points, df.loc[exit_points, "Close"], marker="v", s=70, label="Executed Exit")

plt.title("MGM Price, 60D Mean, and Executed Trades")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df.index, df["zscore60"], label="60D Z-Score")
plt.axhline(entry_z, linestyle="--", label="Entry Threshold")
plt.axhline(exit_z, linestyle="--", label="Exit Threshold")
plt.axhline(0, linestyle=":")
plt.title("Rolling 60D Z-Score")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df.index, df["power_10_20"], label="Power 10-20d")
plt.plot(df.index, df["power_20_40"], label="Power 20-40d")
plt.plot(df.index, df["power_40_80"], label="Power 40-80d")
plt.title("Rolling Spectral Band Powers")
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df.index, df["dominant_period"], label="Dominant Period")
plt.title("Rolling Dominant Period (diagnostic only)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["equity_curve"], label="Strategy")
plt.plot(df.index, df["buy_hold_curve"], label="Buy & Hold")
plt.title("Equity Curve")
plt.legend()
plt.show()
