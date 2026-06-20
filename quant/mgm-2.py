######################
### UPGRADES ###
# 1. Applies a Hann window before FFT
# 2. Runs FFT on a detrended log-price residual
# 3. Uses band powers instead of over-interpreting one dominant bin
# 4. Adds a basic mean-reversion backtest with a simple range filter
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
start_cash = 1.0

entry_z = -1.5
exit_z = 0.0
slope_lookback = 10
slope_threshold = 0.05   # crude "range regime" filter
cost_per_trade = 0.001   # 10 bps each side

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

# crude range-regime proxy: low slope on rolling mean
df["ma60_slope"] = df["ma60"].diff(slope_lookback) / slope_lookback
df["range_regime"] = df["ma60_slope"].abs() < slope_threshold

# realized vol (optional diagnostic)
df["rv20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)

# detrended residual for spectral work
df["resid"] = df["log_close"] - df["log_close"].rolling(price_lookback).mean()

# ----------------------------
# Rolling spectral features
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

    # demean residual window
    x = x - np.mean(x)

    # Hann window to reduce leakage
    win = np.hanning(fft_window)
    xw = x * win

    fft_vals = np.fft.rfft(xw)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(fft_window, d=1)

    if len(power) == 0:
        continue

    power[0] = 0.0  # ignore DC

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

    # dominant period only as a diagnostic, not the main signal
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

# spectral regime proxy:
# require a decent amount of medium/slow-cycle energy
df["spectral_ok"] = (
    (df["power_total"] > df["power_total"].rolling(60).median()) &
    (df["power_low_ratio"] > 0.55)
)

# ----------------------------
# Signal logic
# ----------------------------
df["entry_long"] = (
    (df["zscore60"] < entry_z) &
    (df["range_regime"]) &
    (df["spectral_ok"])
)

df["exit_long"] = (
    (df["zscore60"] > exit_z) |
    (~df["range_regime"])
)

# ----------------------------
# Simple long-only backtest
# ----------------------------
position = np.zeros(len(df), dtype=int)
trade_flag = np.zeros(len(df), dtype=int)

in_pos = False

for i in range(1, len(df)):
    if not in_pos:
        if bool(df["entry_long"].iloc[i - 1]):
            in_pos = True
            position[i] = 1
            trade_flag[i] = 1
        else:
            position[i] = 0
    else:
        if bool(df["exit_long"].iloc[i - 1]):
            in_pos = False
            position[i] = 0
            trade_flag[i] = 1
        else:
            position[i] = 1

df["position"] = position
df["trade_flag"] = trade_flag

# strategy returns: position decided using prior day's signal
df["asset_ret"] = close.pct_change()
df["strategy_ret_gross"] = df["position"].shift(1).fillna(0) * df["asset_ret"]

# transaction cost each time position changes
df["turnover"] = df["position"].diff().abs().fillna(0)
df["strategy_ret_net"] = df["strategy_ret_gross"] - df["turnover"] * cost_per_trade

df["equity_curve"] = (1 + df["strategy_ret_net"].fillna(0)).cumprod()
df["buy_hold_curve"] = (1 + df["asset_ret"].fillna(0)).cumprod()

# ----------------------------
# Trade list
# ----------------------------
trades = []
entry_idx = None
entry_price = None

for i in range(len(df)):
    pos_now = df["position"].iloc[i]
    pos_prev = df["position"].iloc[i - 1] if i > 0 else 0

    if pos_prev == 0 and pos_now == 1:
        entry_idx = df.index[i]
        entry_price = df["Close"].iloc[i]

    elif pos_prev == 1 and pos_now == 0 and entry_idx is not None:
        exit_idx = df.index[i]
        exit_price = df["Close"].iloc[i]
        gross_ret = exit_price / entry_price - 1
        net_ret = gross_ret - 2 * cost_per_trade

        trades.append({
            "entry_date": entry_idx,
            "exit_date": exit_idx,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "days_held": (exit_idx - entry_idx).days
        })

        entry_idx = None
        entry_price = None

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
    "Close", "zscore60", "range_regime",
    "dominant_period", "power_10_20", "power_20_40", "power_40_80",
    "power_low_ratio", "spectral_ok", "entry_long", "exit_long", "position"
]
print(df[cols].tail(15))

if len(trades_df) > 0:
    print("\n=== Recent Trades ===")
    print(trades_df.tail(10))
else:
    print("\nNo completed trades found with current parameters.")

# ----------------------------
# Plots
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"], label="MGM Close")
plt.plot(df.index, df["ma60"], label="60D Mean")
entries = df.index[df["entry_long"].fillna(False)]
exits = df.index[df["exit_long"].fillna(False)]
plt.scatter(entries, df.loc[entries, "Close"], marker="^", s=60, label="Entry setup")
plt.scatter(exits, df.loc[exits, "Close"], marker="v", s=60, label="Exit setup")
plt.title("MGM Price, 60D Mean, and Signal Setups")
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
