import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = "MGM"
df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)

# Flatten yfinance columns if they come back as MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close", "Volume"]].dropna().copy()

# Make absolutely sure these are Series, not 1-col DataFrames
close = df["Close"]
volume = df["Volume"]

df["log_ret"] = np.log(close).diff()

# Rolling z-score on price
lookback = 60
df["ma60"] = close.rolling(lookback).mean()
df["std60"] = close.rolling(lookback).std()
df["zscore60"] = (close - df["ma60"]) / df["std60"]

# Rolling FFT dominant cycle on detrended prices
window = 126  # ~6 months
dominant_periods = [np.nan] * len(df)
dominant_powers = [np.nan] * len(df)

close_vals = close.to_numpy()

for i in range(window, len(df)):
    x = close_vals[i - window:i].copy()

    # detrend with linear fit
    t = np.arange(window)
    coef = np.polyfit(t, x, 1)
    trend = coef[0] * t + coef[1]
    x_detrended = x - trend

    # demean + FFT
    x_detrended = x_detrended - x_detrended.mean()
    fft_vals = np.fft.rfft(x_detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(window, d=1)

    # ignore zero frequency
    power[0] = 0

    # convert frequency to period in trading days
    periods = np.full_like(freqs, np.nan, dtype=float)
    valid = freqs > 0
    periods[valid] = 1.0 / freqs[valid]

    # keep only plausible swing-trading cycles
    band = (periods >= 10) & (periods <= 80)

    if np.any(band):
        idx_local = np.argmax(power[band])
        idx_global = np.where(band)[0][idx_local]
        dominant_periods[i] = periods[idx_global]
        dominant_powers[i] = power[idx_global]

df["dominant_period"] = dominant_periods
df["dominant_power"] = dominant_powers

# Example naive signal
df["signal_long"] = (
    (df["zscore60"] < -1.5) &
    (df["dominant_period"].between(15, 60)) &
    (df["dominant_power"] > df["dominant_power"].rolling(60).median())
).astype(int)

# Simple next-5-day forward return for research
df["fwd_5d_ret"] = close.shift(-5) / close - 1

print(df[["Close", "zscore60", "dominant_period", "dominant_power", "signal_long", "fwd_5d_ret"]].tail(15))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Close"], label="MGM Close")
plt.plot(df.index, df["ma60"], label="60D Mean")
plt.legend()
plt.title("MGM Price and 60D Mean")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df.index, df["dominant_period"])
plt.title("Rolling Dominant Cycle Period (days)")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df.index, df["zscore60"])
plt.axhline(-1.5, linestyle="--")
plt.axhline(0, linestyle="--")
plt.axhline(1.5, linestyle="--")
plt.title("Rolling 60D Z-Score")
plt.show()
