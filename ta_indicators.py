"""
ta_indicators.py

Technical-analysis indicators: RSI, Connors RSI, MACD and Williams %R.
There is no Streamlit code in this file, so every function can also be
called and tested from Spyder.

Conventions used here (they differ slightly between software packages):
  * RSI uses Wilder's smoothing. The first average is the simple mean of the
    first `length` price changes; later values use
    avg = (previous avg * (length - 1) + new value) / length.
  * EMA (used by MACD) is seeded with the simple mean of the first `span`
    values and then uses alpha = 2 / (span + 1).
  * Connors RSI is the mean of three components: RSI of the close, RSI of the
    up/down streak length, and the percent rank of the one-period return.
  * Percent rank is the share of the previous `length` returns that are
    strictly lower than the current return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Parameters (every default can be changed from the app sidebar)
# ---------------------------------------------------------------------------
@dataclass
class TaParams:
    rsi_length: int = 14
    rsi_upper: float = 70.0
    rsi_lower: float = 30.0
    crsi_rsi_length: int = 3
    crsi_streak_length: int = 2
    crsi_rank_length: int = 100
    crsi_upper: float = 90.0
    crsi_lower: float = 10.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    willr_length: int = 14
    willr_upper: float = -20.0
    willr_lower: float = -80.0


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------
def _seeded_recursion(values: pd.Series, length: int, alpha: float) -> pd.Series:
    """
    Recursive average: out[t] = alpha * x[t] + (1 - alpha) * out[t-1].
    The first output is the simple mean of the first `length` valid values.
    A missing value keeps the previous average.
    """
    x = values.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < length:
        return pd.Series(out, index=values.index)
    start = valid[length - 1]
    out[start] = np.nanmean(x[valid[0]:start + 1])
    for i in range(start + 1, len(x)):
        out[i] = out[i - 1] if np.isnan(x[i]) else alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=values.index)


def wilder_average(values: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1 / length)."""
    return _seeded_recursion(values, length, 1.0 / length)


def ema(values: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2 / (span + 1))."""
    return _seeded_recursion(values, span, 2.0 / (span + 1.0))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index on a 0 to 100 scale."""
    # Step 1: split price changes into gains and losses
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Step 2: smooth both with Wilder's average
    avg_gain = wilder_average(gain, length)
    avg_loss = wilder_average(loss, length)
    # Step 3: RSI = 100 - 100 / (1 + average gain / average loss)
    out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    out = out.where(avg_loss != 0, 100.0)                       # no losses: 100
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)  # no movement: 50
    return out.where(avg_gain.notna() & avg_loss.notna())       # warm-up stays empty


def streak(close: pd.Series) -> pd.Series:
    """
    Consecutive up days (+1, +2, ...) or down days (-1, -2, ...).
    An unchanged close resets the streak to 0. The first value is 0.
    """
    c = close.to_numpy(dtype=float)
    out = np.full(c.shape, np.nan)
    if len(c):
        out[0] = 0.0
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + 1.0 if out[i - 1] > 0 else 1.0
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - 1.0 if out[i - 1] < 0 else -1.0
        else:
            out[i] = 0.0
    return pd.Series(out, index=close.index)


def percent_rank(x: pd.Series, length: int = 100) -> pd.Series:
    """Share (0 to 100) of the previous `length` values that are strictly below the current value."""
    v = x.to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    if n > length:
        # Row j of the window view is v[j:j+length]; the window for day t is v[t-length:t]
        windows = sliding_window_view(v, length)[: n - length]
        current = v[length:]
        valid = ~np.isnan(windows).any(axis=1) & ~np.isnan(current)
        share = (windows < current[:, None]).mean(axis=1) * 100.0
        out[length:] = np.where(valid, share, np.nan)
    return pd.Series(out, index=x.index)


def connors_rsi(close: pd.Series, rsi_length: int = 3, streak_length: int = 2,
                rank_length: int = 100) -> pd.Series:
    """Mean of RSI(close), RSI(streak) and the percent rank of the one-period return."""
    price_part = rsi(close, rsi_length)
    streak_part = rsi(streak(close), streak_length)
    rank_part = percent_rank(close.pct_change() * 100.0, rank_length)
    return (price_part + streak_part + rank_part) / 3.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (fast EMA minus slow EMA), signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"MACD": line, "MACD signal": sig, "MACD histogram": line - sig})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Williams %R on a -100 to 0 scale."""
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()
    span = (highest - lowest).where(lambda s: s != 0)
    return -100.0 * (highest - close) / span


def prepare_ohlc(raw: pd.DataFrame, date_col, close_col, high_col=None, low_col=None,
                 asset_col=None, asset_value=None, dayfirst=False):
    """
    Build the price table used by the indicators from an uploaded table.
    Returns (prices, info). prices has the index 'date' and the columns
    close, and high and low when both were mapped.
    """
    # Step 1: optionally keep one asset of a multi-asset file
    df = raw
    if asset_col is not None and asset_value is not None:
        df = df.loc[df[asset_col].astype(str).str.strip() == str(asset_value)]

    # Step 2: standard column names and numeric prices
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst).dt.normalize(),
                        "close": pd.to_numeric(df[close_col], errors="coerce")})
    if high_col is not None and low_col is not None:
        out["high"] = pd.to_numeric(df[high_col], errors="coerce")
        out["low"] = pd.to_numeric(df[low_col], errors="coerce")

    # Step 3: drop unusable rows, sort and remove duplicate dates
    n_raw = len(out)
    out = out.dropna(subset=["date", "close"])
    out = out.loc[out["close"] > 0]
    out = out.drop_duplicates(subset="date", keep="last").sort_values("date").set_index("date")
    info = {"rows_read": n_raw, "rows_used": len(out), "inconsistent_ohlc": 0}

    # Step 4: report rows where high, low and close contradict each other
    if "high" in out.columns:
        bad = (out["high"] < out["low"]) | (out["close"] > out["high"]) | (out["close"] < out["low"])
        info["inconsistent_ohlc"] = int(bad.sum())
    return out, info


def compute_indicators(prices: pd.DataFrame, params: TaParams) -> pd.DataFrame:
    """
    prices needs a 'close' column and, for Williams %R, 'high' and 'low'.
    Returns one table with all indicators on the same date index.
    """
    close = prices["close"]
    parts = [
        rsi(close, params.rsi_length).rename("RSI"),
        connors_rsi(close, params.crsi_rsi_length, params.crsi_streak_length,
                    params.crsi_rank_length).rename("Connors RSI"),
        macd(close, params.macd_fast, params.macd_slow, params.macd_signal),
    ]
    if {"high", "low"}.issubset(prices.columns):
        parts.append(williams_r(prices["high"], prices["low"], close, params.willr_length)
                     .rename("Williams %R"))
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Plain-language reading of the latest values
# ---------------------------------------------------------------------------
def zone_label(value: float, lower: float, upper: float) -> str:
    """Overbought, oversold or neutral against the chosen thresholds."""
    if pd.isna(value):
        return "n/a"
    if value >= upper:
        return "overbought zone"
    if value <= lower:
        return "oversold zone"
    return "neutral"


def latest_snapshot(ind: pd.DataFrame, params: TaParams) -> pd.DataFrame:
    """One row per indicator with the latest value, the zone and the rule used."""
    rows = []

    def last_valid(col):
        s = ind[col].dropna()
        return (s.index[-1], float(s.iloc[-1])) if len(s) else (pd.NaT, np.nan)

    for col, lo, hi in [("RSI", params.rsi_lower, params.rsi_upper),
                        ("Connors RSI", params.crsi_lower, params.crsi_upper),
                        ("Williams %R", params.willr_lower, params.willr_upper)]:
        if col in ind.columns:
            date, val = last_valid(col)
            rows.append({"Indicator": col, "Date": date, "Value": val,
                         "Reading": zone_label(val, lo, hi),
                         "Rule": f"upper {hi:g} / lower {lo:g}"})

    hist = ind["MACD histogram"].dropna()
    if len(hist):
        side = "MACD above signal line" if hist.iloc[-1] > 0 else (
            "MACD below signal line" if hist.iloc[-1] < 0 else "MACD equals signal line")
        sign = np.sign(hist)
        changes = sign[sign != sign.shift()].iloc[1:]
        cross = changes.index[-1] if len(changes) else pd.NaT
        rows.append({"Indicator": "MACD", "Date": hist.index[-1], "Value": float(ind["MACD"].dropna().iloc[-1]),
                     "Reading": side,
                     "Rule": "last crossing: " + (cross.strftime("%Y-%m-%d") if pd.notna(cross) else "none in sample")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-08)  New file. Added: TaParams, wilder_average, ema, rsi, streak,
#   percent_rank, connors_rsi, macd, williams_r, prepare_ohlc,
#   compute_indicators, zone_label, latest_snapshot.
