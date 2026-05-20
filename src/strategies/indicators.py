"""Technical indicator calculations using pandas-ta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    result = ta.rsi(df["close"], length=length)
    if result is None:
        return pd.Series(dtype=float, index=df.index)
    return result


def compute_ema(df: pd.DataFrame, length: int = 20) -> pd.Series:
    result = ta.ema(df["close"], length=length)
    if result is None:
        return pd.Series(dtype=float, index=df.index)
    return result


def compute_sma(df: pd.DataFrame, length: int = 50) -> pd.Series:
    result = ta.sma(df["close"], length=length)
    if result is None:
        return pd.Series(dtype=float, index=df.index)
    return result


def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if result is None:
        return pd.DataFrame(index=df.index)
    return result


def compute_bollinger_bands(
    df: pd.DataFrame, length: int = 20, std: float = 2.0
) -> pd.DataFrame:
    result = ta.bbands(df["close"], length=length, std=std)
    if result is None:
        return pd.DataFrame(index=df.index)
    return result


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    result = ta.atr(df["high"], df["low"], df["close"], length=length)
    if result is None:
        return pd.Series(dtype=float, index=df.index)
    return result


def compute_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    result = ta.adx(df["high"], df["low"], df["close"], length=length)
    if result is None:
        return pd.DataFrame(index=df.index)
    return result


def compute_stoch_rsi(
    df: pd.DataFrame, length: int = 14, rsi_length: int = 14, k: int = 3, d: int = 3
) -> pd.DataFrame:
    result = ta.stochrsi(df["close"], length=length, rsi_length=rsi_length, k=k, d=d)
    if result is None:
        return pd.DataFrame(index=df.index)
    return result


def compute_volume_sma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return df["volume"].rolling(window=length).mean()


def detect_divergence(
    price: pd.Series, indicator: pd.Series, lookback: int = 20
) -> str:
    """Detect bullish/bearish divergence between price and indicator.

    Returns 'bullish', 'bearish', or 'none'.
    """
    if len(price) < lookback or len(indicator) < lookback:
        return "none"

    recent_price = price.iloc[-lookback:]
    recent_ind = indicator.iloc[-lookback:]

    price_min_idx = int(np.argmin(recent_price.values))
    price_max_idx = int(np.argmax(recent_price.values))

    if price_min_idx > lookback // 2:
        first_half_price_low = recent_price.iloc[: lookback // 2].min()
        second_half_price_low = recent_price.iloc[lookback // 2 :].min()
        first_half_ind_low = recent_ind.iloc[: lookback // 2].min()
        second_half_ind_low = recent_ind.iloc[lookback // 2 :].min()

        if second_half_price_low < first_half_price_low and second_half_ind_low > first_half_ind_low:
            return "bullish"

    if price_max_idx > lookback // 2:
        first_half_price_high = recent_price.iloc[: lookback // 2].max()
        second_half_price_high = recent_price.iloc[lookback // 2 :].max()
        first_half_ind_high = recent_ind.iloc[: lookback // 2].max()
        second_half_ind_high = recent_ind.iloc[lookback // 2 :].max()

        if (
            second_half_price_high > first_half_price_high
            and second_half_ind_high < first_half_ind_high
        ):
            return "bearish"

    return "none"
