"""Fast scalping strategy for short timeframes (1m–5m)."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.core.models import Signal, SignalAction
from src.strategies.base import BaseStrategy
from src.strategies.indicators import (
    compute_bollinger_bands,
    compute_ema,
    compute_stoch_rsi,
    compute_volume_sma,
)


class ScalpingStrategy(BaseStrategy):
    """
    Scalping strategy optimised for 1m–5m timeframes.

    Uses tight Bollinger Bands + Stochastic RSI for entries,
    confirmed by EMA alignment and volume spikes.
    """

    def __init__(
        self,
        rsi_length: int = 7,
        bb_length: int = 14,
        bb_std: float = 2.0,
        ema_fast: int = 5,
        ema_slow: int = 13,
    ) -> None:
        self._rsi_length = rsi_length
        self._bb_length = bb_length
        self._bb_std = bb_std
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow

    @property
    def name(self) -> str:
        return "scalping"

    def analyze(self, df: pd.DataFrame) -> Signal:
        if len(df) < 30:
            return Signal(action=SignalAction.HOLD, score=0.0, reason="Insufficient data")

        signals: dict[str, float] = {"buy": 0.0, "sell": 0.0}
        reasons: list[str] = []

        self._check_stoch_rsi(df, signals, reasons)
        self._check_bb(df, signals, reasons)
        self._check_ema(df, signals, reasons)
        self._check_volume(df, signals, reasons)
        self._check_candle_pattern(df, signals, reasons)

        reason_str = " | ".join(reasons) if reasons else "No signals"

        if signals["buy"] > signals["sell"] and signals["buy"] >= 0.5:
            logger.debug("SCALP BUY: score={:.2f} — {}", signals["buy"], reason_str)
            return Signal(action=SignalAction.BUY, score=signals["buy"], reason=reason_str)
        elif signals["sell"] > signals["buy"] and signals["sell"] >= 0.5:
            logger.debug("SCALP SELL: score={:.2f} — {}", signals["sell"], reason_str)
            return Signal(action=SignalAction.SELL, score=signals["sell"], reason=reason_str)

        return Signal(action=SignalAction.HOLD, score=0.0, reason=reason_str)

    def _check_stoch_rsi(
        self, df: pd.DataFrame, signals: dict[str, float], reasons: list[str]
    ) -> None:
        stoch = compute_stoch_rsi(df, length=self._rsi_length)
        if stoch.empty:
            return

        k_col = [c for c in stoch.columns if "k" in c.lower()]
        d_col = [c for c in stoch.columns if "d" in c.lower()]
        if not k_col or not d_col:
            return

        k_now = stoch[k_col[0]].iloc[-1]
        d_now = stoch[d_col[0]].iloc[-1]
        k_prev = stoch[k_col[0]].iloc[-2] if len(stoch) > 1 else k_now
        d_prev = stoch[d_col[0]].iloc[-2] if len(stoch) > 1 else d_now

        if pd.isna(k_now) or pd.isna(d_now):
            return

        if k_now < 20 and k_prev <= d_prev and k_now > d_now:
            signals["buy"] += 0.35
            reasons.append(f"StochRSI bullish cross({k_now:.0f})")
        elif k_now > 80 and k_prev >= d_prev and k_now < d_now:
            signals["sell"] += 0.35
            reasons.append(f"StochRSI bearish cross({k_now:.0f})")

    def _check_bb(
        self, df: pd.DataFrame, signals: dict[str, float], reasons: list[str]
    ) -> None:
        bb = compute_bollinger_bands(df, self._bb_length, self._bb_std)
        if bb.empty:
            return

        cols = bb.columns
        lower_col = [c for c in cols if "l" in c.lower() and "b" in c.lower()]
        upper_col = [c for c in cols if "u" in c.lower() and "b" in c.lower()]

        if not lower_col or not upper_col:
            return

        price = df["close"].iloc[-1]
        lower = bb[lower_col[0]].iloc[-1]
        upper = bb[upper_col[0]].iloc[-1]

        if pd.isna(lower) or pd.isna(upper):
            return

        if price <= lower:
            signals["buy"] += 0.25
            reasons.append("BB lower touch")
        elif price >= upper:
            signals["sell"] += 0.25
            reasons.append("BB upper touch")

    def _check_ema(
        self, df: pd.DataFrame, signals: dict[str, float], reasons: list[str]
    ) -> None:
        ema_f = compute_ema(df, self._ema_fast)
        ema_s = compute_ema(df, self._ema_slow)

        if ema_f.empty or ema_s.empty:
            return

        if ema_f.iloc[-1] > ema_s.iloc[-1]:
            signals["buy"] += 0.15
            reasons.append("EMA bullish")
        else:
            signals["sell"] += 0.15
            reasons.append("EMA bearish")

    def _check_volume(
        self, df: pd.DataFrame, signals: dict[str, float], reasons: list[str]
    ) -> None:
        vol_sma = compute_volume_sma(df, 20)
        if vol_sma.empty or vol_sma.isna().all():
            return

        ratio = df["volume"].iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 0
        if ratio >= 1.5:
            signals["buy"] += 0.10
            signals["sell"] += 0.10
            reasons.append(f"HighVol({ratio:.1f}x)")

    def _check_candle_pattern(
        self, df: pd.DataFrame, signals: dict[str, float], reasons: list[str]
    ) -> None:
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        full_range = last["high"] - last["low"]
        if full_range == 0:
            return

        body_ratio = body / full_range

        if last["close"] > last["open"]:
            lower_wick = last["open"] - last["low"]
            if lower_wick > body * 2 and body_ratio < 0.3:
                signals["buy"] += 0.15
                reasons.append("Hammer")
        else:
            upper_wick = last["high"] - last["open"]
            if upper_wick > body * 2 and body_ratio < 0.3:
                signals["sell"] += 0.15
                reasons.append("ShootingStar")
