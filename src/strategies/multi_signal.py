"""Multi-signal composite strategy combining RSI, EMA, MACD, and Bollinger Bands."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.core.models import Signal, SignalAction
from src.strategies.base import BaseStrategy
from src.strategies.indicators import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_volume_sma,
    detect_divergence,
)


class MultiSignalStrategy(BaseStrategy):
    """
    Combines multiple technical indicators with weighted scoring.

    Signal components and weights:
      - RSI (oversold/overbought zones):      0.15
      - EMA crossover (fast/slow):             0.20
      - MACD histogram + divergence:           0.20
      - Bollinger Band position + squeeze:     0.15
      - Volume confirmation:                   0.10
      - ADX trend strength filter:             0.05
      - Trend filter (EMA200):                 0.15

    Only produces a signal when the weighted score exceeds the threshold
    AND at least 3 indicators agree on direction.
    """

    WEIGHT_RSI = 0.15
    WEIGHT_EMA = 0.20
    WEIGHT_MACD = 0.20
    WEIGHT_BB = 0.15
    WEIGHT_VOLUME = 0.10
    WEIGHT_ADX = 0.05
    WEIGHT_TREND = 0.15

    MIN_AGREEING_INDICATORS = 3

    def __init__(
        self,
        rsi_length: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
        ema_trend: int = 50,
        ema_macro: int = 200,
        bb_length: int = 20,
        bb_std: float = 2.0,
        adx_length: int = 14,
        adx_threshold: float = 20.0,
    ) -> None:
        self._rsi_length = rsi_length
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._ema_trend = ema_trend
        self._ema_macro = ema_macro
        self._bb_length = bb_length
        self._bb_std = bb_std
        self._adx_length = adx_length
        self._adx_threshold = adx_threshold

    @property
    def name(self) -> str:
        return "multi_signal"

    def analyze(self, df: pd.DataFrame) -> Signal:
        min_bars = max(self._ema_macro, self._ema_trend) + 10
        if len(df) < min_bars:
            return Signal(action=SignalAction.HOLD, score=0.0, reason="Insufficient data")

        rsi_score, rsi_direction = self._score_rsi(df)
        ema_score, ema_direction = self._score_ema(df)
        macd_score, macd_direction = self._score_macd(df)
        bb_score, bb_direction = self._score_bollinger(df)
        vol_score = self._score_volume(df)
        adx_score, trending = self._score_adx(df)
        trend_score, trend_direction = self._score_trend(df)

        # Count how many directional indicators agree
        buy_votes = sum(
            1 for d in [rsi_direction, ema_direction, macd_direction, bb_direction, trend_direction]
            if d == "buy"
        )
        sell_votes = sum(
            1 for d in [rsi_direction, ema_direction, macd_direction, bb_direction, trend_direction]
            if d == "sell"
        )

        buy_score = (
            self.WEIGHT_RSI * (rsi_score if rsi_direction == "buy" else 0)
            + self.WEIGHT_EMA * (ema_score if ema_direction == "buy" else 0)
            + self.WEIGHT_MACD * (macd_score if macd_direction == "buy" else 0)
            + self.WEIGHT_BB * (bb_score if bb_direction == "buy" else 0)
            + self.WEIGHT_TREND * (trend_score if trend_direction == "buy" else 0)
            + self.WEIGHT_VOLUME * vol_score
            + self.WEIGHT_ADX * adx_score
        )

        sell_score = (
            self.WEIGHT_RSI * (rsi_score if rsi_direction == "sell" else 0)
            + self.WEIGHT_EMA * (ema_score if ema_direction == "sell" else 0)
            + self.WEIGHT_MACD * (macd_score if macd_direction == "sell" else 0)
            + self.WEIGHT_BB * (bb_score if bb_direction == "sell" else 0)
            + self.WEIGHT_TREND * (trend_score if trend_direction == "sell" else 0)
            + self.WEIGHT_VOLUME * vol_score
            + self.WEIGHT_ADX * adx_score
        )

        if not trending:
            buy_score *= 0.6
            sell_score *= 0.6

        reasons: list[str] = []
        if rsi_direction != "neutral":
            reasons.append(f"RSI({rsi_direction}:{rsi_score:.2f})")
        if ema_direction != "neutral":
            reasons.append(f"EMA({ema_direction}:{ema_score:.2f})")
        if macd_direction != "neutral":
            reasons.append(f"MACD({macd_direction}:{macd_score:.2f})")
        if bb_direction != "neutral":
            reasons.append(f"BB({bb_direction}:{bb_score:.2f})")
        if trend_direction != "neutral":
            reasons.append(f"TREND({trend_direction}:{trend_score:.2f})")
        reasons.append(f"VOL({vol_score:.2f})")
        reasons.append(f"ADX({'trending' if trending else 'ranging'}:{adx_score:.2f})")

        reason_str = " | ".join(reasons)

        # Require minimum indicator agreement for entry
        if buy_score > sell_score and buy_score > 0.3 and buy_votes >= self.MIN_AGREEING_INDICATORS:
            logger.debug("BUY signal: score={:.3f} votes={} — {}", buy_score, buy_votes, reason_str)
            return Signal(action=SignalAction.BUY, score=buy_score, reason=reason_str)
        elif sell_score > buy_score and sell_score > 0.3 and sell_votes >= self.MIN_AGREEING_INDICATORS:
            logger.debug("SELL signal: score={:.3f} votes={} — {}", sell_score, sell_votes, reason_str)
            return Signal(action=SignalAction.SELL, score=sell_score, reason=reason_str)

        return Signal(action=SignalAction.HOLD, score=max(buy_score, sell_score), reason=reason_str)

    def _score_rsi(self, df: pd.DataFrame) -> tuple[float, str]:
        rsi = compute_rsi(df, self._rsi_length)
        if rsi.empty or rsi.isna().all():
            return 0.0, "neutral"

        current = rsi.iloc[-1]
        prev = rsi.iloc[-2] if len(rsi) > 1 else current

        if current < 30:
            strength = (30 - current) / 30
            if current > prev:
                strength *= 1.3
            return min(strength, 1.0), "buy"
        elif current > 70:
            strength = (current - 70) / 30
            if current < prev:
                strength *= 1.3
            return min(strength, 1.0), "sell"
        elif current < 40 and prev < current:
            return 0.3, "buy"
        elif current > 60 and prev > current:
            return 0.3, "sell"

        return 0.0, "neutral"

    def _score_ema(self, df: pd.DataFrame) -> tuple[float, str]:
        ema_fast = compute_ema(df, self._ema_fast)
        ema_slow = compute_ema(df, self._ema_slow)
        ema_trend = compute_ema(df, self._ema_trend)

        if ema_fast.empty or ema_slow.empty or ema_trend.empty:
            return 0.0, "neutral"

        fast_now = ema_fast.iloc[-1]
        slow_now = ema_slow.iloc[-1]
        trend_now = ema_trend.iloc[-1]
        fast_prev = ema_fast.iloc[-2] if len(ema_fast) > 1 else fast_now
        slow_prev = ema_slow.iloc[-2] if len(ema_slow) > 1 else slow_now
        price = df["close"].iloc[-1]

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        score = 0.0
        direction = "neutral"

        if cross_up:
            score = 0.8
            direction = "buy"
            if price > trend_now:
                score = 1.0
        elif cross_down:
            score = 0.8
            direction = "sell"
            if price < trend_now:
                score = 1.0
        elif fast_now > slow_now and price > trend_now:
            diff_pct = (fast_now - slow_now) / slow_now
            score = min(diff_pct * 50, 0.5)
            direction = "buy"
        elif fast_now < slow_now and price < trend_now:
            diff_pct = (slow_now - fast_now) / slow_now
            score = min(diff_pct * 50, 0.5)
            direction = "sell"

        return score, direction

    def _score_macd(self, df: pd.DataFrame) -> tuple[float, str]:
        macd_df = compute_macd(df)
        if macd_df.empty:
            return 0.0, "neutral"

        cols = macd_df.columns
        hist_col = [c for c in cols if "h" in c.lower() or "hist" in c.lower()]
        macd_col = [c for c in cols if "macd" in c.lower() and "h" not in c.lower() and "s" not in c.lower()]

        if not hist_col:
            return 0.0, "neutral"

        histogram = macd_df[hist_col[0]]
        if histogram.isna().all():
            return 0.0, "neutral"

        hist_now = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2] if len(histogram) > 1 else hist_now

        rsi = compute_rsi(df, self._rsi_length)
        divergence = "none"
        if macd_col and not rsi.empty:
            macd_line = macd_df[macd_col[0]]
            divergence = detect_divergence(df["close"], macd_line)

        score = 0.0
        direction = "neutral"

        cross_up = hist_prev <= 0 < hist_now
        cross_down = hist_prev >= 0 > hist_now

        if cross_up:
            score = 0.8
            direction = "buy"
        elif cross_down:
            score = 0.8
            direction = "sell"
        elif hist_now > 0 and hist_now > hist_prev:
            score = 0.4
            direction = "buy"
        elif hist_now < 0 and hist_now < hist_prev:
            score = 0.4
            direction = "sell"

        if divergence == "bullish":
            score = min(score + 0.3, 1.0)
            direction = "buy"
        elif divergence == "bearish":
            score = min(score + 0.3, 1.0)
            direction = "sell"

        return score, direction

    def _score_bollinger(self, df: pd.DataFrame) -> tuple[float, str]:
        bb = compute_bollinger_bands(df, self._bb_length, self._bb_std)
        if bb.empty:
            return 0.0, "neutral"

        cols = bb.columns
        lower_col = [c for c in cols if "l" in c.lower() and "b" in c.lower()]
        upper_col = [c for c in cols if "u" in c.lower() and "b" in c.lower()]

        if not lower_col or not upper_col:
            return 0.0, "neutral"

        price = df["close"].iloc[-1]
        lower = bb[lower_col[0]].iloc[-1]
        upper = bb[upper_col[0]].iloc[-1]

        if pd.isna(lower) or pd.isna(upper):
            return 0.0, "neutral"

        band_width = upper - lower
        if band_width <= 0:
            return 0.0, "neutral"

        position_in_band = (price - lower) / band_width

        atr = compute_atr(df)
        squeeze = False
        if not atr.empty and not atr.isna().all():
            atr_now = atr.iloc[-1]
            if atr_now > 0:
                squeeze = (band_width / price) < (atr_now / price * 1.5)

        score = 0.0
        direction = "neutral"

        if position_in_band <= 0.05:
            score = 0.9
            direction = "buy"
        elif position_in_band <= 0.15:
            score = 0.6
            direction = "buy"
        elif position_in_band >= 0.95:
            score = 0.9
            direction = "sell"
        elif position_in_band >= 0.85:
            score = 0.6
            direction = "sell"

        if squeeze and score > 0:
            score = min(score * 1.2, 1.0)

        return score, direction

    def _score_volume(self, df: pd.DataFrame) -> float:
        vol_sma = compute_volume_sma(df, 20)
        if vol_sma.empty or vol_sma.isna().all():
            return 0.5

        current_vol = df["volume"].iloc[-1]
        avg_vol = vol_sma.iloc[-1]
        if avg_vol <= 0:
            return 0.5

        ratio = current_vol / avg_vol
        if ratio >= 2.0:
            return 1.0
        elif ratio >= 1.5:
            return 0.8
        elif ratio >= 1.0:
            return 0.6
        elif ratio >= 0.5:
            return 0.3
        return 0.1

    def _score_adx(self, df: pd.DataFrame) -> tuple[float, bool]:
        adx_df = compute_adx(df, self._adx_length)
        if adx_df.empty:
            return 0.5, True

        adx_col = [c for c in adx_df.columns if "adx" in c.lower() and "dm" not in c.lower()]
        if not adx_col:
            return 0.5, True

        adx_val = adx_df[adx_col[0]].iloc[-1]
        if pd.isna(adx_val):
            return 0.5, True

        trending = adx_val >= self._adx_threshold
        if adx_val >= 40:
            score = 1.0
        elif adx_val >= 25:
            score = 0.7
        elif adx_val >= 20:
            score = 0.5
        else:
            score = 0.2

        return score, trending

    def _score_trend(self, df: pd.DataFrame) -> tuple[float, str]:
        """EMA200 macro trend filter — only trade with the bigger trend."""
        ema_macro = compute_ema(df, self._ema_macro)
        if ema_macro.empty or ema_macro.isna().all():
            return 0.0, "neutral"

        price = df["close"].iloc[-1]
        macro_val = ema_macro.iloc[-1]
        if pd.isna(macro_val):
            return 0.0, "neutral"

        distance_pct = (price - macro_val) / macro_val

        if distance_pct > 0:
            score = min(abs(distance_pct) * 20, 1.0)
            return score, "buy"
        elif distance_pct < 0:
            score = min(abs(distance_pct) * 20, 1.0)
            return score, "sell"

        return 0.0, "neutral"
