"""Tests for trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.models import SignalAction
from src.strategies.indicators import compute_ema, compute_rsi, detect_divergence
from src.strategies.multi_signal import MultiSignalStrategy


def _make_ohlcv(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    base = 50000.0
    prices = [base]
    for _ in range(n - 1):
        change = rng.normal(0, 100)
        if trend == "up":
            change += 30
        elif trend == "down":
            change -= 30
        prices.append(prices[-1] + change)

    closes = np.array(prices)
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes + rng.uniform(-100, 100, n)
    volumes = rng.uniform(100, 1000, n)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )
    return df


class TestIndicators:
    def test_rsi_range(self) -> None:
        df = _make_ohlcv(100)
        rsi = compute_rsi(df)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_ema_length(self) -> None:
        df = _make_ohlcv(100)
        ema = compute_ema(df, 20)
        assert len(ema) == len(df)

    def test_divergence_returns_valid(self) -> None:
        df = _make_ohlcv(100)
        rsi = compute_rsi(df)
        result = detect_divergence(df["close"], rsi)
        assert result in ("bullish", "bearish", "none")


class TestMultiSignalStrategy:
    def test_hold_on_insufficient_data(self) -> None:
        df = _make_ohlcv(10)
        strat = MultiSignalStrategy()
        signal = strat.analyze(df)
        assert signal.action == SignalAction.HOLD

    def test_produces_signal_with_enough_data(self) -> None:
        df = _make_ohlcv(200, trend="up")
        strat = MultiSignalStrategy()
        signal = strat.analyze(df)
        assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
        assert 0.0 <= signal.score <= 1.0

    def test_score_never_exceeds_one(self) -> None:
        for trend in ["up", "down"]:
            df = _make_ohlcv(200, trend=trend)
            strat = MultiSignalStrategy()
            signal = strat.analyze(df)
            assert signal.score <= 1.0
