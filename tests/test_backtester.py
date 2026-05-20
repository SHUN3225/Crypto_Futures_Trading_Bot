"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.backtester import Backtester
from src.core.config import Settings
from src.strategies.multi_signal import MultiSignalStrategy


def _make_settings(**overrides: object) -> Settings:
    defaults = {
        "api_key": "test",
        "api_secret": "test",
        "mode": "backtest",
        "max_position_pct": 0.02,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
        "trailing_stop_pct": 0.01,
        "max_daily_loss_pct": 0.05,
        "max_open_positions": 3,
        "max_drawdown_pct": 0.10,
        "min_signal_score": 0.3,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _make_ohlcv(n: int = 300, trend: str = "up") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = 50000.0
    prices = [base]
    for _ in range(n - 1):
        change = rng.normal(0, 100)
        if trend == "up":
            change += 20
        elif trend == "down":
            change -= 20
        prices.append(max(prices[-1] + change, 100))

    closes = np.array(prices)
    highs = closes + rng.uniform(50, 300, n)
    lows = closes - rng.uniform(50, 300, n)
    lows = np.maximum(lows, 1.0)
    opens = closes + rng.uniform(-150, 150, n)
    volumes = rng.uniform(100, 1000, n)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )


class TestBacktester:
    def test_run_completes(self) -> None:
        settings = _make_settings()
        strategy = MultiSignalStrategy()
        bt = Backtester(strategy, settings, initial_balance=10_000.0)
        df = _make_ohlcv(300)
        metrics = bt.run(df)

        assert metrics.initial_balance == 10_000.0
        assert metrics.final_balance > 0
        assert metrics.total_trades >= 0
        assert metrics.max_drawdown >= 0

    def test_summary_output(self) -> None:
        settings = _make_settings()
        strategy = MultiSignalStrategy()
        bt = Backtester(strategy, settings, initial_balance=10_000.0)
        df = _make_ohlcv(300)
        metrics = bt.run(df)
        summary = metrics.summary()
        assert "BACKTEST RESULTS" in summary
        assert "Total Trades" in summary

    def test_balance_consistency(self) -> None:
        settings = _make_settings()
        strategy = MultiSignalStrategy()
        bt = Backtester(strategy, settings, initial_balance=10_000.0)
        df = _make_ohlcv(300, trend="up")
        metrics = bt.run(df)
        expected_final = 10_000.0 + sum(t.pnl for t in metrics.trades)
        assert abs(metrics.final_balance - expected_final) < 0.01

    def test_metrics_counts(self) -> None:
        settings = _make_settings()
        strategy = MultiSignalStrategy()
        bt = Backtester(strategy, settings, initial_balance=10_000.0)
        df = _make_ohlcv(300)
        metrics = bt.run(df)
        assert metrics.wins + metrics.losses == metrics.total_trades
        assert metrics.win_rate == (
            metrics.wins / metrics.total_trades if metrics.total_trades > 0 else 0.0
        )
