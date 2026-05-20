"""Tests for risk manager."""

from __future__ import annotations

from datetime import datetime, timezone

from src.core.config import Settings
from src.core.models import Position, Side, TradeResult
from src.core.risk_manager import RiskManager


def _make_settings(**overrides: object) -> Settings:
    defaults = {
        "api_key": "test",
        "api_secret": "test",
        "max_position_pct": 0.02,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
        "trailing_stop_pct": 0.01,
        "max_daily_loss_pct": 0.05,
        "max_open_positions": 3,
        "max_drawdown_pct": 0.10,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestRiskManager:
    def test_can_open_position_ok(self) -> None:
        rm = RiskManager(_make_settings(), 10_000.0)
        ok, reason = rm.can_open_position([])
        assert ok is True
        assert reason == "OK"

    def test_max_positions_reached(self) -> None:
        rm = RiskManager(_make_settings(max_open_positions=1), 10_000.0)
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            quantity=0.01,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        ok, reason = rm.can_open_position([pos])
        assert ok is False
        assert "Max open positions" in reason

    def test_daily_loss_halt(self) -> None:
        rm = RiskManager(_make_settings(max_daily_loss_pct=0.05), 10_000.0)
        loss = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=47500.0,
            quantity=0.01,
            pnl=-600.0,
            pnl_pct=-0.05,
            opened_at=datetime.now(timezone.utc),
        )
        rm.record_trade(loss)
        ok, reason = rm.can_open_position([])
        assert ok is False
        assert "Daily loss" in reason or "daily" in reason.lower()

    def test_max_drawdown_halt(self) -> None:
        rm = RiskManager(_make_settings(max_drawdown_pct=0.10), 10_000.0)
        loss = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=45000.0,
            quantity=0.01,
            pnl=-1100.0,
            pnl_pct=-0.10,
            opened_at=datetime.now(timezone.utc),
        )
        rm.record_trade(loss)
        ok, _ = rm.can_open_position([])
        assert ok is False
        assert rm.state.is_halted is True

    def test_consecutive_loss_cooldown(self) -> None:
        rm = RiskManager(_make_settings(), 100_000.0)
        for _ in range(5):
            loss = TradeResult(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry_price=50000.0,
                exit_price=49900.0,
                quantity=0.001,
                pnl=-10.0,
                pnl_pct=-0.002,
                opened_at=datetime.now(timezone.utc),
            )
            rm.record_trade(loss)
        ok, reason = rm.can_open_position([])
        assert ok is False
        assert "consecutive" in reason.lower()

    def test_position_size_calculation(self) -> None:
        rm = RiskManager(_make_settings(max_position_pct=0.02, stop_loss_pct=0.015, leverage=5), 10_000.0)
        qty = rm.calculate_position_size(50000.0, Side.LONG)
        assert qty > 0
        risk = 10_000 * 0.02
        expected = risk / (50000.0 * 0.015)
        assert abs(qty - round(expected, 6)) < 1e-6
        # Verify: SL hit should lose exactly risk_amount ($200)
        loss_at_sl = qty * 50000.0 * 0.015
        assert abs(loss_at_sl - risk) < 0.01

    def test_stop_loss_long(self) -> None:
        rm = RiskManager(_make_settings(stop_loss_pct=0.015), 10_000.0)
        sl = rm.compute_stop_loss(50000.0, Side.LONG)
        assert sl == round(50000.0 * 0.985, 2)

    def test_stop_loss_short(self) -> None:
        rm = RiskManager(_make_settings(stop_loss_pct=0.015), 10_000.0)
        sl = rm.compute_stop_loss(50000.0, Side.SHORT)
        assert sl == round(50000.0 * 1.015, 2)

    def test_take_profit_long(self) -> None:
        rm = RiskManager(_make_settings(take_profit_pct=0.03), 10_000.0)
        tp = rm.compute_take_profit(50000.0, Side.LONG)
        assert tp == round(50000.0 * 1.03, 2)

    def test_risk_reward_ratio(self) -> None:
        rm = RiskManager(_make_settings(stop_loss_pct=0.015, take_profit_pct=0.03), 10_000.0)
        assert rm.get_risk_reward_ratio() == 2.0

    def test_daily_reset_lifts_daily_halt(self) -> None:
        rm = RiskManager(_make_settings(max_daily_loss_pct=0.05), 10_000.0)
        loss = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=47500.0,
            quantity=0.01,
            pnl=-600.0,
            pnl_pct=-0.05,
            opened_at=datetime.now(timezone.utc),
        )
        rm.record_trade(loss)
        rm.can_open_position([])
        assert rm.state.is_halted is True
        rm.reset_daily()
        assert rm.state.is_halted is False
