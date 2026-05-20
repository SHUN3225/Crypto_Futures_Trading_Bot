"""Tests for domain models."""

from __future__ import annotations

from datetime import datetime, timezone

from src.core.models import Position, RiskState, Side, TradeResult


class TestPosition:
    def test_trailing_stop_long(self) -> None:
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            quantity=0.01,
            stop_loss=49000.0,
            take_profit=52000.0,
            highest_price=50000.0,
        )
        pos.update_trailing(51000.0, 0.01)
        assert pos.highest_price == 51000.0
        expected_stop = 51000.0 * 0.99
        assert pos.stop_loss == expected_stop

    def test_trailing_stop_short(self) -> None:
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=Side.SHORT,
            entry_price=50000.0,
            quantity=0.01,
            stop_loss=51000.0,
            take_profit=48000.0,
            lowest_price=50000.0,
        )
        pos.update_trailing(49000.0, 0.01)
        assert pos.lowest_price == 49000.0
        expected_stop = 49000.0 * 1.01
        assert pos.stop_loss == expected_stop

    def test_trailing_stop_does_not_lower_long(self) -> None:
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            quantity=0.01,
            stop_loss=50490.0,
            take_profit=52000.0,
            highest_price=51000.0,
        )
        # Price drops — highest_price stays at 51000, stop should not lower
        pos.update_trailing(49800.0, 0.01)
        assert pos.stop_loss == 50490.0


class TestTradeResult:
    def test_is_win(self) -> None:
        result = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.01,
            pnl=10.0,
            pnl_pct=0.02,
            opened_at=datetime.now(timezone.utc),
        )
        assert result.is_win is True

    def test_is_loss(self) -> None:
        result = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=49000.0,
            quantity=0.01,
            pnl=-10.0,
            pnl_pct=-0.02,
            opened_at=datetime.now(timezone.utc),
        )
        assert result.is_win is False


class TestRiskState:
    def test_record_trade_win(self) -> None:
        state = RiskState(peak_balance=10000.0, current_balance=10000.0)
        result = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.01,
            pnl=100.0,
            pnl_pct=0.02,
            opened_at=datetime.now(timezone.utc),
        )
        state.record_trade(result)
        assert state.trades_today == 1
        assert state.wins_today == 1
        assert state.current_balance == 10100.0
        assert state.consecutive_losses == 0

    def test_record_trade_loss(self) -> None:
        state = RiskState(peak_balance=10000.0, current_balance=10000.0)
        result = TradeResult(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=50000.0,
            exit_price=49000.0,
            quantity=0.01,
            pnl=-100.0,
            pnl_pct=-0.02,
            opened_at=datetime.now(timezone.utc),
        )
        state.record_trade(result)
        assert state.losses_today == 1
        assert state.consecutive_losses == 1

    def test_drawdown_calculation(self) -> None:
        state = RiskState(peak_balance=10000.0, current_balance=9000.0)
        assert state.current_drawdown == 0.1

    def test_daily_reset(self) -> None:
        state = RiskState(
            peak_balance=10000.0,
            current_balance=9500.0,
            daily_pnl=-500.0,
            trades_today=5,
            wins_today=2,
            losses_today=3,
        )
        state.reset_daily()
        assert state.daily_pnl == 0.0
        assert state.trades_today == 0
        assert state.current_balance == 9500.0  # balance persists
