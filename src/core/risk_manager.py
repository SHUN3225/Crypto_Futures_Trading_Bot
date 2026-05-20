"""Risk management engine — the most critical component."""

from __future__ import annotations

from loguru import logger

from src.core.config import Settings
from src.core.models import Position, RiskState, Side, TradeResult


class RiskManager:
    """Enforces all risk rules before and after trades."""

    def __init__(self, settings: Settings, initial_balance: float) -> None:
        self._settings = settings
        self.state = RiskState(
            peak_balance=initial_balance,
            current_balance=initial_balance,
        )

    def can_open_position(self, open_positions: list[Position]) -> tuple[bool, str]:
        if self.state.is_halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        if len(open_positions) >= self._settings.max_open_positions:
            return False, f"Max open positions ({self._settings.max_open_positions}) reached"

        daily_loss_pct = abs(self.state.daily_pnl) / max(self.state.current_balance, 1)
        if self.state.daily_pnl < 0 and daily_loss_pct >= self._settings.max_daily_loss_pct:
            self.state.is_halted = True
            self.state.halt_reason = (
                f"Daily loss limit hit: {daily_loss_pct:.2%} >= {self._settings.max_daily_loss_pct:.2%}"
            )
            return False, self.state.halt_reason

        if self.state.current_drawdown >= self._settings.max_drawdown_pct:
            self.state.is_halted = True
            self.state.halt_reason = (
                f"Max drawdown hit: {self.state.current_drawdown:.2%} >= {self._settings.max_drawdown_pct:.2%}"
            )
            return False, self.state.halt_reason

        if self.state.consecutive_losses >= 5:
            return False, "5 consecutive losses — cooling down"

        return True, "OK"

    def calculate_position_size(self, entry_price: float, side: Side) -> float:
        risk_amount = self.state.current_balance * self._settings.max_position_pct
        stop_distance = entry_price * self._settings.stop_loss_pct

        if stop_distance <= 0:
            return 0.0

        # qty sized so that SL hit = risk_amount loss (leverage only affects margin)
        raw_qty = risk_amount / stop_distance
        return round(raw_qty, 6)

    def compute_stop_loss(self, entry_price: float, side: Side) -> float:
        if side == Side.LONG:
            return round(entry_price * (1 - self._settings.stop_loss_pct), 2)
        return round(entry_price * (1 + self._settings.stop_loss_pct), 2)

    def compute_take_profit(self, entry_price: float, side: Side) -> float:
        if side == Side.LONG:
            return round(entry_price * (1 + self._settings.take_profit_pct), 2)
        return round(entry_price * (1 - self._settings.take_profit_pct), 2)

    def should_close_position(self, position: Position, current_price: float) -> tuple[bool, str]:
        if position.side == Side.LONG:
            if current_price <= position.stop_loss:
                return True, "Stop-loss hit"
            if current_price >= position.take_profit:
                return True, "Take-profit hit"
        else:
            if current_price >= position.stop_loss:
                return True, "Stop-loss hit"
            if current_price <= position.take_profit:
                return True, "Take-profit hit"
        return False, ""

    def record_trade(self, result: TradeResult) -> None:
        self.state.record_trade(result)
        status = "WIN" if result.is_win else "LOSS"
        logger.info(
            "[{}] {} {} PnL: {:.2f} ({:.2%}) | Balance: {:.2f} | Drawdown: {:.2%}",
            status,
            result.side.value,
            result.symbol,
            result.pnl,
            result.pnl_pct,
            self.state.current_balance,
            self.state.current_drawdown,
        )

        if self.state.consecutive_losses >= 5:
            logger.warning("5 consecutive losses — entering cooldown")

    def reset_daily(self) -> None:
        self.state.reset_daily()
        if self.state.is_halted and "Daily" in self.state.halt_reason:
            self.state.is_halted = False
            self.state.halt_reason = ""
            logger.info("Daily halt lifted — new trading day")

    def get_risk_reward_ratio(self) -> float:
        return self._settings.take_profit_pct / self._settings.stop_loss_pct
