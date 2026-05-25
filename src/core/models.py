"""Domain models for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    action: SignalAction
    score: float  # 0.0 – 1.0 confidence
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    atr_value: float | None = None  # current ATR for dynamic SL/TP


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str = ""
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    atr_value: float | None = None

    def update_trailing(
        self,
        current_price: float,
        trailing_pct: float,
        atr_trailing_mult: float = 0.5,
    ) -> None:
        trail_dist: float
        if self.atr_value and self.atr_value > 0:
            trail_dist = self.atr_value * atr_trailing_mult
        else:
            trail_dist = current_price * trailing_pct

        if self.side == Side.LONG:
            if current_price > self.highest_price:
                self.highest_price = current_price
            if self.atr_value and self.atr_value > 0:
                new_stop = self.highest_price - trail_dist
            else:
                new_stop = self.highest_price * (1 - trailing_pct)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
        else:
            if self.lowest_price == 0.0 or current_price < self.lowest_price:
                self.lowest_price = current_price
            if self.atr_value and self.atr_value > 0:
                new_stop = self.lowest_price + trail_dist
            else:
                new_stop = self.lowest_price * (1 + trailing_pct)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop


@dataclass
class TradeResult:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    opened_at: datetime
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class RiskState:
    """Tracks daily and overall risk metrics."""

    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_losses: int = 0
    is_halted: bool = False
    halt_reason: str = ""

    @property
    def current_drawdown(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    @property
    def daily_win_rate(self) -> float:
        if self.trades_today == 0:
            return 0.0
        return self.wins_today / self.trades_today

    def record_trade(self, result: TradeResult) -> None:
        self.daily_pnl += result.pnl
        self.total_pnl += result.pnl
        self.current_balance += result.pnl
        self.trades_today += 1

        if result.is_win:
            self.wins_today += 1
            self.consecutive_losses = 0
        else:
            self.losses_today += 1
            self.consecutive_losses += 1

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
