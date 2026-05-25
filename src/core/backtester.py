"""Backtesting engine for strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.core.config import Settings
from src.core.models import Position, Side, Signal, SignalAction, TradeResult
from src.core.risk_manager import RiskManager
from src.strategies.base import BaseStrategy


@dataclass
class BacktestMetrics:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    final_balance: float = 0.0
    initial_balance: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    trades: list[TradeResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def roi(self) -> float:
        return (
            (self.final_balance - self.initial_balance) / self.initial_balance
            if self.initial_balance > 0
            else 0.0
        )

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "         BACKTEST RESULTS",
            "=" * 50,
            f"  Initial Balance:   ${self.initial_balance:,.2f}",
            f"  Final Balance:     ${self.final_balance:,.2f}",
            f"  Total PnL:         ${self.total_pnl:,.2f} ({self.roi:.2%})",
            f"  Max Drawdown:      {self.max_drawdown:.2%}",
            "",
            f"  Total Trades:      {self.total_trades}",
            f"  Wins:              {self.wins} ({self.win_rate:.1%})",
            f"  Losses:            {self.losses}",
            f"  Best Trade:        ${self.best_trade:,.2f}",
            f"  Worst Trade:       ${self.worst_trade:,.2f}",
            f"  Avg Win:           ${self.avg_win:,.2f}",
            f"  Avg Loss:          ${self.avg_loss:,.2f}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class Backtester:
    """Walk-forward backtester that simulates the full trading loop."""

    def __init__(
        self,
        strategy: BaseStrategy,
        settings: Settings,
        initial_balance: float = 10_000.0,
        atr_sl_mult: float | None = None,
        atr_tp_mult: float | None = None,
    ) -> None:
        self._strategy = strategy
        self._settings = settings
        self._initial_balance = initial_balance
        self._atr_sl_mult = atr_sl_mult
        self._atr_tp_mult = atr_tp_mult

    # Slippage applied on top of fees (entry + exit)
    SLIPPAGE_PCT = 0.0005  # 0.05% per side

    def run(self, df: pd.DataFrame, min_signal_score: float | None = None) -> BacktestMetrics:
        threshold = min_signal_score if min_signal_score is not None else self._settings.min_signal_score
        risk = RiskManager(self._settings, self._initial_balance)
        position: Position | None = None
        pending_signal: Signal | None = None  # signal waiting for next-bar entry
        metrics = BacktestMetrics(
            initial_balance=self._initial_balance,
            final_balance=self._initial_balance,
            peak_balance=self._initial_balance,
        )
        pnl_series: list[float] = []

        lookback = 210
        last_date = ""
        for i in range(lookback, len(df)):
            current_bar = df.iloc[i]
            current_price = float(current_bar["close"])
            high = float(current_bar["high"])
            low = float(current_bar["low"])
            open_price = float(current_bar["open"])
            bar_time = df.index[i]
            if not isinstance(bar_time, datetime):
                bar_time = datetime.now(timezone.utc)
            else:
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)

            # Daily reset (same as live bot)
            today = bar_time.strftime("%Y-%m-%d")
            if today != last_date:
                risk.reset_daily()
                last_date = today

            # 1) Execute pending entry at this bar's open (no look-ahead)
            if pending_signal is not None and position is None:
                can_open, _ = risk.can_open_position([])
                if can_open:
                    entry_price = self._apply_slippage(open_price, pending_signal.action)
                    position = self._open_position(
                        pending_signal, entry_price, risk, bar_time, pending_signal.atr_value,
                    )
                pending_signal = None

            # 2) Manage open position — SL checked before TP (conservative)
            if position is not None:
                close, reason = self._check_exit(position, high, low)
                if close:
                    raw_exit = self._get_exit_price(position, high, low, reason)
                    exit_action = SignalAction.SELL if position.side == Side.LONG else SignalAction.BUY
                    exit_price = self._apply_slippage(raw_exit, exit_action)
                    result = self._close_position(position, exit_price, bar_time, reason)
                    risk.record_trade(result)
                    metrics.trades.append(result)
                    pnl_series.append(result.pnl)
                    position = None
                else:
                    position.update_trailing(current_price, self._settings.trailing_stop_pct, atr_trailing_mult=0.5)

            # 3) Generate signal for NEXT bar entry (no same-bar entry)
            if position is None and pending_signal is None:
                window = df.iloc[: i + 1]
                signal = self._strategy.analyze(window)
                if signal.score >= threshold and signal.action != SignalAction.HOLD:
                    pending_signal = signal

        if position is not None:
            final_price = float(df.iloc[-1]["close"])
            result = self._close_position(
                position, final_price, df.index[-1] if isinstance(df.index[-1], datetime)
                else datetime.now(timezone.utc), "backtest_end"
            )
            risk.record_trade(result)
            metrics.trades.append(result)
            pnl_series.append(result.pnl)

        self._compute_metrics(metrics, pnl_series, risk)
        return metrics

    def _apply_slippage(self, price: float, action: SignalAction) -> float:
        """Worsen the fill price by slippage."""
        if action == SignalAction.BUY:
            return price * (1 + self.SLIPPAGE_PCT)
        return price * (1 - self.SLIPPAGE_PCT)

    def _check_exit(
        self, pos: Position, high: float, low: float
    ) -> tuple[bool, str]:
        if pos.side == Side.LONG:
            if low <= pos.stop_loss:
                return True, "stop_loss"
            if high >= pos.take_profit:
                return True, "take_profit"
        else:
            if high >= pos.stop_loss:
                return True, "stop_loss"
            if low <= pos.take_profit:
                return True, "take_profit"
        return False, ""

    @staticmethod
    def _get_exit_price(pos: Position, high: float, low: float, reason: str) -> float:
        if reason == "stop_loss":
            return pos.stop_loss
        if reason == "take_profit":
            return pos.take_profit
        return high if pos.side == Side.SHORT else low

    def _open_position(
        self,
        signal: Signal,
        price: float,
        risk: RiskManager,
        bar_time: datetime,
        atr_value: float | None = None,
    ) -> Position:
        side = Side.LONG if signal.action == SignalAction.BUY else Side.SHORT

        # Dynamic ATR-based SL/TP when multipliers are configured
        if self._atr_sl_mult and self._atr_tp_mult and atr_value and atr_value > 0:
            sl_dist = atr_value * self._atr_sl_mult
            tp_dist = atr_value * self._atr_tp_mult
            if side == Side.LONG:
                sl = round(price - sl_dist, 2)
                tp = round(price + tp_dist, 2)
            else:
                sl = round(price + sl_dist, 2)
                tp = round(price - tp_dist, 2)
            risk_amount = risk.state.current_balance * self._settings.max_position_pct
            fee_cost = price * 0.001
            effective_sl_dist = sl_dist - fee_cost
            qty = round(risk_amount / max(effective_sl_dist, sl_dist * 0.5), 6)
        else:
            qty = risk.calculate_position_size(price, side)
            sl = risk.compute_stop_loss(price, side)
            tp = risk.compute_take_profit(price, side)

        return Position(
            symbol=self._settings.symbol,
            side=side,
            entry_price=price,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            opened_at=bar_time,
            highest_price=price,
            lowest_price=price,
            atr_value=atr_value,
        )

    @staticmethod
    def _close_position(
        pos: Position, exit_price: float, bar_time: datetime, reason: str
    ) -> TradeResult:
        if pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        closed_at = bar_time
        if not isinstance(closed_at, datetime):
            closed_at = datetime.now(timezone.utc)
        elif closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)

        return TradeResult(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            opened_at=pos.opened_at,
            closed_at=closed_at,
            reason=reason,
        )

    def _compute_metrics(
        self,
        metrics: BacktestMetrics,
        pnl_series: list[float],
        risk: RiskManager,
    ) -> None:
        metrics.total_trades = len(metrics.trades)
        metrics.total_pnl = sum(t.pnl for t in metrics.trades)
        metrics.final_balance = self._initial_balance + metrics.total_pnl
        metrics.wins = sum(1 for t in metrics.trades if t.is_win)
        metrics.losses = metrics.total_trades - metrics.wins

        if metrics.trades:
            pnls = [t.pnl for t in metrics.trades]
            metrics.best_trade = max(pnls)
            metrics.worst_trade = min(pnls)
            win_pnls = [p for p in pnls if p > 0]
            loss_pnls = [p for p in pnls if p <= 0]
            metrics.avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
            metrics.avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

            total_wins = sum(win_pnls)
            total_losses = abs(sum(loss_pnls))
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        if pnl_series:
            returns = pd.Series(pnl_series) / self._initial_balance
            if returns.std() > 0:
                metrics.sharpe_ratio = float(
                    (returns.mean() / returns.std()) * (252**0.5)
                )

        balance_curve = [self._initial_balance]
        for pnl in pnl_series:
            balance_curve.append(balance_curve[-1] + pnl)
        peak = self._initial_balance
        max_dd = 0.0
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        metrics.max_drawdown = max_dd
        metrics.peak_balance = peak
