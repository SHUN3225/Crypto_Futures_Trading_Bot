"""Bot configuration using pydantic-settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class ExchangeId(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"


class Settings(BaseSettings):
    """Global bot settings loaded from environment / .env file."""

    # --- Exchange ---
    exchange_id: ExchangeId = ExchangeId.BYBIT
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = Field(default="", description="API passphrase (OKX only)")
    testnet: bool = False

    # --- Trading ---
    mode: TradingMode = TradingMode.PAPER
    symbol: str = "BTC/USDT:USDT"
    symbols: str = Field(
        default="",
        description="Comma-separated list of symbols to trade (overrides symbol if set)",
    )
    timeframe: str = "15m"
    leverage: int = Field(default=5, ge=1, le=50)

    def get_symbols(self) -> list[str]:
        """Return list of symbols to trade."""
        if self.symbols.strip():
            return [s.strip() for s in self.symbols.split(",") if s.strip()]
        return [self.symbol]

    # --- Risk Management ---
    max_position_pct: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Max % of balance per trade"
    )
    stop_loss_pct: float = Field(default=0.015, ge=0.005, le=0.1)
    take_profit_pct: float = Field(default=0.02, ge=0.005, le=0.2)
    trailing_stop_pct: float = Field(default=0.008, ge=0.003, le=0.05)
    atr_sl_multiplier: float | None = Field(
        default=1.5, ge=0.5, le=5.0, description="ATR multiplier for dynamic SL (None=use fixed)"
    )
    atr_tp_multiplier: float | None = Field(
        default=2.0, ge=0.5, le=10.0, description="ATR multiplier for dynamic TP (None=use fixed)"
    )
    max_daily_loss_pct: float = Field(
        default=0.05, ge=0.01, le=0.2, description="Max daily drawdown before stopping"
    )
    max_open_positions: int = Field(default=3, ge=1, le=10)
    max_drawdown_pct: float = Field(
        default=0.10, ge=0.03, le=0.3, description="Max total drawdown before emergency stop"
    )

    # --- Strategy ---
    strategy_name: str = "multi_signal"
    min_signal_score: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum combined signal score to enter"
    )
    regime_filter: bool = Field(
        default=True, description="Enable regime filter (penalise counter-trend trades)"
    )

    # --- Polling ---
    poll_interval_seconds: int = Field(default=300, ge=5, le=3600)

    # --- Discord ---
    discord_webhook: str = Field(default="", description="Discord webhook URL for trade alerts")

    # --- Logging ---
    log_level: str = "INFO"
    log_file: Path = Path("logs/bot.log")

    model_config = {"env_prefix": "BOT_", "env_file": ".env", "env_file_encoding": "utf-8"}


def load_settings() -> Settings:
    return Settings()
