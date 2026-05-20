"""Entry-point for the futures trading bot."""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.core.backtester import Backtester
from src.core.bot import TradingBot
from src.core.config import TradingMode, load_settings
from src.core.exchange import ExchangeClient
from src.strategies.multi_signal import MultiSignalStrategy
from src.strategies.scalping import ScalpingStrategy
from src.utils.logger import setup_logger


def run_live(paper: bool = False) -> None:
    settings = load_settings()
    if paper:
        settings.mode = TradingMode.PAPER
    setup_logger(settings.log_level, settings.log_file)
    bot = TradingBot(settings)
    bot.start()


def run_backtest(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
    strategy: str = "multi_signal",
    initial_balance: float = 10_000.0,
) -> None:
    settings = load_settings()
    settings.mode = TradingMode.BACKTEST
    settings.symbol = symbol
    settings.timeframe = timeframe
    setup_logger(settings.log_level, settings.log_file)

    logger.info("Fetching historical data for backtest…")
    exchange = ExchangeClient(settings)
    df = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    logger.info("Loaded {} candles from {} to {}", len(df), df.index[0], df.index[-1])

    strat = (
        MultiSignalStrategy(regime_filter=settings.regime_filter)
        if strategy == "multi_signal"
        else ScalpingStrategy()
    )
    bt = Backtester(
        strat, settings, initial_balance,
        atr_sl_mult=settings.atr_sl_multiplier,
        atr_tp_mult=settings.atr_tp_multiplier,
    )
    metrics = bt.run(df)

    print(metrics.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Futures Trading Bot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("live", help="Run in LIVE mode (real money — use with caution)")
    sub.add_parser("paper", help="Run in PAPER trading mode (simulated)")

    bt = sub.add_parser("backtest", help="Run a historical backtest")
    bt.add_argument("--symbol", default="BTC/USDT:USDT")
    bt.add_argument("--timeframe", default="15m", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    bt.add_argument("--strategy", default="multi_signal", choices=["multi_signal", "scalping"])
    bt.add_argument("--balance", type=float, default=10_000.0)

    args = parser.parse_args()

    if args.command == "live":
        run_live(paper=False)
    elif args.command == "paper":
        run_live(paper=True)
    elif args.command == "backtest":
        run_backtest(args.symbol, args.timeframe, args.strategy, args.balance)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
