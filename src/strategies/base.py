"""Abstract base for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.core.models import Signal


class BaseStrategy(ABC):
    """Every strategy must implement `analyze` and return a Signal."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Signal:
        """Analyze OHLCV data and return a trading signal."""
        ...
