# analyzer_core/strategies/__init__.py
"""
Strategy Layer for Context-Aware GitHub Issue Handling

This module implements the Strategy Layer in the high-level architecture:
Event → Agent → Strategy → Actions → Tools → Execute → Result

The strategy layer determines WHAT to do based on the context and trigger type,
then coordinates with the Actions layer to execute the plan.
"""

from .strategy_engine import StrategyEngine
from .base_strategy import BaseStrategy
# Re-enabling concrete strategies after fixing indentation issues
from .strategies import *

__all__ = [
    'StrategyEngine',
    'BaseStrategy'
]
