"""
Analyzer Core Package

This package contains the core components for intelligent GitHub issue analysis:
- Trigger logic for determining when to activate
- Analysis tools for gathering information
- LLM-powered decision making
- GitHub action execution
"""

from .dispatcher import IntelligentDispatcher
from .trigger_logic import TriggerLogic, TriggerDecision, get_trigger_decision

__all__ = [
    'IntelligentDispatcher',
    'TriggerLogic',
    'TriggerDecision', 
    'get_trigger_decision'
]

__version__ = "1.0.0"
