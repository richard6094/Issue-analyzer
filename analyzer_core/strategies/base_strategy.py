# analyzer_core/strategies/base_strategy.py
"""
Base Strategy Class for the Strategy Layer

The Strategy Layer sits between the Agent and Actions in our architecture:
Event → Agent → Strategy → Actions → Tools → Execute → Result
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Base class for all issue handling strategies
    
    Each strategy encapsulates the decision logic for:
    1. What tools to use for analysis
    2. How to customize prompts based on context
    3. What actions to recommend based on results
    """
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"{__name__}.{strategy_name}")
    
    @abstractmethod
    async def analyze_context(self, 
                            issue_data: Dict[str, Any],
                            comment_data: Optional[Dict[str, Any]] = None,
                            trigger_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the context and determine strategy approach
        
        Returns:
            Context analysis with strategy decisions
        """
        pass
    
    @abstractmethod
    async def select_tools(self, 
                         context_analysis: Dict[str, Any]) -> List[str]:
        """
        Select appropriate tools based on context analysis
        
        Returns:
            List of tool names to execute
        """
        pass
    
    @abstractmethod
    def customize_prompts(self, 
                        base_prompts: Dict[str, str],
                        context_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Customize LLM prompts based on strategy and context
        
        Returns:
            Customized prompts for this strategy
        """
        pass
    
    @abstractmethod
    async def recommend_actions(self, 
                              analysis_results: Dict[str, Any],
                              context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend actions based on analysis results and context
        
        Returns:
            List of recommended actions
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy"""
        return {
            "name": self.strategy_name,
            "description": self.__doc__ or f"Strategy: {self.strategy_name}",
            "capabilities": self._get_capabilities()
        }
    
    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities this strategy provides"""
        return [
            "context_analysis",
            "tool_selection", 
            "prompt_customization",
            "action_recommendation"
        ]
