# analyzer_core/strategies/strategy_engine.py
"""
Strategy Engine - Core of the Strategy Layer

The Strategy Engine is the central coordinator that:
1. Receives trigger decisions from the Agent layer
2. Selects appropriate strategies based on context
3. Coordinates strategy execution
4. Interfaces with the Actions layer
"""

import logging
from typing import Dict, Any, Optional, List
from ..models.tool_models import AvailableTools
from ..trigger_logic import TriggerDecision

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Central engine for strategy selection and execution
    
    Architecture: Event → Agent → [Strategy Engine] → Actions → Tools → Execute → Result
    """
    
    def __init__(self):
        self.strategies = {}
        self.default_strategy = None
        self._load_strategies()
        
    def _load_strategies(self):
        """Load and register all available strategies"""
        try:
            # Re-enabling strategy loading after fixing indentation issues
            from .strategies.issue_created import IssueCreatedStrategy
            from .strategies.comment_response import CommentResponseStrategy
            from .strategies.agent_mention import AgentMentionStrategy
            
            # Register strategies
            self.register_strategy("issue_created", IssueCreatedStrategy())
            self.register_strategy("owner_comment", CommentResponseStrategy())
            self.register_strategy("agent_mention", AgentMentionStrategy())
            
            # Set default strategy
            self.default_strategy = "issue_created"
            
            logger.info(f"Successfully loaded {len(self.strategies)} strategies")
            
            logger.info(f"Loaded {len(self.strategies)} strategies: {list(self.strategies.keys())}")
            
        except ImportError as e:
            logger.warning(f"Some strategies could not be loaded: {e}")
            # Create minimal fallback strategy
            self._create_fallback_strategy()
            self._create_fallback_strategy()
    
    def _create_fallback_strategy(self):
        """Create a minimal fallback strategy if imports fail"""
        from .base_strategy import BaseStrategy
        
        class FallbackStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("fallback")
            
            async def analyze_context(self, issue_data, comment_data=None, trigger_context=None):
                return {
                    "strategy": "fallback",
                    "confidence": 0.5,
                    "approach": "basic_analysis"
                }
            
            async def select_tools(self, context_analysis):
                return ["rag_search", "similar_issues"]
            
            def customize_prompts(self, base_prompts, context_analysis):
                return base_prompts
            
            async def recommend_actions(self, analysis_results, context_analysis):
                return [{"action": "comment", "priority": 1}]
        
        self.register_strategy("fallback", FallbackStrategy())
        self.default_strategy = "fallback"
    
    def register_strategy(self, trigger_type: str, strategy):
        """Register a strategy for a specific trigger type"""
        self.strategies[trigger_type] = strategy
        logger.info(f"Registered strategy '{strategy.strategy_name}' for trigger '{trigger_type}'")
    
    async def execute_strategy(self, 
                             trigger_decision: TriggerDecision,
                             issue_data: Dict[str, Any],
                             comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute strategy based on trigger decision
        
        This is the main entry point for the Strategy Layer
        """
        try:
            logger.info(f"Executing strategy for trigger: {trigger_decision.trigger_type}")
            
            # Select strategy based on trigger type
            strategy = self._select_strategy(trigger_decision)
            
            # Prepare trigger context
            trigger_context = {
                "trigger_type": trigger_decision.trigger_type,
                "reason": trigger_decision.reason,
                "confidence": trigger_decision.confidence,
                "should_trigger": trigger_decision.should_trigger
            }
            
            # Execute strategy workflow
            return await self._execute_strategy_workflow(
                strategy, issue_data, comment_data, trigger_context
            )
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")
            return await self._handle_strategy_failure(e, issue_data, comment_data)
    
    def _select_strategy(self, trigger_decision: TriggerDecision):
        """Select appropriate strategy based on trigger decision"""
        trigger_type = trigger_decision.trigger_type
        
        if trigger_type in self.strategies:
            strategy = self.strategies[trigger_type]
            logger.info(f"Selected strategy: {strategy.strategy_name}")
            return strategy
        else:
            # Use default strategy
            default_strategy = self.strategies[self.default_strategy]
            logger.warning(f"No strategy for '{trigger_type}', using default: {default_strategy.strategy_name}")
            return default_strategy
    
    async def _execute_strategy_workflow(self, 
                                       strategy,
                                       issue_data: Dict[str, Any],
                                       comment_data: Optional[Dict[str, Any]],
                                       trigger_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete strategy workflow"""
        
        # Step 1: Analyze context
        logger.info("Step 1: Strategy context analysis")
        context_analysis = await strategy.analyze_context(
            issue_data, comment_data, trigger_context
        )
        
        # Step 2: Select tools
        logger.info("Step 2: Strategy tool selection")
        selected_tools = await strategy.select_tools(context_analysis)
        
        # Step 3: Customize prompts
        logger.info("Step 3: Strategy prompt customization")
        base_prompts = self._get_base_prompts()
        customized_prompts = strategy.customize_prompts(base_prompts, context_analysis)
        
        # Step 4: Prepare strategy result
        strategy_result = {
            "strategy_name": strategy.strategy_name,
            "context_analysis": context_analysis,
            "selected_tools": selected_tools,
            "customized_prompts": customized_prompts,
            "trigger_context": trigger_context,
            "timestamp": self._get_timestamp()
        }
        
        logger.info(f"Strategy workflow completed: {len(selected_tools)} tools selected")
        return strategy_result
    
    async def _handle_strategy_failure(self, 
                                     error: Exception,
                                     issue_data: Dict[str, Any],
                                     comment_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle strategy execution failures with fallback"""
        logger.error(f"Strategy failed, using emergency fallback: {str(error)}")
        
        return {
            "strategy_name": "emergency_fallback",
            "context_analysis": {"error": str(error), "confidence": 0.1},
            "selected_tools": ["rag_search"],
            "customized_prompts": self._get_base_prompts(),
            "trigger_context": {"trigger_type": "error", "reason": "strategy_failure"},
            "timestamp": self._get_timestamp(),
            "error": str(error)
        }
    
    def _get_base_prompts(self) -> Dict[str, str]:
        """Get base prompts for LLM interactions"""
        return {
            "analysis": "You are analyzing a GitHub issue. Provide helpful insights.",
            "final_response": "Generate a helpful response for the GitHub issue."
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available strategies"""
        return {
            trigger_type: strategy.get_strategy_info()
            for trigger_type, strategy in self.strategies.items()
        }
