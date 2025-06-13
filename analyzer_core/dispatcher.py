# analyzer_core/dispatcher.py
"""
Core dispatcher that orchestrates the analysis workflow using Strategy Engine
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .trigger_logic import get_trigger_decision
from .models.tool_models import AvailableTools, ToolResult
from .models.analysis_models import AnalysisContext
from .tools import get_tool_registry
from .analyzers import InitialAssessor, ResultAnalyzer, FinalAnalyzer
from .actions import ActionExecutor

logger = logging.getLogger(__name__)


class IntelligentDispatcher:
    """Core dispatcher that orchestrates the analysis workflow using Strategy Engine"""    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_assessor = InitialAssessor()
        self.result_analyzer = ResultAnalyzer()
        self.final_analyzer = FinalAnalyzer()
        self.action_executor = ActionExecutor(config)
        self.tool_registry = get_tool_registry()
        
        # Initialize Strategy Engine
        from .strategies.strategy_engine import StrategyEngine
        self.strategy_engine = StrategyEngine()
        
        # Initialize analysis context
        self.analysis_context = AnalysisContext()
        
        logger.info(f"Initialized Intelligent Dispatcher with Strategy Engine for issue #{config.get('issue_number')}")
    
    async def analyze(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main analysis orchestration method using Strategy Engine"""
        try:
            logger.info("Starting intelligent analysis workflow with Strategy Engine...")
            
            # Check trigger logic
            trigger_decision = get_trigger_decision(issue_data, comment_data)
            self.analysis_context.trigger_decision = {
                "should_trigger": trigger_decision.should_trigger,
                "reason": trigger_decision.reason,
                "trigger_type": trigger_decision.trigger_type,
                "confidence": trigger_decision.confidence
            }
            
            if not trigger_decision.should_trigger:
                logger.info(f"Not triggering: {trigger_decision.reason}")
                return self._convert_context_to_dict()
            
            # Store data in context
            self.analysis_context.issue_data = issue_data
            if comment_data:
                self.analysis_context.comment_data = comment_data
            
            # Step 1: Execute Strategy Engine
            logger.info("Step 1: Execute Strategy Engine")
            strategy_result = await self.strategy_engine.execute_strategy(
                trigger_decision, issue_data, comment_data
            )
            
            # Extract strategy recommendations
            selected_tools = strategy_result.get("selected_tools", [])
            customized_prompts = strategy_result.get("customized_prompts", {})
            context_analysis = strategy_result.get("context_analysis", {})
            
            # Store strategy result in context
            self.analysis_context.strategy_result = strategy_result
            
            # Step 2: Execute selected tools with strategy-informed approach
            logger.info("Step 2: Execute strategy-selected tools")
            # Convert tool names to AvailableTools enum values
            tool_enums = []
            for tool_name in selected_tools:
                try:
                    tool_enum = AvailableTools(tool_name)
                    tool_enums.append(tool_enum)
                except ValueError:
                    logger.warning(f"Unknown tool: {tool_name}")
            
            tool_results = await self._execute_tools(tool_enums, issue_data, comment_data)
            self.analysis_context.tool_results.extend([self._tool_result_to_dict(r) for r in tool_results])
            
            # Step 3: Generate final analysis using strategy-customized prompts
            logger.info("Step 3: Generate strategy-informed final analysis")
            final_analysis = await self.final_analyzer.generate(
                issue_data, 
                tool_results,
                comment_data,
                self.config.get("event_name", ""),
                self.config.get("event_action", ""),
                customized_prompts  # Pass strategy-customized prompts
            )
            self.analysis_context.final_analysis = final_analysis
            
            # Step 4: Execute strategy-recommended actions
            logger.info("Step 4: Execute strategy-recommended actions")
            # Get actions from strategy if available
            strategy_actions = await self._get_strategy_actions(strategy_result, final_analysis, context_analysis)
            
            # Execute actions
            actions = await self.action_executor.execute(final_analysis, issue_data, strategy_actions)
            self.analysis_context.actions_taken = actions
            
            logger.info("Strategy-driven analysis workflow completed successfully")
            return self._convert_context_to_dict()
            
        except Exception as e:
            logger.error(f"Strategy-driven analysis error: {str(e)}")
            self.analysis_context.error = str(e)
            return self._convert_context_to_dict()
    
    async def _execute_tools(self, tools: List[AvailableTools], 
                           issue_data: Dict[str, Any],
                           comment_data: Optional[Dict[str, Any]]) -> List[ToolResult]:
        """Execute selected tools"""
        results = []
        for tool_enum in tools:
            tool = self.tool_registry.get(tool_enum)
            if tool:
                try:
                    logger.info(f"Executing tool: {tool_enum.value}")
                    data = await tool.execute(issue_data, comment_data)
                    results.append(ToolResult(
                        tool=tool_enum,
                        success=True,
                        data=data,
                        confidence=data.get('confidence', 0.7)
                    ))
                except Exception as e:
                    logger.error(f"Tool {tool_enum.value} failed: {str(e)}")
                    results.append(ToolResult(
                        tool=tool_enum,
                        success=False,
                        data={},
                        error_message=str(e)
                    ))
            else:
                logger.warning(f"Tool {tool_enum.value} not available in registry")
                results.append(ToolResult(
                    tool=tool_enum,
                    success=False,
                    data={},
                    error_message=f"Tool {tool_enum.value} not implemented"
                ))
        return results
    
    async def _get_strategy_actions(self, strategy_result: Dict[str, Any], 
                                  final_analysis: Dict[str, Any],
                                  context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get actions recommended by the strategy based on analysis results
        """
        try:
            strategy_name = strategy_result.get("strategy_name", "unknown")
            
            # Get the strategy instance
            if strategy_name in self.strategy_engine.strategies:
                strategy = self.strategy_engine.strategies[strategy_name]
                
                # Get strategy-recommended actions
                strategy_actions = await strategy.recommend_actions(final_analysis, context_analysis)
                logger.info(f"Strategy {strategy_name} recommended {len(strategy_actions)} actions")
                return strategy_actions
            else:
                logger.warning(f"Strategy {strategy_name} not found for action recommendations")
                return []
                
        except Exception as e:
            logger.error(f"Error getting strategy actions: {str(e)}")
            return []
    
    def _convert_context_to_dict(self) -> Dict[str, Any]:
        """Convert analysis context to dictionary for JSON serialization"""
        return {
            "timestamp": self.analysis_context.timestamp,
            "issue_data": self.analysis_context.issue_data,
            "comment_data": self.analysis_context.comment_data,
            "trigger_decision": self.analysis_context.trigger_decision,
            "strategy_result": self.analysis_context.strategy_result,
            "decision_history": self.analysis_context.decision_history,
            "tool_results": self.analysis_context.tool_results,
            "final_analysis": self.analysis_context.final_analysis,
            "actions_taken": self.analysis_context.actions_taken,
            "error": self.analysis_context.error
        }
    
    def _decision_to_dict(self, decision) -> Dict[str, Any]:
        """Convert DecisionStep to dictionary"""
        return {
            "reasoning": decision.reasoning,
            "selected_tools": [tool.value for tool in decision.selected_tools],
            "expected_outcome": decision.expected_outcome,
            "priority": decision.priority,
            "user_info_assessment": decision.user_info_assessment
        }
    
    def _tool_result_to_dict(self, result: ToolResult) -> Dict[str, Any]:
        """Convert ToolResult to dictionary"""
        return {
            "tool": result.tool.value,
            "success": result.success,
            "data": result.data,
            "confidence": result.confidence,
            "error_message": result.error_message
        }
