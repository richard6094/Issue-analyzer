# analyzer_core/analyzers/result_analyzer.py
"""
Result analyzer component that determines if additional tools are needed
"""

import json
import logging
from typing import Dict, Any, Optional, List
from LLM.llm_provider import get_llm
from langchain_core.messages import HumanMessage

from ..models.tool_models import DecisionStep, AvailableTools, ToolResult
from ..utils.text_utils import prepare_issue_context

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyzes tool results and decides if additional tools are needed"""
    
    def __init__(self):
        self.llm = get_llm(provider="azure", temperature=0.1)
    
    async def analyze(self, issue_data: Dict[str, Any], 
                     tool_results: List[ToolResult],
                     comment_data: Optional[Dict[str, Any]] = None,
                     event_name: str = "", event_action: str = "") -> Optional[DecisionStep]:
        """
        Analyze tool results and decide if additional tools are needed
        
        Args:
            issue_data: Original issue data
            tool_results: Results from executed tools
            comment_data: Comment data if available
            event_name: GitHub event name
            event_action: GitHub event action
            
        Returns:
            Optional DecisionStep for additional tools
        """
        try:
            # Prepare results summary for LLM
            results_summary = self._prepare_results_summary(tool_results)
            issue_context = prepare_issue_context(issue_data, event_name, event_action)
            
            # Enhanced LLM prompt for tool results analysis
            analysis_prompt = f"""
You are analyzing tool execution results for a GitHub issue. Your task is to determine if additional tools are needed while avoiding redundant information requests.

## ANALYSIS PRINCIPLES:

### 1. INFORMATION SUFFICIENCY CHECK
- Can we provide a meaningful response with current information?
- Are there critical gaps that genuinely require additional data?
- Would additional tools significantly improve our analysis quality?

### 2. REDUNDANCY AVOIDANCE
- **NEVER** request tools that would ask for information already available
- **NEVER** suggest gathering information the user has already provided
- **FOCUS** on tools that analyze existing data rather than request new data

### 3. VALUE ASSESSMENT
- Will additional tools provide actionable insights?
- Is the potential benefit worth the computational cost?
- Are we approaching sufficient information for a helpful response?

## Current Analysis Context:

### Original Issue:
{issue_context}

### Tool Execution Results:
{results_summary}

### Available Additional Tools:
{self._get_tool_descriptions()}

## Decision Framework:

**CONTINUE ANALYSIS IF:**
- Critical technical gaps exist that prevent proper diagnosis
- Additional tools would significantly improve solution quality
- Current information is genuinely insufficient for helpful response

**STOP ANALYSIS IF:**
- We have sufficient information for meaningful guidance
- Additional tools would only provide marginal value
- Risk of redundant information requests exists

## Response Format:
{{
    "needs_more_tools": true/false,
    "reasoning": "Detailed reasoning about information sufficiency and tool necessity",
    "selected_tools": ["tool1", "tool2", ...],  // Only if needs_more_tools is true
    "expected_outcome": "What additional value these tools will provide",
    "priority": 1,
    "sufficiency_assessment": "complete|partial|insufficient"
}}

Focus on providing maximum value while respecting the user's time and existing contributions.
"""
            
            response = await self.llm.agenerate([[HumanMessage(content=analysis_prompt)]])
            response_text = self._extract_response_text(response)
            
            try:
                analysis_data = json.loads(response_text)
                
                if analysis_data.get("needs_more_tools", False):
                    selected_tools = [AvailableTools(tool) for tool in analysis_data.get("selected_tools", [])]
                    
                    return DecisionStep(
                        reasoning=analysis_data.get("reasoning", ""),
                        selected_tools=selected_tools,
                        expected_outcome=analysis_data.get("expected_outcome", ""),
                        priority=analysis_data.get("priority", 2)
                    )
                else:
                    logger.info("LLM determined no additional tools needed")
                    return None
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM analysis decision: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in tool results analysis: {str(e)}")
            return None
    
    def _extract_response_text(self, response) -> str:
        """Extract text from LangChain response"""
        if hasattr(response.generations[0][0], 'text'):
            return response.generations[0][0].text
        elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
            return response.generations[0][0].message.content
        else:
            return str(response.generations[0][0])
    
    def _prepare_results_summary(self, tool_results: List[ToolResult]) -> str:
        """Prepare a summary of tool results for LLM analysis"""
        summary_parts = []
        
        for result in tool_results:
            if result.success:
                summary_parts.append(f"""
**{result.tool.value}** (Success, Confidence: {result.confidence:.1%}):
{self._summarize_tool_data(result.data)}
""")
            else:
                summary_parts.append(f"""
**{result.tool.value}** (Failed): {result.error_message}
""")
        
        return "\n".join(summary_parts)
    
    def _summarize_tool_data(self, data: Dict[str, Any]) -> str:
        """Summarize tool data for LLM consumption"""
        if not data:
            return "No data returned"
        
        # Customize based on common data patterns
        summary_parts = []
        
        for key, value in data.items():
            if isinstance(value, list):
                summary_parts.append(f"- {key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"- {key}: {len(value)} properties")
            elif isinstance(value, str) and len(value) > 100:
                summary_parts.append(f"- {key}: {value[:100]}...")
            else:
                summary_parts.append(f"- {key}: {value}")
        
        return "\n".join(summary_parts)
    
    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools for LLM"""
        descriptions = {
            "rag_search": "Search through existing documentation and knowledge base for relevant information",
            "image_analysis": "Analyze screenshots, diagrams, or visual content in the issue",
            "regression_analysis": "Determine if this is a regression bug by analyzing code changes",
            "code_search": "Search through codebase for relevant code patterns or files",
            "similar_issues": "Find similar issues that have been resolved before",
            "documentation_lookup": "Look up relevant documentation sections",
            "template_generation": "Generate appropriate issue templates or response templates"
        }
        
        formatted_descriptions = []
        for tool, description in descriptions.items():
            formatted_descriptions.append(f"- **{tool}**: {description}")
        
        return "\n".join(formatted_descriptions)
