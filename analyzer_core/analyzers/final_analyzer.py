# analyzer_core/analyzers/final_analyzer.py
"""
Final analyzer component that generates comprehensive analysis and recommendations
"""

import json
import logging
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage

from ..models.tool_models import ToolResult
from ..models.analysis_models import FinalAnalysis
from ..utils.text_utils import prepare_issue_context
from ..utils.json_utils import extract_and_parse_json_response, clean_user_comment, extract_user_comment_from_response

logger = logging.getLogger(__name__)


class FinalAnalyzer:
    """Generates final analysis and recommendations using LLM"""
    
    def __init__(self, llm=None):
        """
        Initialize FinalAnalyzer with optional LLM instance
        
        Args:
            llm: Pre-configured LLM instance. If None, will create one when needed.
        """
        self._llm = llm
        self._llm_initialized = llm is not None
    
    def _get_llm(self):
        """Get or create LLM instance (lazy initialization)"""
        if not self._llm_initialized:
            # Only import when actually needed
            from LLM.llm_provider import get_llm
            self._llm = get_llm(provider="azure", temperature=0.1)
            self._llm_initialized = True
        return self._llm
    
    @property
    def llm(self):
        """LLM property with lazy initialization"""
        return self._get_llm()
    
    async def generate(self, issue_data: Dict[str, Any], 
                      tool_results: List[ToolResult],
                      comment_data: Optional[Dict[str, Any]] = None,
                      event_name: str = "", event_action: str = "",
                      customized_prompts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate final analysis and recommendations using LLM
        
        Args:
            issue_data: Original issue data
            tool_results: All tool results
            comment_data: Comment data if available
            event_name: GitHub event name
            event_action: GitHub event action
            customized_prompts: Strategy-customized prompts (optional)
            
        Returns:
            Final analysis with recommendations and actions
        """
        try:
            issue_context = prepare_issue_context(issue_data, event_name, event_action)
            results_summary = self._prepare_results_summary(tool_results)
            
            # Use strategy-customized prompts if available, otherwise use default
            if customized_prompts and "final_response" in customized_prompts:
                # Use strategy-customized final response prompt
                final_analysis_prompt = self._build_strategy_prompt(
                    customized_prompts["final_response"],
                    issue_context, 
                    results_summary
                )
            else:
                # Use default enhanced LLM prompt for final analysis
                final_analysis_prompt = self._build_default_prompt(issue_context, results_summary)
            
            # Use self.llm property which handles lazy initialization
            response = await self.llm.agenerate([[HumanMessage(content=final_analysis_prompt)]])
            response_text = self._extract_response_text(response)
            
            try:
                # First try to extract and parse JSON properly
                final_analysis = extract_and_parse_json_response(response_text)
                
                # Validate and clean the user_comment
                if "user_comment" in final_analysis:
                    final_analysis["user_comment"] = clean_user_comment(final_analysis["user_comment"])
                
                logger.info(f"Generated final analysis: {final_analysis.get('issue_type', 'unknown')} with {final_analysis.get('confidence', 0):.1%} confidence")
                return final_analysis
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse final analysis, using cleaned text response: {str(e)}")
                # Extract user comment from the raw response if JSON parsing fails
                cleaned_comment = extract_user_comment_from_response(response_text)
                return {
                    "issue_type": "unknown",
                    "severity": "medium",
                    "confidence": 0.5,
                    "summary": "Analysis completed but response parsing failed",
                    "detailed_analysis": response_text,
                    "user_comment": cleaned_comment
                }
                
        except Exception as e:
            logger.error(f"Error in final analysis generation: {str(e)}")
            return {
                "issue_type": "unknown",
                "severity": "medium", 
                "confidence": 0.0,
                "summary": f"Analysis failed: {str(e)}",
                "user_comment": "I encountered an error while analyzing this issue. Please try again or contact support."
            }
    
    def _build_strategy_prompt(self, strategy_prompt: str, issue_context: str, results_summary: str) -> str:
        """Build final analysis prompt using strategy-customized template"""
        return f"""
{strategy_prompt}

## Analysis Context:

### Original Issue:
{issue_context}

### Analysis Results:
{results_summary}

## Response Format:
{{
    "issue_type": "bug_report|feature_request|question|documentation|regression|security|performance",
    "severity": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "summary": "Brief summary acknowledging user's provided information and key findings",
    "detailed_analysis": "Detailed analysis based on user's actual contributions and tool results",
    "root_cause": "Likely root cause based on provided information",
    "recommended_labels": ["label1", "label2", ...],
    "recommended_actions": [
        {{
            "action": "add_label|assign_user|close_issue|request_info",
            "details": "Specific details for the action",
            "priority": 1
        }}
    ],
    "user_comment": "A helpful comment that builds on the user's provided information and offers genuine value"
}}

**CRITICAL**: Your response must demonstrate that you've carefully reviewed and understood the user's contributions.
        """.strip()
    
    def _build_default_prompt(self, issue_context: str, results_summary: str) -> str:
        """Build default final analysis prompt"""
        return f"""
You are a senior GitHub issue analyst providing comprehensive analysis and actionable recommendations.

## RESPONSE GUIDELINES:

### 1. ACKNOWLEDGE USER CONTRIBUTIONS
- **ALWAYS** acknowledge what the user has provided
- **REFERENCE** specific code samples, reproduction steps, or data they shared
- **SHOW** that you understand their effort in providing detailed information

### 2. ANALYSIS BASED ON PROVIDED INFORMATION
- Analyze the user's code samples, reproduction steps, and data
- Provide insights based on what they've actually shared
- Avoid generic responses that ignore their specific details

### 3. AVOID REDUNDANT REQUESTS
- **NEVER** ask for information the user has already provided
- **BUILD** upon their existing contributions
- **ENHANCE** their understanding rather than requesting basics

## Analysis Context:

### Original Issue:
{issue_context}

### Analysis Results:
{results_summary}

## Response Requirements:

### User Comment Guidelines:
Your user comment should:
1. **Start with acknowledgment**: "Thank you for providing [specific details they shared]..."
2. **Demonstrate understanding**: "Based on your code sample showing [specific detail]..."
3. **Provide value-added analysis**: "Looking at your reproduction steps, the issue appears to be..."
4. **Offer actionable next steps**: "To resolve this, I recommend..."
5. **Reference their specific context**: "Given your environment/code/setup..."

### Response Format:
{{
    "issue_type": "bug_report|feature_request|question|documentation|regression|security|performance",
    "severity": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "summary": "Brief summary acknowledging user's provided information and key findings",
    "detailed_analysis": "Detailed analysis based on user's actual contributions and tool results",
    "root_cause": "Likely root cause based on provided information",
    "recommended_labels": ["label1", "label2", ...],
    "recommended_actions": [
        {{
            "action": "add_label|assign_user|close_issue|request_info",
            "details": "Specific details for the action",
            "priority": 1
        }}
    ],
    "user_comment": "A helpful comment that builds on the user's provided information and offers genuine value"
}}

**CRITICAL**: Your response must demonstrate that you've carefully reviewed and understood the user's contributions. Generic responses that ignore their specific details are unacceptable.
        """.strip()
    
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