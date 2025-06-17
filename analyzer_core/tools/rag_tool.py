# analyzer_core/tools/rag_tool.py
"""
RAG search tool for finding relevant information
"""

import re
import json
from typing import Dict, Any, Optional
from .base_tool import BaseTool
from RAG.rag_helper import query_vectordb_for_regression
from ..utils.json_utils import extract_json_from_llm_response


class RAGSearchTool(BaseTool):
    """RAG search tool for finding relevant information"""
    
    def __init__(self):
        super().__init__("rag_search")

    async def execute(self, issue_data: Dict[str, Any],
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute RAG search for relevant information"""        
        try:
            if not self.validate_input(issue_data):
                return {"error": "Invalid input data", "confidence": 0.0}
            
            issue_content = self.get_issue_content(issue_data, comment_data)
            
            context = query_vectordb_for_regression(
                issue_title=issue_data.get('title', ''),
                issue_body=issue_data.get('body', ''),
                n_results=5  # Increase to 5 results for more information
            )
            
            # Calculate actual result count
            results_count = self._count_rag_results(context)
            
            # Generate intelligent usage suggestions            
            use_suggestion = await self._generate_use_suggestion(
                issue_data=issue_data,
                rag_context=context,
                results_count=results_count
            )
            
            # Calculate dynamic confidence
            confidence = self._calculate_confidence(results_count, context)
            
            return {
                "context": context,
                "results_count": results_count,
                "is_successful": 1 if results_count > 0 else 0,
                "confidence": confidence,
                "use_suggestion": use_suggestion
            }        
        except Exception as e:
            self.logger.error(f"RAG search failed: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "confidence": 0.0}
    
    def _count_rag_results(self, context: str) -> int:
        """Calculate the actual number of issues returned by RAG search"""        
        if not context or context == "No similar issues found in the database.":
            return 0
        
        # Count actual issue numbers by matching "## ISSUE #" pattern
        issue_matches = re.findall(r'## ISSUE #\d+:', context)
        return len(issue_matches)
    
    async def _generate_use_suggestion(self, issue_data: Dict[str, Any],
                                      rag_context: str, results_count: int) -> Dict[str, Any]:
        """Generate intelligent usage suggestions based on RAG results"""
        if results_count == 0:
            return {
                "summary": "No similar issues found in the knowledge base",
                "relevance": "low",
                "actionable_insights": ["This appears to be a new or unique issue pattern"],                "recommended_approach": "Consider this as a potentially new issue that may require fresh investigation",
                "user_friendly_summary": "No similar historical cases found. This might be a new type of issue."
            }
        
        try:
            # Dynamic import of LLM to avoid circular imports
            from LLM.llm_provider import get_llm
            llm = get_llm(provider="azure")
            
            # Build intelligent analysis prompt            
            prompt = self._build_analysis_prompt(issue_data, rag_context, results_count)
            
            # Call LLM to generate suggestions
            from langchain_core.messages import HumanMessage
            
            response = await llm.agenerate([[HumanMessage(content=prompt)]])
            response_text = self._extract_response_text(response)
            
            # Parse JSON response
            suggestion_data = self._parse_json_response(response_text)
            
            # Validate and clean results
            return self._validate_and_clean_suggestion(suggestion_data, results_count)
            
        except Exception as e:            
            self.logger.warning(f"Failed to generate intelligent use suggestion: {e}")
            # Fallback processing: generate basic suggestions based on rules
            return self._generate_fallback_suggestion(rag_context, results_count)
    
    def _build_analysis_prompt(self, issue_data: Dict[str, Any], rag_context: str, results_count: int) -> str:
        """Build LLM analysis prompt"""
        return f"""
You are an expert technical assistant analyzing RAG search results to provide actionable insights for GitHub issue resolution.

**Current Issue:**
Title: {issue_data.get('title', 'N/A')}
Body: {issue_data.get('body', 'N/A')[:800]}{'...' if len(issue_data.get('body', '')) > 800 else ''}

**RAG Search Results ({results_count} similar issues found):**
{rag_context[:2000]}{'...' if len(rag_context) > 2000 else ''}

**Your Task:**
Analyze the RAG results and provide structured, actionable suggestions. Focus on:

1. **Pattern Recognition**: Identify common patterns between the current issue and historical cases
2. **Solution Extraction**: Extract concrete solutions, workarounds, or fixes that worked before
3. **Risk Assessment**: Note any pitfalls, failed approaches, or warnings from past cases
4. **Actionable Guidance**: Provide specific, implementable recommendations

**Response Format (JSON only):**
{{
    "summary": "1-2 sentence summary of what similar issues were found and their general nature",
    "relevance": "high|medium|low - assess how closely related the historical cases are to current issue",
    "actionable_insights": [
        "Specific insight 1 with concrete details",
        "Specific insight 2 with actionable guidance", 
        "Specific insight 3 with implementation steps"
    ],
    "recommended_approach": "Detailed, step-by-step recommendation based on successful historical resolutions",
    "user_friendly_summary": "A clear, jargon-free explanation for end users about what was found and what it means"
}}

**Guidelines:**
- Be specific and actionable, not generic
- Extract actual solutions/code/commands when available
- Mention specific issue numbers if they're particularly relevant
- Focus on proven solutions from the historical data
- Keep user_friendly_summary accessible to non-technical users
"""
    def _extract_response_text(self, response) -> str:
        """Extract text from LangChain response"""
        if hasattr(response.generations[0][0], 'text'):
            return response.generations[0][0].text
        elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
            return response.generations[0][0].message.content
        else:
            return str(response.generations[0][0])
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM using existing utility"""
        result = extract_json_from_llm_response(response_text, self.logger)
        if result is None:
            raise ValueError("No valid JSON found in response")
        return result
    
    def _validate_and_clean_suggestion(self, suggestion_data: Dict[str, Any], results_count: int) -> Dict[str, Any]:
        """Validate and clean suggestion data"""
        return {
            "summary": suggestion_data.get("summary", f"Found {results_count} similar issues in knowledge base"),
            "relevance": suggestion_data.get("relevance", "medium").lower(),
            "actionable_insights": suggestion_data.get("actionable_insights", [])[:4],  # Limit to 4 items
            "recommended_approach": suggestion_data.get("recommended_approach", "Review the similar issues above for potential solutions"),
            "user_friendly_summary": suggestion_data.get("user_friendly_summary", "Similar issues were found that might help resolve this problem")
        }    
    
    def _generate_fallback_suggestion(self, rag_context: str, results_count: int) -> Dict[str, Any]:
        """Generate fallback suggestions when LLM fails"""
        # Simple rule-based analysis
        insights = []
        approach = "Review the similar issues for potential solutions"
        
        if "solution" in rag_context.lower():
            insights.append("Previous solutions available in similar cases")
            approach = "Check the solution patterns from similar issues"
        
        if "error" in rag_context.lower():
            insights.append("Similar error patterns identified in historical cases")
        
        if "workaround" in rag_context.lower():
            insights.append("Workarounds available from previous cases")
        
        if not insights:
            insights = ["Similar issue patterns found", "Historical context available for reference"]
        
        return {
            "summary": f"Found {results_count} similar issues with potentially helpful information",
            "relevance": "medium" if results_count > 2 else "low",
            "actionable_insights": insights,
            "recommended_approach": approach,            
            "user_friendly_summary": f"Found {results_count} similar cases that might provide helpful context"
        }
    
    def _calculate_confidence(self, results_count: int, context: str) -> float:
        """Calculate dynamic confidence score"""
        if results_count == 0:
            return 0.3
        
        # Base confidence based on result count
        base_confidence = min(0.4 + (results_count * 0.15), 0.85)
        
        # Adjust based on content quality
        quality_bonus = 0.0
        if context:
            if "solution" in context.lower():
                quality_bonus += 0.1
            if "error" in context.lower():
                quality_bonus += 0.05
            if "reproduce" in context.lower():
                quality_bonus += 0.05
        
        return min(base_confidence + quality_bonus, 0.9)
