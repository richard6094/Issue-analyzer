# analyzer_core/strategies/strategies/issue_created.py
"""
Issue Created Strategy

Handles new issue creation scenarios with comprehensive initial assessment.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class IssueCreatedStrategy(BaseStrategy):
    """
    Strategy for handling newly created issues
    
    Focus:
    - Comprehensive initial assessment
    - Information completeness evaluation
    - Appropriate tool selection for thorough analysis
    - Setting up proper tracking and categorization
    """
    
    def __init__(self):
        super().__init__("issue_created")
    
    async def analyze_context(self, 
                            issue_data: Dict[str, Any],
                            comment_data: Optional[Dict[str, Any]] = None,
                            trigger_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze context for a newly created issue using LLM-based analysis
        """
        
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        labels = [label.get('name', '') for label in issue_data.get('labels', [])]
        author = issue_data.get('user', {}).get('login', '')
        
        # Use LLM for context analysis with chain of thought
        context_analysis = await self._llm_analyze_issue_context(
            title, body, labels, author, trigger_context
        )
        
        # Add strategy metadata
        context_analysis.update({
            "strategy": "issue_created",            "author": author,
            "existing_labels": labels,
            "approach": "llm_driven_initial_assessment"
        })
        
        logger.info(f"LLM Context Analysis: {context_analysis.get('issue_type', 'unknown')} issue, "
                   f"{context_analysis.get('complexity', 'unknown')} complexity, "
                   f"{context_analysis.get('confidence', 0):.2f} confidence")
        return context_analysis
    
    async def select_tools(self, context_analysis: Dict[str, Any]) -> List[str]:        """
        Select tools using LLM-based reasoning
        """
        
        # Use LLM to select appropriate tools based on context analysis
        selected_tools = await self._llm_select_tools(context_analysis)
        
        logger.info(f"LLM selected {len(selected_tools)} tools for issue_created strategy: {selected_tools}")
        return selected_tools
    
    def customize_prompts(self, 
                        base_prompts: Dict[str, str],
                        context_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Customize prompts using LLM-based prompt engineering
        """
        
        # Use the context analysis to inform prompt customization
        issue_type = context_analysis.get("issue_type", "unknown")
        complexity = context_analysis.get("complexity", "medium")
        urgency = context_analysis.get("urgency", "medium")
        info_completeness = context_analysis.get("information_completeness", "partial")
        
        customized = base_prompts.copy()
        
        # Create context-aware analysis prompt
        customized["analysis"] = f"""
You are analyzing a newly created GitHub issue. Based on the context analysis, this is a {issue_type} issue with {complexity} complexity and {urgency} urgency.

Context from Strategy Analysis:
{self._format_context_for_prompt(context_analysis)}

Your analysis should:
1. Build upon the initial context analysis
2. Focus on providing actionable insights
3. Consider the issue type and complexity level
4. Respect the information the user has already provided

Guidelines:
- For {issue_type} issues: Focus on appropriate response patterns for this issue type
- Given {complexity} complexity: Adjust depth and thoroughness of analysis
- Information completeness is {info_completeness}: Handle accordingly

Base prompt: {base_prompts.get('analysis', '')}
        """.strip()
        
        # Create context-aware final response prompt
        customized["final_response"] = f"""
You are responding to a newly created {issue_type} issue with {complexity} complexity.

Context Analysis Results:
{self._format_context_for_prompt(context_analysis)}

Response Guidelines:
1. Acknowledge the issue professionally and helpfully
2. Provide initial assessment based on the context analysis
3. Be specific and actionable in your guidance
4. Request additional information ONLY if genuinely needed
5. Set appropriate expectations based on complexity and urgency
6. Be welcoming, especially for new contributors

Tone and Approach:
- Professional but friendly
- Confident but not dismissive
- Helpful and constructive
- Appropriate for a {issue_type} with {complexity} complexity

Base prompt: {base_prompts.get('final_response', '')}
        """.strip()
        
        return customized
    
    async def recommend_actions(self, 
                              analysis_results: Dict[str, Any],
                              context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend actions using LLM-based reasoning
        """
        
        # Use LLM to recommend actions based on analysis results and context
        recommended_actions = await self._llm_recommend_actions(analysis_results, context_analysis)
        
        logger.info(f"LLM recommended {len(recommended_actions)} actions for new issue")        
        return recommended_actions
    
    async def _llm_analyze_issue_context(self, title: str, body: str, labels: List[str], 
                                       author: str, trigger_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM with chain of thought to analyze issue context
        """
        from ....LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        context_prompt = f"""
You are an expert GitHub issue analyst. Analyze this newly created issue using a systematic approach.

## Issue Information:
**Title:** {title}
**Author:** {author}
**Existing Labels:** {', '.join(labels) if labels else 'None'}
**Trigger Context:** {trigger_context or 'Issue creation'}

**Body:**
{body}

## Analysis Framework:

Use chain of thought reasoning to analyze:

### Step 1: Issue Type Classification
Analyze the content and determine the primary issue type. Consider:
- Language patterns (bug reports vs feature requests vs questions)
- Keywords and context clues
- User intent and expectations

### Step 2: Complexity Assessment
Evaluate the technical and procedural complexity:
- Technical depth required
- Multiple systems/components involved
- Investigation complexity
- Solution complexity

### Step 3: Information Completeness
Assess what information the user has provided:
- Reproduction steps
- Environment details
- Error messages/logs
- Code samples
- Expected vs actual behavior

### Step 4: Urgency and Impact Assessment
Determine urgency based on:
- Language indicators (critical, urgent, blocking)
- Impact scope (production, many users, data loss)
- Business/project impact

### Step 5: Technical Content Analysis
Identify if this requires technical expertise:
- Code-related issues
- System architecture
- Performance problems
- Integration issues

## Response Format:
Provide your analysis as JSON:
{{
    "reasoning": {{
        "issue_type_analysis": "Your step-by-step reasoning for issue type",
        "complexity_analysis": "Your reasoning for complexity assessment",
        "information_analysis": "Your assessment of information completeness",
        "urgency_analysis": "Your reasoning for urgency level",
        "technical_analysis": "Your assessment of technical content"
    }},
    "conclusions": {{
        "issue_type": "bug|feature_request|question|general",
        "complexity": "low|medium|high|critical",
        "information_completeness": "complete|partial|insufficient",
        "urgency": "low|medium|high|critical",
        "has_technical_content": true/false,
        "needs_triage": true/false,
        "estimated_resolution_effort": "quick|moderate|significant|major"
    }},
    "confidence": 0.0-1.0,
    "key_insights": ["insight1", "insight2", ...],
    "special_considerations": ["consideration1", "consideration2", ...]
}}

Think through each step carefully and provide detailed reasoning.
        """
        
        try:
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=context_prompt)]])
            
            # Extract response text
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
            # Parse JSON response
            try:
                analysis = json.loads(response_text)
                # Flatten the structure for easier access
                result = analysis.get("conclusions", {})
                result["reasoning"] = analysis.get("reasoning", {})
                result["confidence"] = analysis.get("confidence", 0.7)
                result["key_insights"] = analysis.get("key_insights", [])
                result["special_considerations"] = analysis.get("special_considerations", [])
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM analysis response as JSON, using fallback")
                return self._fallback_context_analysis()
                
        except Exception as e:
            logger.error(f"LLM context analysis failed: {str(e)}")
            return self._fallback_context_analysis()
    
    async def _llm_select_tools(self, context_analysis: Dict[str, Any]) -> List[str]:
        """
        Use LLM to select appropriate tools based on context analysis
        """
        from ....LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        available_tools = [
            "rag_search", "similar_issues", "regression_analysis", "code_search",
            "documentation_lookup", "image_analysis", "template_generation"
        ]
        
        tool_selection_prompt = f"""
You are an expert tool selector for GitHub issue analysis. Based on the context analysis, select the most appropriate tools.

## Context Analysis:
{json.dumps(context_analysis, indent=2)}

## Available Tools:
1. **rag_search**: Search knowledge base for relevant information and solutions
2. **similar_issues**: Find similar historical issues and their resolutions
3. **regression_analysis**: Analyze if this might be a regression from recent changes
4. **code_search**: Search codebase for relevant code patterns and implementations
5. **documentation_lookup**: Search documentation for relevant guides and references
6. **image_analysis**: Analyze screenshots or visual content in the issue
7. **template_generation**: Generate templates to help users provide better information

## Selection Criteria:

### Chain of Thought Process:

**Step 1: Core Information Gathering**
- Always consider rag_search for knowledge base lookup
- Consider similar_issues for pattern recognition and historical context

**Step 2: Issue Type Specific Tools**
- Bug reports: Consider regression_analysis, code_search
- Feature requests: Consider documentation_lookup, similar_issues
- Questions: Prioritize documentation_lookup, rag_search

**Step 3: Complexity and Technical Content**
- High complexity or technical issues: Add code_search
- Performance/system issues: Add regression_analysis

**Step 4: Information Completeness**
- Insufficient information: Consider template_generation
- Visual content mentioned: Add image_analysis

**Step 5: Optimization**
- Select 2-4 tools for optimal efficiency
- Prioritize tools that provide maximum value
- Avoid redundant information gathering

## Response Format:
{{
    "reasoning": {{
        "core_tools_rationale": "Why you selected the core tools",
        "specialized_tools_rationale": "Why you added specialized tools",
        "excluded_tools_rationale": "Why you excluded certain tools"
    }},
    "selected_tools": ["tool1", "tool2", ...],
    "tool_priorities": {{
        "tool1": 1,
        "tool2": 2
    }},
    "expected_insights": "What insights you expect these tools to provide"
}}

Provide thoughtful tool selection based on the specific context.
        """
        
        try:
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=tool_selection_prompt)]])
            
            # Extract response text
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
            # Parse JSON response
            try:
                selection = json.loads(response_text)
                selected_tools = selection.get("selected_tools", [])
                # Validate tools are available
                valid_tools = [tool for tool in selected_tools if tool in available_tools]
                return valid_tools if valid_tools else self._fallback_tool_selection()
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM tool selection response, using fallback")
                return self._fallback_tool_selection()
                
        except Exception as e:
            logger.error(f"LLM tool selection failed: {str(e)}")
            return self._fallback_tool_selection()
    
    async def _llm_recommend_actions(self, analysis_results: Dict[str, Any], 
                                   context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to recommend actions based on analysis results and context
        """
        from ....LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        action_prompt = f"""
You are an expert GitHub issue manager. Based on the analysis results and context, recommend appropriate actions.

## Context Analysis:
{json.dumps(context_analysis, indent=2)}

## Analysis Results:
{json.dumps(analysis_results, indent=2)}

## Available Actions:
1. **add_comment**: Post a helpful comment to the issue
2. **add_label**: Add appropriate labels for categorization
3. **assign_reviewer**: Assign to expert reviewer
4. **request_info**: Request additional information
5. **close_issue**: Close if duplicate/invalid
6. **create_pr**: Create pull request for simple fixes
7. **escalate**: Escalate to higher priority

## Action Selection Framework:

### Step 1: Communication Actions
- Always consider add_comment for helpful guidance
- Evaluate if request_info is needed (avoid redundant requests)

### Step 2: Categorization Actions
- Determine appropriate labels based on issue type, complexity, urgency
- Consider technical area labels

### Step 3: Workflow Actions
- Assess if assign_reviewer is needed for complex issues
- Consider escalate for urgent/critical issues

### Step 4: Resolution Actions
- Evaluate if close_issue is appropriate (duplicates, invalid)
- Consider create_pr for simple fixes

## Response Format:
{{
    "reasoning": {{
        "communication_strategy": "Your reasoning for communication actions",
        "categorization_strategy": "Your reasoning for labeling strategy",
        "workflow_strategy": "Your reasoning for workflow actions",
        "prioritization": "How you prioritized the actions"
    }},
    "recommended_actions": [
        {{
            "action": "action_name",
            "priority": 1-5,
            "details": "Specific details for this action",
            "rationale": "Why this action is recommended"
        }}
    ],
    "success_criteria": "How to measure if these actions are successful"
}}

Focus on actions that provide maximum value to the user and project maintainers.
        """
        
        try:
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=action_prompt)]])
            
            # Extract response text
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
            # Parse JSON response
            try:
                actions = json.loads(response_text)
                recommended_actions = actions.get("recommended_actions", [])
                return recommended_actions if recommended_actions else self._fallback_actions()
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM action recommendations, using fallback")
                return self._fallback_actions()
                
        except Exception as e:
            logger.error(f"LLM action recommendation failed: {str(e)}")
            return self._fallback_actions()
    
    def _format_context_for_prompt(self, context_analysis: Dict[str, Any]) -> str:
        """Format context analysis for use in prompts"""
        formatted = []
        for key, value in context_analysis.items():
            if key not in ['strategy', 'author', 'existing_labels', 'approach']:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    def _fallback_context_analysis(self) -> Dict[str, Any]:
        """Fallback context analysis if LLM fails"""
        return {
            "issue_type": "general",
            "complexity": "medium",
            "information_completeness": "partial",
            "urgency": "medium",
            "has_technical_content": False,
            "confidence": 0.5,
            "key_insights": ["Fallback analysis used"],
            "special_considerations": ["LLM analysis unavailable"]
        }
    
    def _fallback_tool_selection(self) -> List[str]:
        """Fallback tool selection if LLM fails"""
        return ["rag_search", "similar_issues"]
    
    def _fallback_actions(self) -> List[Dict[str, Any]]:
        """Fallback actions if LLM fails"""
        return [
            {
                "action": "add_comment",
                "priority": 1,
                "details": "Provide helpful response",
                "rationale": "Default action for issue engagement"
            }
        ]
