#!/usr/bin/env python3
"""
Intelligent Function Dispatcher for GitHub Actions

This script implements an LLM-driven intelligent dispatcher that analyzes issues
and makes decisions about which tools to use, similar to GitHub Copilot's approach.
Instead of mechanical execution, the LLM reasons about the problem and selects
appropriate tools dynamically.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from LLM.llm_provider import get_llm
from RAG.rag_helper import default_rag_helper
from image_recognition.image_recognition_provider import get_image_recognition_model, analyze_image
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('intelligent_dispatch_logs.txt')
    ]
)
logger = logging.getLogger(__name__)


class AvailableTools(Enum):
    """Available tools that the LLM can choose to use"""
    RAG_SEARCH = "rag_search"
    IMAGE_ANALYSIS = "image_analysis"
    REGRESSION_ANALYSIS = "regression_analysis"
    CODE_SEARCH = "code_search"
    SIMILAR_ISSUES = "similar_issues"
    DOCUMENTATION_LOOKUP = "documentation_lookup"
    TEMPLATE_GENERATION = "template_generation"


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool: AvailableTools
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DecisionStep:
    """A single decision step in the analysis process"""
    reasoning: str
    selected_tools: List[AvailableTools]
    expected_outcome: str
    priority: int = 1


class IntelligentGitHubDispatcher:
    """
    Intelligent Function Dispatcher that uses LLM reasoning to make decisions
    """
    
    def __init__(self):
        """Initialize the intelligent dispatcher"""
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.repo_owner = os.environ.get("REPO_OWNER")
        self.repo_name = os.environ.get("REPO_NAME")
        self.repo_full_name = os.environ.get("REPO_FULL_NAME")
        self.issue_number = os.environ.get("ISSUE_NUMBER")
        self.event_name = os.environ.get("GITHUB_EVENT_NAME")
        self.event_action = os.environ.get("GITHUB_EVENT_ACTION")
        self.sender_login = os.environ.get("SENDER_LOGIN")
        self.sender_type = os.environ.get("SENDER_TYPE")
        self.force_analysis = os.environ.get("FORCE_ANALYSIS", "false").lower() == "true"
        
        # Validate required environment variables
        if not all([self.github_token, self.repo_owner, self.repo_name, self.issue_number]):
            raise ValueError("Missing required environment variables")
        
        # Initialize tools registry
        self.available_tools = {
            AvailableTools.RAG_SEARCH: self._execute_rag_search,
            AvailableTools.IMAGE_ANALYSIS: self._execute_image_analysis,
            AvailableTools.REGRESSION_ANALYSIS: self._execute_regression_analysis,
            AvailableTools.CODE_SEARCH: self._execute_code_search,
            AvailableTools.SIMILAR_ISSUES: self._execute_similar_issues,
            AvailableTools.DOCUMENTATION_LOOKUP: self._execute_documentation_lookup,
            AvailableTools.TEMPLATE_GENERATION: self._execute_template_generation,
        }
        
        # Initialize results storage
        self.analysis_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "issue_data": {},
            "decision_history": [],
            "tool_results": [],
            "final_analysis": {},
            "actions_taken": []
        }
        
        logger.info(f"Initialized Intelligent GitHub Dispatcher for issue #{self.issue_number}")

    async def analyze_and_decide(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method that uses LLM reasoning to decide on actions
        
        Args:
            issue_data: GitHub issue data
            
        Returns:
            Complete analysis and decision result
        """
        try:
            # Store issue data in context
            self.analysis_context["issue_data"] = issue_data
            
            # Step 1: Initial assessment and tool selection
            initial_decision = await self._make_initial_assessment(issue_data)
            self.analysis_context["decision_history"].append(initial_decision)
            
            # Step 2: Execute selected tools and gather information
            tool_results = await self._execute_selected_tools(initial_decision.selected_tools, issue_data)
            self.analysis_context["tool_results"].extend(tool_results)
            
            # Step 3: Analyze results and decide on next steps
            next_decision = await self._analyze_tool_results(issue_data, tool_results)
            
            # Step 4: Execute additional tools if needed
            if next_decision and next_decision.selected_tools:
                additional_results = await self._execute_selected_tools(next_decision.selected_tools, issue_data)
                self.analysis_context["tool_results"].extend(additional_results)
                self.analysis_context["decision_history"].append(next_decision)
            
            # Step 5: Generate final analysis and recommendations
            final_analysis = await self._generate_final_analysis(issue_data, self.analysis_context["tool_results"])
            self.analysis_context["final_analysis"] = final_analysis
            
            # Step 6: Take actions based on analysis
            actions_taken = await self._take_intelligent_actions(final_analysis)
            self.analysis_context["actions_taken"] = actions_taken
            
            logger.info("Intelligent analysis completed successfully")
            return self.analysis_context
            
        except Exception as e:
            logger.error(f"Error in intelligent analysis: {str(e)}")
            return {"error": str(e), "context": self.analysis_context}

    async def _make_initial_assessment(self, issue_data: Dict[str, Any]) -> DecisionStep:
        """
        Use LLM to make initial assessment and select appropriate tools
        
        Args:
            issue_data: GitHub issue data
            
        Returns:
            DecisionStep with reasoning and selected tools
        """
        try:
            logger.info("Making initial assessment with LLM...")
            
            # Prepare the issue context
            issue_context = self._prepare_issue_context(issue_data)
            
            # Create tool descriptions for the LLM
            tool_descriptions = self._get_tool_descriptions()
            
            # LLM prompt for initial assessment
            assessment_prompt = f"""
You are an intelligent GitHub issue analyst. Your task is to analyze the given issue and decide which tools would be most helpful to understand and resolve it.

## Issue Context:
{issue_context}

## Available Tools:
{tool_descriptions}

## Your Task:
Analyze this issue and decide which tools would be most helpful. Consider:
1. What type of problem is this? (bug, feature request, question, etc.)
2. What information would be most valuable to gather?
3. Which tools would provide that information?
4. What is the priority of each tool?

Please respond with a JSON object containing:
{{
    "reasoning": "Your detailed reasoning about the issue and why you chose these tools",
    "selected_tools": ["tool1", "tool2", ...],  // Array of tool names from the available tools
    "expected_outcome": "What you expect to learn from these tools",
    "priority": 1  // 1=high, 2=medium, 3=low
}}

Focus on being helpful and selecting tools that will provide the most value for understanding and resolving this specific issue.
"""
            
            # Get LLM decision using Azure OpenAI
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=assessment_prompt)]])
            # Handle LangChain agenerate response format
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
            # Parse LLM response
            try:
                decision_data = json.loads(response_text)
                selected_tools = [AvailableTools(tool) for tool in decision_data.get("selected_tools", [])]
                
                decision = DecisionStep(
                    reasoning=decision_data.get("reasoning", ""),
                    selected_tools=selected_tools,
                    expected_outcome=decision_data.get("expected_outcome", ""),
                    priority=decision_data.get("priority", 1)
                )
                
                logger.info(f"LLM selected {len(selected_tools)} tools: {[tool.value for tool in selected_tools]}")
                return decision
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM decision, using fallback: {str(e)}")
                # Fallback to basic tool selection
                return self._fallback_tool_selection(issue_data)
                
        except Exception as e:
            logger.error(f"Error in initial assessment: {str(e)}")
            return self._fallback_tool_selection(issue_data)

    def _prepare_issue_context(self, issue_data: Dict[str, Any]) -> str:
        """Prepare issue context for LLM analysis"""
        title = issue_data.get('title', 'No title')
        body = issue_data.get('body', 'No description')[:1000]  # Limit length
        labels = [label.get('name', '') for label in issue_data.get('labels', [])]
        comments_count = issue_data.get('comments', 0)
        
        # Check for images
        has_images = self._has_images_in_text(body)
        
        context = f"""
**Title:** {title}

**Description:** {body}

**Labels:** {', '.join(labels) if labels else 'None'}

**Comments:** {comments_count}

**Contains Images:** {'Yes' if has_images else 'No'}

**Event Type:** {self.event_name} - {self.event_action}
"""
        return context

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

    def _has_images_in_text(self, text: str) -> bool:
        """Check if text contains image references"""
        import re
        image_patterns = [
            r'!\[.*?\]\(.*?\)',  # Markdown images
            r'<img.*?>',  # HTML img tags
            r'https?://[^\s]+\.(png|jpg|jpeg|gif|svg)',  # Direct image URLs
        ]
        
        for pattern in image_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _fallback_tool_selection(self, issue_data: Dict[str, Any]) -> DecisionStep:
        """Fallback tool selection when LLM parsing fails"""
        tools = [AvailableTools.RAG_SEARCH, AvailableTools.SIMILAR_ISSUES]
        
        # Add image analysis if images detected
        if self._has_images_in_text(issue_data.get('body', '')):
            tools.append(AvailableTools.IMAGE_ANALYSIS)
        
        return DecisionStep(
            reasoning="Fallback selection due to LLM parsing error",
            selected_tools=tools,
            expected_outcome="Gather basic information about the issue",
            priority=2
        )

    async def _execute_selected_tools(self, selected_tools: List[AvailableTools], issue_data: Dict[str, Any]) -> List[ToolResult]:
        """Execute the tools selected by the LLM"""
        results = []
        
        for tool in selected_tools:
            try:
                logger.info(f"Executing tool: {tool.value}")
                tool_function = self.available_tools.get(tool)
                
                if tool_function:
                    result = await tool_function(issue_data)
                    results.append(ToolResult(
                        tool=tool,
                        success=True,
                        data=result,
                        confidence=result.get('confidence', 0.7)
                    ))
                else:
                    logger.warning(f"Tool {tool.value} not implemented")
                    results.append(ToolResult(
                        tool=tool,
                        success=False,
                        data={},
                        error_message=f"Tool {tool.value} not implemented"
                    ))
                    
            except Exception as e:
                logger.error(f"Error executing tool {tool.value}: {str(e)}")
                results.append(ToolResult(
                    tool=tool,
                    success=False,
                    data={},
                    error_message=str(e)
                ))
        
        return results

    async def _analyze_tool_results(self, issue_data: Dict[str, Any], tool_results: List[ToolResult]) -> Optional[DecisionStep]:
        """
        Analyze tool results and decide if additional tools are needed
        
        Args:
            issue_data: Original issue data
            tool_results: Results from executed tools
            
        Returns:
            Optional DecisionStep for additional tools
        """
        try:
            # Prepare results summary for LLM
            results_summary = self._prepare_results_summary(tool_results)
            issue_context = self._prepare_issue_context(issue_data)
            
            analysis_prompt = f"""
You are analyzing the results from tools that were executed to understand a GitHub issue.

## Original Issue:
{issue_context}

## Tool Results:
{results_summary}

## Available Additional Tools:
{self._get_tool_descriptions()}

## Your Task:
Based on the tool results, determine if we need additional information to properly understand and address this issue.

Please respond with a JSON object:
{{
    "needs_more_tools": true/false,
    "reasoning": "Your reasoning about whether more tools are needed",
    "selected_tools": ["tool1", "tool2", ...],  // Only if needs_more_tools is true
    "expected_outcome": "What you expect to learn",  // Only if needs_more_tools is true
    "priority": 1  // 1=high, 2=medium, 3=low
}}

Consider:
1. Do we have enough information to provide a helpful response?
2. Are there gaps in our understanding?
3. Would additional tools provide significant value?
"""
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=analysis_prompt)]])
            # Handle LangChain agenerate response format
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
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

    async def _generate_final_analysis(self, issue_data: Dict[str, Any], tool_results: List[ToolResult]) -> Dict[str, Any]:
        """
        Generate final analysis and recommendations using LLM
        
        Args:
            issue_data: Original issue data
            tool_results: All tool results
            
        Returns:
            Final analysis with recommendations and actions
        """
        try:
            issue_context = self._prepare_issue_context(issue_data)
            results_summary = self._prepare_results_summary(tool_results)
            
            final_analysis_prompt = f"""
You are a senior GitHub issue analyst. Based on the issue and the tool analysis results, provide a comprehensive analysis and actionable recommendations.

## Original Issue:
{issue_context}

## Analysis Results:
{results_summary}

## Your Task:
Provide a comprehensive analysis and recommendations. Respond with a JSON object containing:

{{
    "issue_type": "bug_report|feature_request|question|documentation|regression|security|performance",
    "severity": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "summary": "Brief summary of the issue and findings",
    "detailed_analysis": "Detailed analysis based on tool results",
    "root_cause": "Likely root cause if identified",
    "recommended_labels": ["label1", "label2", ...],
    "recommended_actions": [
        {{
            "action": "add_label|add_comment|assign_user|close_issue|request_info",
            "details": "Specific details for the action",
            "priority": 1
        }}
    ],
    "user_comment": "A helpful comment to post on the issue that synthesizes all findings and provides guidance"
}}

Make your analysis helpful, actionable, and professional. The user_comment should provide real value to the issue creator.
"""
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=final_analysis_prompt)]])
            # Handle LangChain agenerate response format
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
            try:
                final_analysis = json.loads(response_text)
                logger.info(f"Generated final analysis: {final_analysis.get('issue_type', 'unknown')} with {final_analysis.get('confidence', 0):.1%} confidence")
                return final_analysis
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse final analysis, using text response: {str(e)}")
                return {
                    "issue_type": "unknown",
                    "severity": "medium",
                    "confidence": 0.5,
                    "summary": "Analysis completed but response parsing failed",
                    "detailed_analysis": response_text,
                    "user_comment": response_text
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

    async def _take_intelligent_actions(self, final_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Take actions based on the final analysis
        
        Args:
            final_analysis: Final analysis from LLM
            
        Returns:
            List of actions taken
        """
        actions_taken = []
        
        try:
            # Add recommended labels
            recommended_labels = final_analysis.get("recommended_labels", [])
            if recommended_labels:
                await self._add_labels_to_issue(recommended_labels)
                actions_taken.append({
                    "action": "labels_added",
                    "details": recommended_labels,
                    "success": True
                })
            
            # Add user comment
            user_comment = final_analysis.get("user_comment", "")
            if user_comment:
                await self._add_comment_to_issue(user_comment)
                actions_taken.append({
                    "action": "comment_added",
                    "details": "Analysis comment posted",
                    "success": True
                })
            
            # Process recommended actions
            recommended_actions = final_analysis.get("recommended_actions", [])
            for action in recommended_actions:
                try:
                    action_type = action.get("action", "")
                    action_details = action.get("details", "")
                    
                    if action_type == "add_label" and action_details:
                        await self._add_labels_to_issue([action_details])
                        actions_taken.append({
                            "action": "label_added",
                            "details": action_details,
                            "success": True
                        })
                    elif action_type == "add_comment" and action_details:
                        await self._add_comment_to_issue(action_details)
                        actions_taken.append({
                            "action": "comment_added",
                            "details": "Additional comment posted",
                            "success": True
                        })
                    # Add more action types as needed
                    
                except Exception as e:
                    logger.error(f"Error executing action {action}: {str(e)}")
                    actions_taken.append({
                        "action": action.get("action", "unknown"),
                        "details": f"Failed: {str(e)}",
                        "success": False
                    })
            
        except Exception as e:
            logger.error(f"Error taking intelligent actions: {str(e)}")
            actions_taken.append({
                "action": "error",
                "details": str(e),
                "success": False
            })
        
        return actions_taken

    # Tool implementation methods
    async def _execute_rag_search(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG search for relevant information"""
        try:
            issue_content = f"{issue_data.get('title', '')} {issue_data.get('body', '')}"
            
            results, context = default_rag_helper.query_for_regression_analysis(
                issue_title=issue_data.get('title', ''),
                issue_body=issue_data.get('body', ''),
                n_results=5
            )
            
            return {
                "search_results": results,
                "context": context,
                "results_count": len(results) if results else 0,
                "confidence": 0.8 if results else 0.3
            }
        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}

    async def _execute_image_analysis(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image analysis for visual content"""
        try:
            body = issue_data.get('body', '')
            image_urls = self._extract_image_urls(body)
            
            if not image_urls:
                return {"message": "No images found", "confidence": 0.0}
            
            image_model = get_image_recognition_model(provider="azure")
            analysis_results = []
            
            for url in image_urls[:3]:  # Limit to 3 images
                try:
                    result = analyze_image(
                        image_url=url,
                        prompt=f"Analyze this image in the context of a GitHub issue: {issue_data.get('title', '')}",
                        llm=image_model
                    )
                    analysis_results.append({
                        "url": url,
                        "analysis": result
                    })
                except Exception as e:
                    logger.warning(f"Failed to analyze image {url}: {str(e)}")
            
            return {
                "images_analyzed": len(analysis_results),
                "results": analysis_results,
                "confidence": 0.8 if analysis_results else 0.2
            }
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}

    async def _execute_regression_analysis(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regression analysis"""
        try:
            from scripts.analyze_issue import analyze_issue_for_regression, get_azure_chat_model
            
            issue_content = f"{issue_data.get('title', '')} {issue_data.get('body', '')}"
            chat_model = get_azure_chat_model("gpt-4o")
            
            result = analyze_issue_for_regression(
                issue_content=issue_content,
                issue_number=int(self.issue_number),
                chat_model=chat_model,
                issue_title=issue_data.get('title'),
                issue_body=issue_data.get('body')
            )
            
            return result
        except Exception as e:
            logger.error(f"Regression analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}

    async def _execute_code_search(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code search (placeholder for future implementation)"""
        return {
            "message": "Code search not yet implemented",
            "confidence": 0.0
        }

    async def _execute_similar_issues(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute similar issues search"""
        try:
            # This could be enhanced with more sophisticated similarity search
            issue_content = f"{issue_data.get('title', '')} {issue_data.get('body', '')}"
            
            results, context = default_rag_helper.query_for_regression_analysis(
                issue_title=issue_data.get('title', ''),
                issue_body=issue_data.get('body', ''),
                n_results=3
            )
            
            return {
                "similar_issues": results,
                "context": context,
                "count": len(results) if results else 0,
                "confidence": 0.7 if results else 0.3
            }
        except Exception as e:
            logger.error(f"Similar issues search failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}

    async def _execute_documentation_lookup(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation lookup (placeholder for future implementation)"""
        return {
            "message": "Documentation lookup not yet implemented",
            "confidence": 0.0
        }

    async def _execute_template_generation(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute template generation"""
        try:
            # Basic template suggestions based on issue content
            title = issue_data.get('title', '').lower()
            body = issue_data.get('body', '').lower()
            
            if any(word in title + body for word in ['bug', 'error', 'crash', 'fail']):
                template_type = "bug_report"
            elif any(word in title + body for word in ['feature', 'request', 'add', 'enhance']):
                template_type = "feature_request"
            elif any(word in title + body for word in ['question', 'how', 'help', 'why']):
                template_type = "question"
            else:
                template_type = "general"
            
            templates = {
                "bug_report": {
                    "sections": ["Steps to Reproduce", "Expected Behavior", "Actual Behavior", "Environment Details"],
                    "description": "Bug report template for systematic issue reporting"
                },
                "feature_request": {
                    "sections": ["Problem Description", "Proposed Solution", "Alternatives Considered", "Use Cases"],
                    "description": "Feature request template for enhancement proposals"
                },
                "question": {
                    "sections": ["Context", "Question", "What I've Tried", "Expected Outcome"],
                    "description": "Question template for support requests"
                },
                "general": {
                    "sections": ["Description", "Context", "Expected Outcome"],
                    "description": "General template for various issue types"
                }
            }
            
            return {
                "recommended_template": template_type,
                "template_data": templates[template_type],
                "confidence": 0.6
            }
        except Exception as e:
            logger.error(f"Template generation failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}

    def _extract_image_urls(self, text: str) -> List[str]:
        """Extract image URLs from text"""
        import re
        patterns = [
            r'!\[.*?\]\((https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|svg))\)',  # Markdown images
            r'<img[^>]+src=["\']([^"\']+)["\']',  # HTML img tags
            r'(https?://[^\s]+\.(?:png|jpg|jpeg|gif|svg))',  # Direct URLs
        ]
        
        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            urls.extend(matches)
        
        return list(set(urls))  # Remove duplicates

    async def _add_labels_to_issue(self, labels: List[str]):
        """Add labels to the GitHub issue"""
        try:
            import requests
            
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}/labels"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            data = {"labels": labels}
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                logger.info(f"Successfully added labels: {labels}")
            else:
                logger.error(f"Failed to add labels. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error adding labels: {str(e)}")

    async def _add_comment_to_issue(self, comment_text: str):
        """Add a comment to the GitHub issue"""
        try:
            import requests
            
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}/comments"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            data = {"body": comment_text}
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                logger.info("Successfully added comment to issue")
            else:
                logger.error(f"Failed to add comment. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")

    async def fetch_issue_data(self) -> Optional[Dict[str, Any]]:
        """Fetch issue data from GitHub API"""
        try:
            import requests
            
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch issue data. Status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching issue data: {str(e)}")
            return None

    def save_results(self):
        """Save analysis results to file"""
        try:
            with open("intelligent_dispatch_results.json", "w") as f:
                json.dump(self.analysis_context, f, indent=2, default=str)
            logger.info("Results saved to intelligent_dispatch_results.json")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    async def run(self):
        """Main execution method"""
        try:
            logger.info("Starting Intelligent Function Dispatcher execution...")
            
            # Fetch issue data
            issue_data = await self.fetch_issue_data()
            if not issue_data:
                logger.error("Failed to fetch issue data")
                return
            
            # Run intelligent analysis
            result = await self.analyze_and_decide(issue_data)
            
            logger.info("Intelligent Function Dispatcher execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            self.analysis_context["error"] = str(e)
        finally:
            self.save_results()


async def main():
    """Main entry point for GitHub Actions"""
    try:
        dispatcher = IntelligentGitHubDispatcher()
        await dispatcher.run()
        print("Intelligent Function Dispatcher completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Intelligent Function Dispatcher failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
