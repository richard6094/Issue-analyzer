# analyzer_core/analyzers/initial_assessor.py
"""
Initial assessment component that analyzes issues and selects tools
"""

import json
import logging
from typing import Dict, Any, Optional, List
from LLM.llm_provider import get_llm
from langchain_core.messages import HumanMessage

from ..models.tool_models import DecisionStep, AvailableTools
from ..utils.text_utils import assess_user_provided_information, prepare_issue_context

logger = logging.getLogger(__name__)


class InitialAssessor:
    """Handles initial assessment of issues"""
    
    def __init__(self):
        self.llm = get_llm(provider="azure", temperature=0.1)
    
    async def assess(self, issue_data: Dict[str, Any], 
                    comment_data: Optional[Dict[str, Any]] = None,
                    event_name: str = "", event_action: str = "") -> DecisionStep:
        """Make initial assessment and select appropriate tools"""
        try:
            logger.info("Making initial assessment with LLM...")
            
            # Assess user provided information first
            user_info_assessment = assess_user_provided_information(issue_data)
            
            # Prepare the issue context
            issue_context = prepare_issue_context(issue_data, event_name, event_action)
            
            # Create tool descriptions for the LLM
            tool_descriptions = self._get_tool_descriptions()
            
            # Enhanced LLM prompt for initial assessment
            assessment_prompt = f"""
You are an intelligent GitHub issue analyst. Your primary responsibility is to analyze issues accurately while respecting information already provided by users.

## CRITICAL GUIDELINES - READ CAREFULLY:

### 1. USER INFORMATION RECOGNITION
Before selecting any tools, you MUST first identify what information the user has ALREADY PROVIDED:

**Check for Existing Information:**
- **Code Samples**: Look for code blocks (```), function definitions, class declarations
- **Reproduction Steps**: Look for numbered steps, "how to reproduce", step-by-step instructions
- **Error Messages**: Look for stack traces, console outputs, error logs
- **Screenshots/Images**: Look for image links, attachments, visual content
- **Environment Details**: Look for version numbers, browser info, system specifications

### 2. INFORMATION COMPLETENESS ASSESSMENT
Evaluate what the user has provided:
- **COMPLETE**: User provided comprehensive information for diagnosis
- **PARTIAL**: User provided some information but key details are missing
- **INSUFFICIENT**: User provided minimal information, significant gaps exist

### 3. TOOL SELECTION PRINCIPLES
- **NEVER** request information the user has already provided
- **PRIORITIZE** tools that work with existing information
- **ONLY** suggest information requests for genuinely missing critical data

## Issue Context:
{issue_context}

## Available Tools:
{tool_descriptions}

## Your Analysis Task:

**Step 1: Information Inventory**
First, create an inventory of what the user has already provided.

**Step 2: Gap Analysis**  
Identify what critical information is genuinely missing.

**Step 3: Tool Selection**
Select tools that will provide maximum value without redundancy.

## Response Format:
{{
    "provided_information": {{
        "has_code_samples": true/false,
        "has_reproduction_steps": true/false,
        "has_error_messages": true/false,
        "has_screenshots": true/false,
        "completeness_level": "complete|partial|insufficient"
    }},
    "reasoning": "Your detailed reasoning about the issue and tool selection based on existing information",
    "selected_tools": ["tool1", "tool2", ...],
    "expected_outcome": "What you expect to learn from these tools",
    "priority": 1
}}

**REMEMBER**: Your goal is to be helpful while respecting the user's time and the information they've already provided.
"""
            
            # Get LLM decision using Azure OpenAI
            response = await self.llm.agenerate([[HumanMessage(content=assessment_prompt)]])
            
            # Handle LangChain response format
            response_text = self._extract_response_text(response)
            
            # Parse LLM response
            try:
                decision_data = json.loads(response_text)
                selected_tools = [AvailableTools(tool) for tool in decision_data.get("selected_tools", [])]
                
                decision = DecisionStep(
                    reasoning=decision_data.get("reasoning", ""),
                    selected_tools=selected_tools,
                    expected_outcome=decision_data.get("expected_outcome", ""),
                    priority=decision_data.get("priority", 1),
                    user_info_assessment=user_info_assessment
                )
                
                logger.info(f"LLM selected {len(selected_tools)} tools: {[tool.value for tool in selected_tools]}")
                logger.info(f"User info completeness: {user_info_assessment['completeness_level']}")
                return decision
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM decision, using fallback: {str(e)}")
                # Fallback to basic tool selection
                return self._fallback_tool_selection(issue_data)
                
        except Exception as e:
            logger.error(f"Error in initial assessment: {str(e)}")
            return self._fallback_tool_selection(issue_data)
    
    def _extract_response_text(self, response) -> str:
        """Extract text from LangChain response"""
        if hasattr(response.generations[0][0], 'text'):
            return response.generations[0][0].text
        elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
            return response.generations[0][0].message.content
        else:
            return str(response.generations[0][0])
    
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
    
    def _fallback_tool_selection(self, issue_data: Dict[str, Any]) -> DecisionStep:
        """Fallback tool selection when LLM parsing fails"""
        from ..utils.text_utils import has_images_in_text
        
        tools = [AvailableTools.RAG_SEARCH, AvailableTools.SIMILAR_ISSUES]
        
        # Add image analysis if images detected
        if has_images_in_text(issue_data.get('body', '')):
            tools.append(AvailableTools.IMAGE_ANALYSIS)
        
        return DecisionStep(
            reasoning="Fallback selection due to LLM parsing error",
            selected_tools=tools,
            expected_outcome="Gather basic information about the issue",
            priority=2
        )
