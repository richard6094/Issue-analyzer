#!/usr/bin/env python3
"""
Intelligent Function Dispatcher for GitHub Actions - Entry Point

This script serves as the entry point for the intelligent issue analyzer system.
It has been refactored to use the modular analyzer_core components.
"""

import os
import sys
import json
import asyncio
import logging

# Add parent directory to sys.path to allow importing sibling packages
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the new modular components
from analyzer_core import IntelligentDispatcher
from analyzer_core.utils import fetch_issue_data, fetch_comment_data

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


async def main():
    """Main entry point for GitHub Actions"""
    try:
        logger.info("Starting Intelligent Function Dispatcher (Refactored Version)...")
        
        # Load configuration from environment
        config = {
            "github_token": os.environ.get("GITHUB_TOKEN"),
            "repo_owner": os.environ.get("REPO_OWNER"),
            "repo_name": os.environ.get("REPO_NAME"),
            "repo_full_name": os.environ.get("REPO_FULL_NAME"),
            "issue_number": os.environ.get("ISSUE_NUMBER"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
            "event_action": os.environ.get("GITHUB_EVENT_ACTION"),
            "sender_login": os.environ.get("SENDER_LOGIN"),
            "sender_type": os.environ.get("SENDER_TYPE"),
            "force_analysis": os.environ.get("FORCE_ANALYSIS", "false").lower() == "true"
        }
        
        # Validate configuration
        required_fields = ["github_token", "repo_owner", "repo_name", "issue_number"]
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
        
        # Fetch data
        logger.info("Fetching issue and comment data...")
        issue_data = await fetch_issue_data(config)
        if not issue_data:
            raise ValueError("Failed to fetch issue data")
        
        comment_data = await fetch_comment_data(config)
        
        # Create and run dispatcher
        logger.info("Initializing intelligent dispatcher...")
        dispatcher = IntelligentDispatcher(config)
        
        logger.info("Running analysis workflow...")
        result = await dispatcher.analyze(issue_data, comment_data)
        
        # Save results
        output_file = "intelligent_dispatch_results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        logger.info("Intelligent Function Dispatcher completed successfully")
        
        # Print summary
        if result.get("trigger_decision", {}).get("should_trigger"):
            logger.info(f"Analysis completed. Trigger type: {result.get('trigger_decision', {}).get('trigger_type')}")
            logger.info(f"Actions taken: {len(result.get('actions_taken', []))}")
        else:
            logger.info(f"Analysis skipped. Reason: {result.get('trigger_decision', {}).get('reason')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Intelligent Function Dispatcher failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


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
            
            # Assess user provided information first
            user_info_assessment = self._assess_user_provided_information(issue_data)
            
            # Prepare the issue context
            issue_context = self._prepare_issue_context(issue_data)
            
            # Create tool descriptions for the LLM
            tool_descriptions = self._get_tool_descriptions()
              # Enhanced LLM prompt for initial assessment (based on PROMPT_IMPROVEMENT_PROPOSAL)
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
                # Store the user information assessment in the decision for later use
                if hasattr(decision, 'user_info_assessment'):
                    decision.user_info_assessment = user_info_assessment
                
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

    def _prepare_issue_context(self, issue_data: Dict[str, Any]) -> str:
        """Prepare issue context for LLM analysis"""
        title = issue_data.get('title', 'No title')
        body = issue_data.get('body', 'No description')
        labels = [label.get('name', '') for label in issue_data.get('labels', [])]
        comments_count = issue_data.get('comments', 0)
        
        # Check for images and base64 data
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
            issue_context = self._prepare_issue_context(issue_data)            # Enhanced LLM prompt for tool results analysis (based on PROMPT_IMPROVEMENT_PROPOSAL)
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
            results_summary = self._prepare_results_summary(tool_results)            # Enhanced LLM prompt for final analysis (based on PROMPT_IMPROVEMENT_PROPOSAL)
            final_analysis_prompt = f"""
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
    "recommended_labels": ["label1", "label2", ...],    "recommended_actions": [
        {{
            "action": "add_label|assign_user|close_issue|request_info",
            "details": "Specific details for the action",
            "priority": 1
        }}
    ],
    "user_comment": "A helpful comment that builds on the user's provided information and offers genuine value"
}}

**CRITICAL**: Your response must demonstrate that you've carefully reviewed and understood the user's contributions. Generic responses that ignore their specific details are unacceptable.
"""
            llm = get_llm(provider="azure", temperature=0.1)
            response = await llm.agenerate([[HumanMessage(content=final_analysis_prompt)]])            # Handle LangChain agenerate response format
            if hasattr(response.generations[0][0], 'text'):
                response_text = response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
                response_text = response.generations[0][0].message.content
            else:
                response_text = str(response.generations[0][0])
            
            try:
                # First try to extract and parse JSON properly
                final_analysis = self._extract_and_parse_json_response(response_text)
                
                # Validate and clean the user_comment
                if "user_comment" in final_analysis:
                    final_analysis["user_comment"] = self._clean_user_comment(final_analysis["user_comment"])
                
                logger.info(f"Generated final analysis: {final_analysis.get('issue_type', 'unknown')} with {final_analysis.get('confidence', 0):.1%} confidence")
                return final_analysis
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse final analysis, using cleaned text response: {str(e)}")
                # Extract user comment from the raw response if JSON parsing fails
                cleaned_comment = self._extract_user_comment_from_response(response_text)
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
              
    async def _take_intelligent_actions(self, final_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Take actions based on the final analysis
        
        Args:
            final_analysis: Final analysis from LLM
            
        Returns:
            List of actions taken
        """
        actions_taken = []
        all_labels_to_add = set()  # Use set to avoid duplicate labels
        
        try:
            # Step 1: Collect all labels (from both recommended_labels and recommended_actions)
            recommended_labels = final_analysis.get("recommended_labels", [])
            if recommended_labels:
                all_labels_to_add.update(recommended_labels)
            
            # Process recommended actions to collect additional labels and other actions
            recommended_actions = final_analysis.get("recommended_actions", [])
            non_label_actions = []
            
            for action in recommended_actions:
                action_type = action.get("action", "")
                action_details = action.get("details", "")
                
                if action_type == "add_label" and action_details:
                    # Collect label for batch addition
                    if all_labels_to_add:
                        labels_list = list(all_labels_to_add)
                        await self._add_labels_to_issue(labels_list)
                    actions_taken.append({
                        "action": "label_added", 
                        "details": action_details,
                        "success": True
                    })
                elif action_type == "add_comment":
                    # Skip comment actions - we'll use user_comment instead
                    logger.info("Skipping add_comment action - using main user_comment instead")
                    actions_taken.append({
                        "action": "comment_action_merged",
                        "details": "Comment action merged into main user_comment",
                        "success": True
                    })
                else:
                    # Other actions (assign_user, close_issue, etc.)
                    non_label_actions.append(action)
            
            # Step 2: Add the main user comment
            user_comment = final_analysis.get("user_comment", "")
            if user_comment:
                await self._add_comment_to_issue(user_comment)
                actions_taken.append({
                    "action": "comment_added",
                    "details": "Analysis comment posted",
                    "success": True
                })
            
            # Step 3: Execute other non-label, non-comment actions
            for action in non_label_actions:
                try:
                    action_type = action.get("action", "")
                    action_details = action.get("details", "")
                    
                    if action_type == "assign_user" and action_details:
                        # TODO: Implement user assignment
                        logger.info(f"User assignment not yet implemented: {action_details}")
                        actions_taken.append({
                            "action": "assign_user_pending",
                            "details": f"Assignment to {action_details} - not yet implemented",
                            "success": False
                        })
                    elif action_type == "close_issue":
                        # TODO: Implement issue closing
                        logger.info("Issue closing not yet implemented")
                        actions_taken.append({
                            "action": "close_issue_pending",
                            "details": "Issue closing - not yet implemented",
                            "success": False
                        })
                    elif action_type == "request_info":
                        # This would be handled through the user_comment
                        logger.info("Info request handled through main comment")
                        actions_taken.append({
                            "action": "info_request_handled",
                            "details": "Information request included in main comment",
                            "success": True
                        })
                    else:
                        logger.warning(f"Unknown action type: {action_type}")
                        actions_taken.append({
                            "action": "unknown_action",
                            "details": f"Unknown action type: {action_type}",
                            "success": False
                        })
                    
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

    def _assess_user_provided_information(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess what information the user has already provided
        
        Args:
            issue_data: GitHub issue data
            
        Returns:
            Assessment of user-provided information
        """
        try:
            body = issue_data.get('body', '')
            title = issue_data.get('title', '')
            combined_text = f"{title} {body}".lower()
            
            assessment = {
                'has_code_samples': bool(re.search(r'```|`[^`]+`', body)),
                'has_reproduction_steps': any(indicator in combined_text for indicator in [
                    'steps to reproduce', 'reproduction steps', 'how to reproduce',
                    'to reproduce:', 'step 1', 'step 2', 'reproduc'
                ]),
                'has_error_messages': any(indicator in combined_text for indicator in [
                    'error:', 'exception:', 'traceback', 'stack trace',
                    'console.error', 'throw', 'failed with', 'error'
                ]),
                'has_screenshots': self._has_images_in_text(body),
                'has_environment_details': any(indicator in combined_text for indicator in [
                    'version', 'browser', 'os', 'operating system', 'environment',
                    'chrome', 'firefox', 'safari', 'edge', 'windows', 'mac', 'linux'
                ])
            }
            
            # Determine completeness level
            provided_count = sum(assessment.values())
            if provided_count >= 4:
                assessment['completeness_level'] = 'complete'
            elif provided_count >= 2:
                assessment['completeness_level'] = 'partial'
            else:
                assessment['completeness_level'] = 'insufficient'
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing user provided information: {str(e)}")
            return {
                'has_code_samples': False,
                'has_reproduction_steps': False,
                'has_error_messages': False,
                'has_screenshots': False,
                'has_environment_details': False,
                'completeness_level': 'insufficient'
            }

    def _extract_and_parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response, handling nested JSON structures
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        import re
        
        # First try to parse the entire response as JSON
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON within the response using regex
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON block without markdown formatting
        json_pattern2 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern2, response_text, re.DOTALL)
        
        for json_str in json_matches:
            try:
                parsed = json.loads(json_str)
                # Verify it has expected structure
                if isinstance(parsed, dict) and any(key in parsed for key in ['issue_type', 'user_comment', 'summary']):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # If all else fails, raise an error
        raise json.JSONDecodeError("Could not extract valid JSON from response", response_text, 0)
    
    def _clean_user_comment(self, user_comment: str) -> str:
        """
        Clean and validate user comment, handling nested JSON structures
        
        Args:
            user_comment: Raw user comment string
            
        Returns:
            Cleaned user comment text
        """
        if not user_comment:
            return "I've analyzed this issue. Please let me know if you need any clarification."
        
        # Check if the user_comment itself contains JSON
        try:
            # Try to parse as JSON to see if it's a nested structure
            parsed_comment = json.loads(user_comment)
            if isinstance(parsed_comment, dict):
                # Extract the actual comment from nested JSON
                if "user_comment" in parsed_comment:
                    return self._clean_user_comment(parsed_comment["user_comment"])
                elif "message" in parsed_comment:
                    return str(parsed_comment["message"])
                elif "content" in parsed_comment:
                    return str(parsed_comment["content"])
                else:
                    # If it's a JSON object but no clear comment field, return a fallback
                    logger.warning(f"Nested JSON found in user_comment but no clear text field: {parsed_comment}")
                    return "I've completed the analysis of this issue. The results have been processed successfully."
        except json.JSONDecodeError:
            # Not JSON, treat as regular string
            pass
        
        # Clean up common formatting issues
        cleaned = str(user_comment).strip()
        
        # Remove markdown code blocks if they contain JSON
        import re
        json_block_pattern = r'```json\s*\{.*?\}\s*```'
        if re.search(json_block_pattern, cleaned, re.DOTALL):
            # This appears to be a JSON block, extract meaningful text
            logger.warning(f"Removing JSON code block from user comment: {cleaned[:100]}...")
            return "I've analyzed this issue and the results are available. Please let me know if you need any additional information."
        
        # Remove any remaining JSON-like structures
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        cleaned = re.sub(json_pattern, '', cleaned).strip()
        
        # Ensure we have some meaningful content
        if len(cleaned) < 10:
            return "I've completed the analysis of this issue. The details have been recorded successfully."
        
        return cleaned
    
    def _extract_user_comment_from_response(self, response_text: str) -> str:
        """
        Extract a user-friendly comment from raw LLM response when JSON parsing fails
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            User-friendly comment text
        """
        import re
        
        # Try to find any user_comment field in the text
        user_comment_pattern = r'"user_comment":\s*"([^"]*)"'
        match = re.search(user_comment_pattern, response_text)
        if match:
            return self._clean_user_comment(match.group(1))
        
        # Try to find analysis or summary text that could serve as a comment
        lines = response_text.split('\n')
        meaningful_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, JSON brackets, and technical formatting
            if (line and 
                not line.startswith('{') and 
                not line.startswith('}') and 
                not line.startswith('"') and
                not line.startswith('[') and
                not line.startswith(']') and
                len(line) > 20):  # Minimum length for meaningful content
                meaningful_lines.append(line)
        
        if meaningful_lines:
            # Take the first meaningful line as the comment
            comment = meaningful_lines[0]
            # Clean up any JSON fragments
            comment = re.sub(r'[{}"\[\]]', '', comment).strip()
            if len(comment) > 10:
                return comment
        
        # Fallback comment
        return "I've analyzed this issue. The analysis has been completed and the results are being processed."

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

    async def fetch_comment_data(self) -> Optional[Dict[str, Any]]:
        """Fetch comment data from GitHub API if event is comment-related"""
        try:
            # Only fetch comment data for comment events
            if self.event_name != "issue_comment":
                return None
            
            # Get comment ID from environment or GitHub event payload
            comment_id = os.environ.get("COMMENT_ID")
            if not comment_id:
                # Try to get from GITHUB_EVENT_PATH if available
                event_path = os.environ.get("GITHUB_EVENT_PATH")
                if event_path and os.path.exists(event_path):
                    import json
                    with open(event_path, 'r') as f:
                        event_data = json.load(f)
                        comment_id = event_data.get('comment', {}).get('id')
            
            if not comment_id:
                logger.warning("Comment ID not available for comment event")
                return None
            
            import requests
            
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/comments/{comment_id}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch comment data. Status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching comment data: {str(e)}")
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
            
            # Fetch comment data if applicable
            comment_data = await self.fetch_comment_data()
            
            # Get trigger decision
            trigger_decision = get_trigger_decision(issue_data, comment_data)
            
            # Store trigger decision in context
            self.analysis_context["trigger_decision"] = {
                "should_trigger": trigger_decision.should_trigger,
                "reason": trigger_decision.reason,
                "trigger_type": trigger_decision.trigger_type,
                "confidence": trigger_decision.confidence
            }
            
            # If we shouldn't trigger, log and exit early
            if not trigger_decision.should_trigger:
                logger.info(f"Not triggering analysis: {trigger_decision.reason}")
                return self.analysis_context
            
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
