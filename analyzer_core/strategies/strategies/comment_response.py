# analyzer_core/strategies/strategies/comment_response.py
"""
Comment Response Strategy

Handles responses to issue owner comments with context-aware analysis.
"""

import logging
from typing import Dict, Any, Optional, List
from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class CommentResponseStrategy(BaseStrategy):
    """
    Strategy for handling issue owner comments
    
    Focus:
    - Understanding comment context and intent
    - Building on conversation history
    - Providing contextual responses
    - Avoiding redundant information requests
    """
    
    def __init__(self):
        super().__init__("comment_response")
    
    async def analyze_context(self, 
                            issue_data: Dict[str, Any],
                            comment_data: Optional[Dict[str, Any]] = None,
                            trigger_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze context for issue owner comment using LLM-based analysis
        """
        
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        comment_body = comment_data.get('body', '') if comment_data else ''
        comment_author = comment_data.get('user', {}).get('login', '') if comment_data else ''
        issue_author = issue_data.get('user', {}).get('login', '')
        labels = [label.get('name', '') for label in issue_data.get('labels', [])]
        
        # Use LLM for context analysis with conversation awareness
        context_analysis = await self._llm_analyze_comment_context(
            title, body, comment_body, comment_author, issue_author, labels, trigger_context
        )
        
        # Add strategy metadata
        context_analysis.update({
            "strategy": "comment_response",
            "comment_author": comment_author,
            "issue_author": issue_author,
            "existing_labels": labels,
            "approach": "llm_driven_comment_analysis"
        })
        
        logger.info(f"LLM Comment Analysis: {context_analysis.get('comment_intent', 'unknown')} intent, "
                   f"{context_analysis.get('conversation_stage', 'unknown')} stage, "
                   f"{context_analysis.get('confidence', 0):.2f} confidence")
        return context_analysis
    
    async def select_tools(self, context_analysis: Dict[str, Any]) -> List[str]:
        """
        Select tools using LLM-based reasoning for comment responses
        """
        
        # Use LLM to select appropriate tools based on comment context
        selected_tools = await self._llm_select_comment_tools(context_analysis)
        
        logger.info(f"LLM selected {len(selected_tools)} tools for comment_response strategy: {selected_tools}")
        return selected_tools
    
    def customize_prompts(self, 
                        base_prompts: Dict[str, str],
                        context_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Customize prompts for comment response analysis
        """
        
        comment_intent = context_analysis.get("comment_intent", "unknown")
        conversation_stage = context_analysis.get("conversation_stage", "unknown")
        relationship = context_analysis.get("author_relationship", "unknown")
        
        customized = base_prompts.copy()
        
        # Create conversation-aware analysis prompt
        customized["analysis"] = f"""
You are analyzing a comment in an ongoing GitHub issue conversation. This comment shows {comment_intent} intent at the {conversation_stage} stage.

Context from Strategy Analysis:
{self._format_context_for_prompt(context_analysis)}

Your analysis should:
1. Build on the existing conversation context
2. Understand the comment's intent and any new information
3. Avoid redundant requests for information already provided
4. Consider the relationship between commenters
5. Focus on moving the conversation forward productively

Guidelines:
- For {comment_intent} comments: Respond appropriately to the specific intent
- At {conversation_stage} stage: Adjust approach for conversation maturity
- Relationship context: {relationship} - adjust tone and approach accordingly

Base prompt: {base_prompts.get('analysis', '')}
        """.strip()
          # Create conversation-aware final response prompt
        customized["final_response"] = f"""
You are responding to a comment in an ongoing GitHub issue conversation.

**CRITICAL**: You MUST acknowledge and respond to the specific comment that triggered this analysis.

Context Analysis Results:
{self._format_context_for_prompt(context_analysis)}

Response Guidelines:
1. **ACKNOWLEDGE THE TRIGGERING COMMENT**: Start by directly referencing what the user said in their comment
2. **BUILD ON THE CONVERSATION**: Show understanding of the conversation flow and previous exchanges  
3. **ADDRESS SPECIFIC INTENT**: Respond to the {comment_intent} intent shown in the comment
4. **CONVERSATIONAL STAGE AWARENESS**: This is the {conversation_stage} stage - adjust your approach accordingly
5. **PROVIDE CONTEXTUAL VALUE**: Offer insights that build on their specific comment content
6. **AVOID REDUNDANCY**: Don't ask for information they've already provided in the comment

**CRITICAL REQUIREMENTS**:
- Your user_comment MUST reference specific content from the triggering comment
- Show that you've read and understood what they wrote
- Respond conversationally, not with generic issue analysis
- If they provided new information, acknowledge it specifically
- If they asked a question, answer it directly
- If they expressed frustration, acknowledge their concern

Tone and Approach:
- Conversational and contextual (not robotic or templated)
- Acknowledging previous exchanges and the specific comment
- Focused on the comment's intent: {comment_intent}
- Appropriate for ongoing dialogue with {relationship}
- Show genuine engagement with their contribution

Base prompt: {base_prompts.get('final_response', '')}
        """.strip()
        
        return customized
    
    async def recommend_actions(self, 
                              analysis_results: Dict[str, Any],
                              context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend actions using LLM-based reasoning for comment responses
        """
        
        # Use LLM to recommend actions based on comment analysis
        recommended_actions = await self._llm_recommend_comment_actions(analysis_results, context_analysis)
        
        logger.info(f"LLM recommended {len(recommended_actions)} actions for comment response")
        return recommended_actions
    
    async def _llm_analyze_comment_context(self, title: str, body: str, comment_body: str,
                                         comment_author: str, issue_author: str, labels: List[str],
                                         trigger_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:        
        
        """
        Use LLM with chain of thought to analyze comment context
        """
        from LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        context_prompt = f"""
You are an expert conversation analyst for GitHub issues. Analyze this comment in the context of the ongoing issue conversation.

## Issue Information:
**Title:** {title}
**Issue Author:** {issue_author}
**Comment Author:** {comment_author}
**Existing Labels:** {', '.join(labels) if labels else 'None'}
**Trigger Context:** {trigger_context or 'Comment created'}

**Original Issue Body:**
{body}

**New Comment:**
{comment_body}

## Analysis Framework:

Use chain of thought reasoning to analyze:

### Step 1: Comment Intent Analysis
Determine what the commenter is trying to achieve:
- Providing additional information
- Asking for clarification
- Reporting progress/updates
- Expressing urgency or frustration
- Requesting specific help
- Providing feedback on suggestions

### Step 2: Conversation Stage Assessment
Analyze where this conversation stands:
- Initial exchange (first few comments)
- Active investigation (back-and-forth troubleshooting)
- Clarification stage (seeking/providing details)
- Resolution stage (testing solutions)
- Follow-up stage (confirming fixes)

### Step 3: Information Delta Analysis
Identify new information provided:
- New technical details
- Changed circumstances
- Additional context
- Test results
- Environment changes

### Step 4: Response Requirements
Determine what type of response is needed:
- Technical guidance
- Process clarification
- Acknowledgment and next steps
- Additional investigation
- Escalation needs

### Step 5: Relationship Context
Consider the relationship dynamics:
- Issue owner vs external contributor
- New user vs experienced contributor
- Maintainer vs community member

## Response Format:
Provide your analysis as JSON:
{{
    "reasoning": {{
        "intent_analysis": "Your analysis of the comment's intent",
        "conversation_analysis": "Your assessment of conversation stage",
        "information_analysis": "New information provided in the comment",
        "response_analysis": "What type of response is needed",
        "relationship_analysis": "Relationship context and dynamics"
    }},
    "conclusions": {{
        "comment_intent": "info_providing|question_asking|progress_update|help_request|feedback|urgency_expression",
        "conversation_stage": "initial|investigation|clarification|resolution|follow_up",
        "new_information_level": "significant|moderate|minimal|none",
        "response_urgency": "immediate|normal|low",
        "author_relationship": "issue_owner|external_contributor|maintainer|community_member",
        "needs_escalation": true/false,
        "conversation_sentiment": "positive|neutral|frustrated|confused"
    }},
    "confidence": 0.0-1.0,
    "key_insights": ["insight1", "insight2", ...],
    "response_priorities": ["priority1", "priority2", ...]
}}

Focus on understanding the conversational context and what would be most helpful.
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
                result["response_priorities"] = analysis.get("response_priorities", [])
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM comment analysis response as JSON, using fallback")
                return self._fallback_comment_analysis()
                
        except Exception as e:
            logger.error(f"LLM comment analysis failed: {str(e)}")
            return self._fallback_comment_analysis()
    async def _llm_select_comment_tools(self, context_analysis: Dict[str, Any]) -> List[str]:
        """
        Use LLM to select appropriate tools for comment analysis
        """
        from LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        available_tools = [
            "rag_search", "similar_issues", "regression_analysis", "code_search",
            "documentation_lookup", "image_analysis", "template_generation"
        ]
        
        tool_selection_prompt = f"""
You are selecting analysis tools for responding to a GitHub issue comment. Consider the conversation context and avoid redundancy.

## Comment Context Analysis:
{json.dumps(context_analysis, indent=2)}

## Available Tools:
1. **rag_search**: Search knowledge base for relevant information and solutions
2. **similar_issues**: Find similar historical issues and their resolutions
3. **regression_analysis**: Analyze if this might be a regression from recent changes
4. **code_search**: Search codebase for relevant code patterns and implementations
5. **documentation_lookup**: Search documentation for relevant guides and references
6. **image_analysis**: Analyze screenshots or visual content in the comment
7. **template_generation**: Generate templates to help users provide better information

## Selection Criteria for Comments:

### Chain of Thought Process:

**Step 1: Conversation Context**
- What stage is this conversation at?
- What information has already been gathered?
- What's the comment intent?

**Step 2: Avoid Redundancy**
- Don't re-gather information already obtained
- Focus on new needs revealed by the comment
- Consider conversation progression

**Step 3: Comment-Specific Needs**
- New information provided: May need verification tools
- Questions asked: May need documentation/knowledge tools
- Technical issues: May need code search or regression analysis
- Progress updates: May need minimal additional tools

**Step 4: Efficient Selection**
- Prioritize tools that address the specific comment
- Limit to 1-3 tools for comment responses
- Focus on moving conversation forward

## Response Format:
{{
    "reasoning": {{
        "conversation_context": "How conversation stage affects tool selection",
        "comment_specific_needs": "What the comment specifically requires",
        "redundancy_avoidance": "How you avoided redundant tool usage",
        "efficiency_rationale": "Why this minimal set is sufficient"
    }},
    "selected_tools": ["tool1", "tool2", ...],
    "tool_priorities": {{
        "tool1": 1,
        "tool2": 2
    }},
    "expected_value": "What value these tools will add to the conversation"
}}

Focus on tools that specifically address the comment's needs without redundancy.
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
                return valid_tools if valid_tools else self._fallback_comment_tools()
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM tool selection response, using fallback")
                return self._fallback_comment_tools()
                
        except Exception as e:
            logger.error(f"LLM tool selection failed: {str(e)}")
            return self._fallback_comment_tools()
    async def _llm_recommend_comment_actions(self, analysis_results: Dict[str, Any], 
                                           context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to recommend actions for comment responses
        """
        from LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        action_prompt = f"""
You are recommending actions for responding to a GitHub issue comment. Consider the conversation context and progression.

## Comment Context Analysis:
{json.dumps(context_analysis, indent=2)}

## Analysis Results:
{json.dumps(analysis_results, indent=2)}

## Available Actions:
1. **add_comment**: Post a helpful response to the comment
2. **add_label**: Add appropriate labels based on new information
3. **assign_reviewer**: Assign to expert reviewer if needed
4. **request_info**: Request additional information (use sparingly)
5. **close_issue**: Close if resolved or duplicate
6. **update_status**: Update issue status/priority
7. **escalate**: Escalate based on urgency or complexity

## Action Selection Framework for Comments:

### Step 1: Primary Response
- Almost always include add_comment for direct response
- Focus on addressing the specific comment intent

### Step 2: Conversation Progression
- Consider what moves the conversation forward
- Avoid actions that repeat previous requests
- Focus on new needs revealed by the comment

### Step 3: Status Updates
- Update labels if new information changes classification
- Consider status changes if progress is made
- Escalate if urgency increases

### Step 4: Efficiency
- Limit actions to what's truly needed
- Prioritize conversation flow over administrative tasks

## Response Format:
{{
    "reasoning": {{
        "response_strategy": "Your strategy for responding to this comment",
        "conversation_progression": "How actions move conversation forward",
        "administrative_needs": "Any needed status/classification updates",
        "efficiency_considerations": "Why this action set is optimal"
    }},
    "recommended_actions": [
        {{
            "action": "action_name",
            "priority": 1-5,
            "details": "Specific details for this action",
            "rationale": "Why this action addresses the comment"
        }}
    ],
    "conversation_impact": "How these actions will impact the ongoing conversation"
}}

Focus on actions that directly respond to the comment and move the conversation forward.
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
                return recommended_actions if recommended_actions else self._fallback_comment_actions()
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM action recommendations, using fallback")
                return self._fallback_comment_actions()
                
        except Exception as e:
            logger.error(f"LLM action recommendation failed: {str(e)}")
            return self._fallback_comment_actions()
    
    def _format_context_for_prompt(self, context_analysis: Dict[str, Any]) -> str:
        """Format context analysis for use in prompts"""
        formatted = []
        for key, value in context_analysis.items():
            if key not in ['strategy', 'comment_author', 'issue_author', 'existing_labels', 'approach']:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    def _fallback_comment_analysis(self) -> Dict[str, Any]:
        """Fallback comment analysis if LLM fails"""
        return {
            "comment_intent": "question_asking",
            "conversation_stage": "investigation",
            "new_information_level": "moderate",
            "response_urgency": "normal",
            "author_relationship": "issue_owner",
            "confidence": 0.5,
            "key_insights": ["Fallback analysis used"],
            "response_priorities": ["Provide helpful response"]
        }
    
    def _fallback_comment_tools(self) -> List[str]:
        """Fallback tool selection for comments if LLM fails"""
        return ["rag_search"]
    
    def _fallback_comment_actions(self) -> List[Dict[str, Any]]:
        """Fallback actions for comments if LLM fails"""
        return [
            {
                "action": "add_comment",
                "priority": 1,
                "details": "Respond to comment helpfully",
                "rationale": "Default action for comment engagement"
            }
        ]
