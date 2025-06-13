# analyzer_core/strategies/strategies/agent_mention.py
"""
Agent Mention Strategy

Handles scenarios where the agent is explicitly mentioned by non-owners.
"""

import logging
from typing import Dict, Any, Optional, List
from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AgentMentionStrategy(BaseStrategy):
    """
    Strategy for handling explicit agent mentions by non-issue owners
    
    Focus:
    - Understanding why the agent was mentioned
    - Providing helpful assistance to community members
    - Facilitating collaboration between users
    - Respectful engagement with non-owners
    """
    
    def __init__(self):
        super().__init__("agent_mention")
    
    async def analyze_context(self, 
                            issue_data: Dict[str, Any],
                            comment_data: Optional[Dict[str, Any]] = None,
                            trigger_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze context for agent mention using LLM-based analysis
        """
        
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        comment_body = comment_data.get('body', '') if comment_data else ''
        comment_author = comment_data.get('user', {}).get('login', '') if comment_data else ''
        issue_author = issue_data.get('user', {}).get('login', '')
        labels = [label.get('name', '') for label in issue_data.get('labels', [])]
        
        # Use LLM for context analysis with mention-specific considerations
        context_analysis = await self._llm_analyze_mention_context(
            title, body, comment_body, comment_author, issue_author, labels, trigger_context
        )
        
        # Add strategy metadata
        context_analysis.update({
            "strategy": "agent_mention",
            "comment_author": comment_author,
            "issue_author": issue_author,
            "existing_labels": labels,
            "approach": "llm_driven_mention_analysis"
        })
        
        logger.info(f"LLM Mention Analysis: {context_analysis.get('mention_reason', 'unknown')} reason, "
                   f"{context_analysis.get('user_role', 'unknown')} user, "
                   f"{context_analysis.get('confidence', 0):.2f} confidence")
        return context_analysis
    
    async def select_tools(self, context_analysis: Dict[str, Any]) -> List[str]:
        """
        Select tools using LLM-based reasoning for agent mentions
        """
        
        # Use LLM to select appropriate tools based on mention context
        selected_tools = await self._llm_select_mention_tools(context_analysis)
        
        logger.info(f"LLM selected {len(selected_tools)} tools for agent_mention strategy: {selected_tools}")
        return selected_tools
    
    def customize_prompts(self, 
                        base_prompts: Dict[str, str],
                        context_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Customize prompts for agent mention responses
        """
        
        mention_reason = context_analysis.get("mention_reason", "unknown")
        user_role = context_analysis.get("user_role", "unknown")
        collaboration_type = context_analysis.get("collaboration_type", "unknown")
        
        customized = base_prompts.copy()
        
        # Create mention-aware analysis prompt
        customized["analysis"] = f"""
You are analyzing an agent mention in a GitHub issue. The agent was mentioned for {mention_reason} by a {user_role}.

Context from Strategy Analysis:
{self._format_context_for_prompt(context_analysis)}

Your analysis should:
1. Understand why the agent was mentioned and by whom
2. Consider the appropriateness of agent involvement
3. Respect both issue owner and community member perspectives
4. Focus on facilitating helpful collaboration
5. Be mindful of community dynamics and etiquette

Guidelines:
- Mention reason: {mention_reason} - Address the specific need
- User role: {user_role} - Adjust approach for community dynamics
- Collaboration type: {collaboration_type} - Facilitate appropriate interaction

Base prompt: {base_prompts.get('analysis', '')}
        """.strip()
        
        # Create mention-aware final response prompt
        customized["final_response"] = f"""
You are responding to an agent mention in a GitHub issue by a {user_role} for {mention_reason}.

Context Analysis Results:
{self._format_context_for_prompt(context_analysis)}

Response Guidelines:
1. Acknowledge the mention respectfully and helpfully
2. Address the specific reason for the mention: {mention_reason}
3. Facilitate collaboration between community members and issue owner
4. Provide value while respecting issue ownership
5. Be helpful without overstepping boundaries
6. Encourage positive community interaction

Tone and Approach:
- Respectful and welcoming to community involvement
- Helpful while deferring to issue owner when appropriate
- Professional and collaborative
- Encouraging of community participation

Base prompt: {base_prompts.get('final_response', '')}
        """.strip()
        
        return customized
    
    async def recommend_actions(self, 
                              analysis_results: Dict[str, Any],
                              context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend actions using LLM-based reasoning for agent mentions
        """
        
        # Use LLM to recommend actions based on mention analysis
        recommended_actions = await self._llm_recommend_mention_actions(analysis_results, context_analysis)
        
        logger.info(f"LLM recommended {len(recommended_actions)} actions for agent mention")
        return recommended_actions
    
    async def _llm_analyze_mention_context(self, title: str, body: str, comment_body: str,
                                         comment_author: str, issue_author: str, labels: List[str],
                                         trigger_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:        
        
        """
        Use LLM with chain of thought to analyze agent mention context
        """
        from LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        context_prompt = f"""
You are an expert at analyzing agent mentions in GitHub issues. Understand why the agent was mentioned and how to respond appropriately.

## Issue Information:
**Title:** {title}
**Issue Author:** {issue_author}
**Comment Author:** {comment_author}
**Existing Labels:** {', '.join(labels) if labels else 'None'}
**Trigger Context:** {trigger_context or 'Agent mentioned'}

**Original Issue Body:**
{body}

**Comment with Agent Mention:**
{comment_body}

## Analysis Framework:

Use chain of thought reasoning to analyze:

### Step 1: Mention Reason Analysis
Determine why the agent was mentioned:
- Seeking help with the issue
- Asking for clarification or guidance
- Requesting agent analysis
- Trying to get attention for the issue
- Providing additional information for the agent
- Asking about automated processes

### Step 2: User Role Assessment
Analyze the mentioning user's role:
- Community contributor helping with the issue
- Experienced user providing guidance
- New user seeking assistance
- Maintainer or collaborator
- Random user passing by

### Step 3: Collaboration Context
Understand the collaborative context:
- Is this helpful community involvement?
- Does this complement or conflict with issue owner needs?
- Is the mention appropriate and constructive?
- What's the best way to facilitate collaboration?

### Step 4: Response Appropriateness
Determine appropriate level of response:
- Full analysis and assistance
- Guidance and facilitation
- Acknowledgment and deferral to issue owner
- Educational response about agent usage

### Step 5: Community Impact
Consider broader community implications:
- How does this affect issue ownership?
- What does this teach about community collaboration?
- How can this encourage positive community behavior?

## Response Format:
Provide your analysis as JSON:
{{
    "reasoning": {{
        "mention_reason_analysis": "Why the agent was mentioned",
        "user_role_analysis": "Assessment of the mentioning user's role",
        "collaboration_analysis": "How this fits into issue collaboration",
        "appropriateness_analysis": "Whether and how the agent should respond",
        "community_impact_analysis": "Broader implications for community"
    }},
    "conclusions": {{
        "mention_reason": "help_seeking|guidance_request|analysis_request|attention_seeking|info_providing|process_inquiry",
        "user_role": "community_contributor|experienced_user|new_user|maintainer|casual_user",
        "collaboration_type": "helpful_contribution|appropriate_assistance|boundary_crossing|unclear_intent",
        "response_level": "full_engagement|guided_assistance|acknowledgment_only|educational_response",
        "community_appropriateness": "highly_appropriate|appropriate|neutral|inappropriate",
        "issue_owner_consideration": "helpful_to_owner|neutral_to_owner|potentially_conflicting"
    }},
    "confidence": 0.0-1.0,
    "key_insights": ["insight1", "insight2", ...],
    "response_guidelines": ["guideline1", "guideline2", ...]
}}

Focus on understanding the community dynamics and how to respond helpfully while respecting issue ownership.
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
                result["response_guidelines"] = analysis.get("response_guidelines", [])
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM mention analysis response as JSON, using fallback")
                return self._fallback_mention_analysis()
                
        except Exception as e:
            logger.error(f"LLM mention analysis failed: {str(e)}")
            return self._fallback_mention_analysis()
    async def _llm_select_mention_tools(self, context_analysis: Dict[str, Any]) -> List[str]:
        """
        Use LLM to select appropriate tools for agent mention responses
        """
        from LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        available_tools = [
            "rag_search", "similar_issues", "regression_analysis", "code_search",
            "documentation_lookup", "image_analysis", "template_generation"
        ]
        
        tool_selection_prompt = f"""
You are selecting analysis tools for responding to an agent mention. Consider community dynamics and appropriateness.

## Mention Context Analysis:
{json.dumps(context_analysis, indent=2)}

## Available Tools:
1. **rag_search**: Search knowledge base for relevant information and solutions
2. **similar_issues**: Find similar historical issues and their resolutions
3. **regression_analysis**: Analyze if this might be a regression from recent changes
4. **code_search**: Search codebase for relevant code patterns and implementations
5. **documentation_lookup**: Search documentation for relevant guides and references
6. **image_analysis**: Analyze screenshots or visual content in the mention
7. **template_generation**: Generate templates to help users provide better information

## Selection Criteria for Agent Mentions:

### Chain of Thought Process:

**Step 1: Mention Appropriateness**
- Is this a legitimate request for agent assistance?
- What level of analysis is warranted?
- Should the agent provide full analysis or guidance?

**Step 2: Community Context**
- How does this serve the community and issue owner?
- What tools provide value without overstepping?
- How can tools facilitate collaboration?

**Step 3: Response Level**
- Full engagement: Use comprehensive tools
- Guided assistance: Use educational/reference tools
- Acknowledgment: Minimal or no tools
- Educational: Use documentation and guidance tools

**Step 4: Efficiency and Respect**
- Don't over-analyze if simple guidance suffices
- Consider what tools respect issue ownership
- Focus on tools that help the mentioning user help themselves

## Response Format:
{{
    "reasoning": {{
        "appropriateness_assessment": "Whether full tool usage is appropriate",
        "community_consideration": "How tool selection serves community needs",
        "response_level_rationale": "Why this level of tool engagement is right",
        "efficiency_and_respect": "How tool selection balances help with boundaries"
    }},
    "selected_tools": ["tool1", "tool2", ...],
    "tool_priorities": {{
        "tool1": 1,
        "tool2": 2
    }},
    "community_value": "How these tools serve the broader community and issue owner"
}}

Focus on tools that provide appropriate assistance while respecting community dynamics.
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
                return valid_tools if valid_tools else self._fallback_mention_tools()
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM tool selection response, using fallback")
                return self._fallback_mention_tools()
                
        except Exception as e:
            logger.error(f"LLM tool selection failed: {str(e)}")
            return self._fallback_mention_tools()
    async def _llm_recommend_mention_actions(self, analysis_results: Dict[str, Any], 
                                           context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to recommend actions for agent mention responses
        """
        from LLM.llm_provider import get_llm
        from langchain.schema import HumanMessage
        import json
        
        action_prompt = f"""
You are recommending actions for responding to an agent mention. Consider community dynamics and appropriate boundaries.

## Mention Context Analysis:
{json.dumps(context_analysis, indent=2)}

## Analysis Results:
{json.dumps(analysis_results, indent=2)}

## Available Actions:
1. **add_comment**: Post a helpful response to the mention
2. **add_label**: Add appropriate labels if justified
3. **assign_reviewer**: Assign to expert reviewer if needed
4. **notify_owner**: Notify issue owner of community involvement
5. **close_issue**: Close if resolved (rarely appropriate for mentions)
6. **facilitate_collaboration**: Take actions that help community members work together
7. **educational_response**: Provide guidance on community processes

## Action Selection Framework for Agent Mentions:

### Step 1: Primary Response
- Almost always include add_comment for acknowledging the mention
- Focus on being helpful while respecting boundaries

### Step 2: Community Facilitation
- Consider actions that help community members collaborate
- Facilitate positive community involvement
- Respect issue ownership while being welcoming

### Step 3: Appropriate Boundaries
- Don't take administrative actions unless clearly beneficial
- Avoid actions that override issue owner preferences
- Focus on guidance and facilitation over control

### Step 4: Educational Opportunity
- Use mentions as opportunities to educate about community processes
- Help users understand how to effectively engage

## Response Format:
{{
    "reasoning": {{
        "response_strategy": "Your strategy for responding to the mention",
        "community_facilitation": "How to facilitate positive community involvement",
        "boundary_considerations": "How to respect appropriate boundaries",
        "educational_opportunities": "What this can teach about community engagement"
    }},
    "recommended_actions": [
        {{
            "action": "action_name",
            "priority": 1-5,
            "details": "Specific details for this action",
            "rationale": "Why this action is appropriate for a mention"
        }}
    ],
    "community_impact": "How these actions will impact community dynamics and collaboration"
}}

Focus on actions that acknowledge the mention helpfully while fostering positive community engagement.
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
                return recommended_actions if recommended_actions else self._fallback_mention_actions()
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM action recommendations, using fallback")
                return self._fallback_mention_actions()
                
        except Exception as e:
            logger.error(f"LLM action recommendation failed: {str(e)}")
            return self._fallback_mention_actions()
    
    def _format_context_for_prompt(self, context_analysis: Dict[str, Any]) -> str:
        """Format context analysis for use in prompts"""
        formatted = []
        for key, value in context_analysis.items():
            if key not in ['strategy', 'comment_author', 'issue_author', 'existing_labels', 'approach']:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    def _fallback_mention_analysis(self) -> Dict[str, Any]:
        """Fallback mention analysis if LLM fails"""
        return {
            "mention_reason": "help_seeking",
            "user_role": "community_contributor",
            "collaboration_type": "helpful_contribution",
            "response_level": "guided_assistance",
            "community_appropriateness": "appropriate",
            "confidence": 0.5,
            "key_insights": ["Fallback analysis used"],
            "response_guidelines": ["Provide helpful guidance"]
        }
    
    def _fallback_mention_tools(self) -> List[str]:
        """Fallback tool selection for mentions if LLM fails"""
        return ["documentation_lookup"]
    
    def _fallback_mention_actions(self) -> List[Dict[str, Any]]:
        """Fallback actions for mentions if LLM fails"""
        return [
            {
                "action": "add_comment",
                "priority": 1,
                "details": "Acknowledge mention and provide guidance",
                "rationale": "Default action for agent mention engagement"
            }
        ]
