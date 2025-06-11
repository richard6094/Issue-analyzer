#!/usr/bin/env python3
"""
Trigger Logic for Issue Agent

This module defines the logic for when the issue agent should be triggered
to avoid unnecessary processing and bot loops.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TriggerDecision:
    """Decision on whether to trigger the agent"""
    should_trigger: bool
    reason: str
    trigger_type: str = "none"  # "issue_created", "owner_comment", "agent_mention", "none"
    confidence: float = 1.0


class TriggerLogic:
    """
    Intelligent trigger logic that determines when the issue agent should activate
    
    Key principles:
    1. Always trigger on issue creation (unless explicitly disabled)
    2. Always trigger on issue owner comments (unless disabled)
    3. Only trigger on non-owner comments if agent is explicitly mentioned
    4. Never trigger on bot comments
    5. Respect user disable/enable commands
    """
    
    def __init__(self):
        """Initialize trigger logic"""
        self.bot_keywords = [
            "[bot]",
            "github-actions",
            "dependabot", 
            "codecov",
            "renovate",
            "greenkeeper",
            "snyk-bot",
            "microsoft-github-policy-service"
        ]
        
        self.agent_mention_patterns = [
            r"@issue-agent",
            r"@github-actions\[bot\]",
            r"@issue[\-_]?agent",
            r"issue[\-_]?agent\s+please",
            r"hey\s+agent",
            r"could\s+the\s+agent"
        ]
        
        self.disable_patterns = [
            r"@issue-agent\s+disable",
            r"@issue-agent\s+stop",
            r"@issue-agent\s+quiet",
            r"@issue-agent\s+silent",
            r"disable\s+issue[\-_]?agent",
            r"stop\s+issue[\-_]?agent"
        ]
        
        self.enable_patterns = [
            r"@issue-agent\s+enable",
            r"@issue-agent\s+start",
            r"@issue-agent\s+resume",
            r"enable\s+issue[\-_]?agent",
            r"start\s+issue[\-_]?agent"
        ]

    def should_trigger(self, 
                      event_name: str,
                      event_action: str,
                      issue_data: Dict[str, Any],
                      comment_data: Optional[Dict[str, Any]] = None) -> TriggerDecision:
        """
        Main decision logic for trigger conditions
        
        Args:
            event_name: GitHub event name (issues, issue_comment, etc.)
            event_action: GitHub event action (opened, created, etc.)
            issue_data: GitHub issue data
            comment_data: GitHub comment data (if applicable)
            
        Returns:
            TriggerDecision with should_trigger flag and reasoning
        """
        try:
            # Handle different event types
            if event_name == "issues" and event_action == "opened":
                return self._handle_issue_created(issue_data)
            
            elif event_name == "issue_comment" and event_action == "created":
                return self._handle_comment_created(issue_data, comment_data)
            
            elif event_name == "workflow_dispatch":
                return TriggerDecision(
                    should_trigger=True,
                    reason="Manual workflow dispatch - always trigger",
                    trigger_type="manual_dispatch",
                    confidence=1.0
                )
            
            else:
                return TriggerDecision(
                    should_trigger=False,
                    reason=f"Event {event_name}.{event_action} not configured for triggering",
                    trigger_type="none",
                    confidence=1.0
                )
                
        except Exception as e:
            logger.error(f"Error in trigger decision: {str(e)}")
            return TriggerDecision(
                should_trigger=False,
                reason=f"Error in trigger logic: {str(e)}",
                trigger_type="error",
                confidence=0.0
            )

    def _handle_issue_created(self, issue_data: Dict[str, Any]) -> TriggerDecision:
        """Handle issue creation trigger logic"""
        try:
            issue_title = issue_data.get('title', '').lower()
            issue_body = issue_data.get('body', '').lower()
            issue_author = issue_data.get('user', {}).get('login', '')
            
            # Check if author is a bot
            if self._is_bot_user(issue_author):
                return TriggerDecision(
                    should_trigger=False,
                    reason=f"Issue created by bot user: {issue_author}",
                    trigger_type="none",
                    confidence=1.0
                )
            
            # Check for explicit disable in issue content
            issue_content = f"{issue_title} {issue_body}"
            if self._check_disable_patterns(issue_content):
                return TriggerDecision(
                    should_trigger=False,
                    reason="Issue contains explicit agent disable command",
                    trigger_type="none",
                    confidence=1.0
                )
            
            # Default: trigger on all human-created issues
            return TriggerDecision(
                should_trigger=True,
                reason="New issue created by human user - default trigger",
                trigger_type="issue_created",
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Error handling issue creation: {str(e)}")
            return TriggerDecision(
                should_trigger=False,
                reason=f"Error processing issue creation: {str(e)}",
                trigger_type="error",
                confidence=0.0
            )

    def _handle_comment_created(self, 
                               issue_data: Dict[str, Any], 
                               comment_data: Dict[str, Any]) -> TriggerDecision:
        """Handle comment creation trigger logic"""
        try:
            if not comment_data:
                return TriggerDecision(
                    should_trigger=False,
                    reason="No comment data provided",
                    trigger_type="none",
                    confidence=1.0
                )
            
            comment_author = comment_data.get('user', {}).get('login', '')
            comment_body = comment_data.get('body', '').lower()
            issue_author = issue_data.get('user', {}).get('login', '')
            
            # 1. Never trigger on bot comments
            if self._is_bot_user(comment_author):
                return TriggerDecision(
                    should_trigger=False,
                    reason=f"Comment from bot user: {comment_author}",
                    trigger_type="none",
                    confidence=1.0
                )
            
            # 2. Check for disable/enable commands first (highest priority)
            if self._check_disable_patterns(comment_body):
                return TriggerDecision(
                    should_trigger=True,
                    reason="Disable command detected - trigger to process command",
                    trigger_type="agent_command",
                    confidence=1.0
                )
            
            if self._check_enable_patterns(comment_body):
                return TriggerDecision(
                    should_trigger=True,
                    reason="Enable command detected - trigger to process command",
                    trigger_type="agent_command",
                    confidence=1.0
                )
            
            # 3. Check if agent is disabled for this issue (via issue labels or previous commands)
            if self._is_agent_disabled_for_issue(issue_data):
                return TriggerDecision(
                    should_trigger=False,
                    reason="Agent is disabled for this issue",
                    trigger_type="none",
                    confidence=1.0
                )
            
            # 4. Always trigger on issue owner comments (unless disabled)
            if comment_author == issue_author:
                return TriggerDecision(
                    should_trigger=True,
                    reason="Comment from issue owner - default trigger",
                    trigger_type="owner_comment",
                    confidence=1.0
                )
            
            # 5. For non-owner comments, only trigger if agent is explicitly mentioned
            if self._check_agent_mention(comment_body):
                return TriggerDecision(
                    should_trigger=True,
                    reason="Agent explicitly mentioned by non-owner user",
                    trigger_type="agent_mention",
                    confidence=0.9
                )
            
            # 6. Default: don't trigger on non-owner comments without mention
            return TriggerDecision(
                should_trigger=False,
                reason=f"Comment from non-owner user ({comment_author}) without agent mention",
                trigger_type="none",
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Error handling comment creation: {str(e)}")
            return TriggerDecision(
                should_trigger=False,
                reason=f"Error processing comment: {str(e)}",
                trigger_type="error",
                confidence=0.0
            )

    def _is_bot_user(self, username: str) -> bool:
        """Check if a username belongs to a bot"""
        if not username:
            return False
        
        username_lower = username.lower()
        
        # Check for bot keywords in username
        for keyword in self.bot_keywords:
            if keyword in username_lower:
                return True
        
        return False

    def _check_agent_mention(self, text: str) -> bool:
        """Check if the agent is mentioned in the text"""
        for pattern in self.agent_mention_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_disable_patterns(self, text: str) -> bool:
        """Check for agent disable patterns"""
        for pattern in self.disable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_enable_patterns(self, text: str) -> bool:
        """Check for agent enable patterns"""
        for pattern in self.enable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_agent_disabled_for_issue(self, issue_data: Dict[str, Any]) -> bool:
        """
        Check if the agent is disabled for this specific issue
        
        This can be implemented by:
        1. Checking for specific labels (e.g., "agent-disabled")
        2. Looking for disable commands in previous comments
        3. Checking issue metadata
        """
        try:
            # Check labels for disable indicator
            labels = issue_data.get('labels', [])
            for label in labels:
                label_name = label.get('name', '').lower()
                if 'agent-disabled' in label_name or 'no-agent' in label_name:
                    return True
            
            # TODO: Could extend this to check previous comments for disable commands
            # This would require fetching comment history
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if agent is disabled: {str(e)}")
            return False

    def get_environment_context(self) -> Dict[str, Any]:
        """Get trigger-relevant environment context"""
        return {
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
            "event_action": os.environ.get("GITHUB_EVENT_ACTION"), 
            "sender_login": os.environ.get("SENDER_LOGIN"),
            "sender_type": os.environ.get("SENDER_TYPE"),
            "issue_number": os.environ.get("ISSUE_NUMBER"),
            "repo_full_name": os.environ.get("REPO_FULL_NAME"),
            "force_analysis": os.environ.get("FORCE_ANALYSIS", "false").lower() == "true"
        }

    def log_trigger_decision(self, decision: TriggerDecision, context: Dict[str, Any] = None):
        """Log the trigger decision for debugging"""
        context = context or {}
        
        log_level = logging.INFO if decision.should_trigger else logging.DEBUG
        
        logger.log(
            log_level,
            f"TRIGGER DECISION: {decision.should_trigger} "
            f"({decision.trigger_type}) - {decision.reason} "
            f"[confidence: {decision.confidence:.2f}] "
            f"[context: {context.get('event_name', 'unknown')}.{context.get('event_action', 'unknown')}]"
        )


# Global instance for easy import
trigger_logic = TriggerLogic()


def get_trigger_decision(issue_data: Dict[str, Any], 
                        comment_data: Optional[Dict[str, Any]] = None) -> TriggerDecision:
    """
    Convenience function to get trigger decision with environment context
    
    Args:
        issue_data: GitHub issue data
        comment_data: GitHub comment data (optional)
        
    Returns:
        TriggerDecision
    """
    context = trigger_logic.get_environment_context()
    
    # Handle force analysis override
    if context.get("force_analysis", False):
        decision = TriggerDecision(
            should_trigger=True,
            reason="Force analysis flag enabled - override all conditions",
            trigger_type="force_override",
            confidence=1.0
        )
    else:
        decision = trigger_logic.should_trigger(
            event_name=context.get("event_name", ""),
            event_action=context.get("event_action", ""),
            issue_data=issue_data,
            comment_data=comment_data
        )
    
    # Log the decision
    trigger_logic.log_trigger_decision(decision, context)
    
    return decision
