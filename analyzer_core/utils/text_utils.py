# analyzer_core/utils/text_utils.py
"""
Text processing utilities
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def assess_user_provided_information(issue_data: Dict[str, Any]) -> Dict[str, Any]:
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
            'has_screenshots': has_images_in_text(body),
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


def has_images_in_text(text: str) -> bool:
    """Check if text contains image references"""
    image_patterns = [
        r'!\[.*?\]\(.*?\)',  # Markdown images
        r'<img.*?>',  # HTML img tags
        r'https?://[^\s]+\.(png|jpg|jpeg|gif|svg)',  # Direct image URLs
    ]
    
    for pattern in image_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def prepare_issue_context(issue_data: Dict[str, Any], event_name: str = "", event_action: str = "") -> str:
    """Prepare issue context for LLM analysis"""
    title = issue_data.get('title', 'No title')
    body = issue_data.get('body', 'No description')
    labels = [label.get('name', '') for label in issue_data.get('labels', [])]
    comments_count = issue_data.get('comments', 0)
    
    # Check for images
    has_images = has_images_in_text(body)
    
    context = f"""
**Title:** {title}

**Description:** {body}

**Labels:** {', '.join(labels) if labels else 'None'}

**Comments:** {comments_count}

**Contains Images:** {'Yes' if has_images else 'No'}

**Event Type:** {event_name} - {event_action}
"""
    return context
