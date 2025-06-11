# analyzer_core/utils/github_utils.py
"""
GitHub API utilities
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def fetch_issue_data(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch issue data from GitHub API"""
    try:
        url = f"https://api.github.com/repos/{config['repo_owner']}/{config['repo_name']}/issues/{config['issue_number']}"
        headers = {
            "Authorization": f"token {config['github_token']}",
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


async def fetch_comment_data(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch comment data from GitHub API if event is comment-related"""
    try:
        # Only fetch comment data for comment events
        if config.get("event_name") != "issue_comment":
            return None
        
        # Get comment ID from environment or GitHub event payload
        comment_id = os.environ.get("COMMENT_ID")
        if not comment_id:
            # Try to get from GITHUB_EVENT_PATH if available
            event_path = os.environ.get("GITHUB_EVENT_PATH")
            if event_path and os.path.exists(event_path):
                with open(event_path, 'r') as f:
                    event_data = json.load(f)
                    comment_id = event_data.get('comment', {}).get('id')
        
        if not comment_id:
            logger.warning("Comment ID not available for comment event")
            return None
        
        url = f"https://api.github.com/repos/{config['repo_owner']}/{config['repo_name']}/issues/comments/{comment_id}"
        headers = {
            "Authorization": f"token {config['github_token']}",
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
