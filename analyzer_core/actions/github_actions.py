# analyzer_core/actions/github_actions.py
"""
GitHub API interaction handler
"""

import requests
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class GitHubActionExecutor:
    """Handles GitHub API interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.token = config["github_token"]
        self.repo_owner = config["repo_owner"]
        self.repo_name = config["repo_name"]
        self.issue_number = config["issue_number"]
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    async def add_labels(self, labels: List[str]) -> bool:
        """Add labels to an issue"""
        try:
            if not labels:
                return True
                
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}/labels"
            response = requests.post(url, headers=self.headers, json={"labels": labels})
            
            if response.status_code == 200:
                logger.info(f"Successfully added labels: {labels}")
                return True
            else:
                logger.error(f"Failed to add labels. Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error adding labels: {str(e)}")
            return False
    
    async def add_comment(self, comment: str) -> bool:
        """Add a comment to an issue"""
        try:
            if not comment.strip():
                return True
                
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}/comments"
            response = requests.post(url, headers=self.headers, json={"body": comment})
            
            if response.status_code == 201:
                logger.info("Successfully added comment to issue")
                return True
            else:
                logger.error(f"Failed to add comment. Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            return False
    
    async def assign_user(self, username: str) -> bool:
        """Assign a user to an issue"""
        try:
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}/assignees"
            response = requests.post(url, headers=self.headers, json={"assignees": [username]})
            
            if response.status_code == 201:
                logger.info(f"Successfully assigned user: {username}")
                return True
            else:
                logger.error(f"Failed to assign user. Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error assigning user: {str(e)}")
            return False
    
    async def close_issue(self, reason: str = "") -> bool:
        """Close an issue"""
        try:
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.issue_number}"
            data = {"state": "closed"}
            
            if reason:
                data["state_reason"] = reason
                
            response = requests.patch(url, headers=self.headers, json=data)
            
            if response.status_code == 200:
                logger.info(f"Successfully closed issue with reason: {reason}")
                return True
            else:
                logger.error(f"Failed to close issue. Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error closing issue: {str(e)}")
            return False
