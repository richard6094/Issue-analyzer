"""
GitHub API Client for Smart Function Dispatcher

This module provides GitHub API integration for executing operations
like adding labels, comments, and managing issues.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GitHubAPIConfig:
    """Configuration for GitHub API client"""
    token: str
    base_url: str = "https://api.github.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class GitHubAPIClient:
    """
    Async GitHub API client for issue management operations
    """
    
    def __init__(self, config: GitHubAPIConfig = None):
        """Initialize GitHub API client"""
        if config:
            self.config = config
        else:
            # Initialize from environment variables
            token = os.environ.get('GITHUB_TOKEN')
            if not token:
                raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")
            
            self.config = GitHubAPIConfig(token=token)
        
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                'Authorization': f'token {self.config.token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'GitHub-Issue-Agent/1.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(
        self,
        method: str,
        url: str,
        data: Dict = None,
        params: Dict = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to GitHub API with retry logic
        
        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            url: API endpoint URL
            data: Request body data
            params: Query parameters
            
        Returns:
            Dict: Response data
            
        Raises:
            GitHubAPIError: On API errors
        """
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        full_url = f"{self.config.base_url}{url}" if not url.startswith('http') else url
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(
                    method=method,
                    url=full_url,
                    json=data,
                    params=params
                ) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else {}
                    
                    if response.status >= 200 and response.status < 300:
                        return response_data
                    elif response.status >= 400:
                        error_message = response_data.get('message', f'HTTP {response.status}')
                        
                        # Check for rate limiting
                        if response.status == 403 and 'rate limit' in error_message.lower():
                            reset_time = response.headers.get('X-RateLimit-Reset', '0')
                            logger.warning(f"Rate limited. Reset at: {reset_time}")
                            
                            # Wait for rate limit reset if it's soon
                            import time
                            current_time = int(time.time())
                            reset_timestamp = int(reset_time)
                            
                            if reset_timestamp - current_time < 300:  # Wait up to 5 minutes
                                wait_time = reset_timestamp - current_time + 1
                                logger.info(f"Waiting {wait_time} seconds for rate limit reset")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        raise GitHubAPIError(
                            message=error_message,
                            status_code=response.status,
                            response_data=response_data
                        )
                    
            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise GitHubAPIError(f"Request failed after {self.config.max_retries} attempts: {str(e)}")
            except Exception as e:
                raise GitHubAPIError(f"Unexpected error: {str(e)}")
        
        raise GitHubAPIError(f"Failed after {self.config.max_retries} attempts")
    
    async def get_issue(self, repo_full_name: str, issue_number: int) -> Dict[str, Any]:
        """
        Get issue details
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            issue_number: Issue number
            
        Returns:
            Dict: Issue data
        """
        url = f"/repos/{repo_full_name}/issues/{issue_number}"
        return await self._make_request('GET', url)
    
    async def add_labels_to_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        labels: List[str]
    ) -> Dict[str, Any]:
        """
        Add labels to an issue
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            issue_number: Issue number
            labels: List of label names to add
            
        Returns:
            Dict: Updated labels
        """
        url = f"/repos/{repo_full_name}/issues/{issue_number}/labels"
        data = {"labels": labels}
        
        logger.info(f"Adding labels {labels} to {repo_full_name}#{issue_number}")
        return await self._make_request('POST', url, data=data)
    
    async def remove_label_from_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        label: str
    ) -> bool:
        """
        Remove a label from an issue
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            issue_number: Issue number
            label: Label name to remove
            
        Returns:
            bool: Success status
        """
        url = f"/repos/{repo_full_name}/issues/{issue_number}/labels/{label}"
        
        try:
            await self._make_request('DELETE', url)
            logger.info(f"Removed label '{label}' from {repo_full_name}#{issue_number}")
            return True
        except GitHubAPIError as e:
            if e.status_code == 404:
                logger.warning(f"Label '{label}' not found on {repo_full_name}#{issue_number}")
                return False
            raise
    
    async def add_comment_to_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str
    ) -> Dict[str, Any]:
        """
        Add a comment to an issue
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            issue_number: Issue number
            body: Comment body (markdown supported)
            
        Returns:
            Dict: Created comment data
        """
        url = f"/repos/{repo_full_name}/issues/{issue_number}/comments"
        data = {"body": body}
        
        logger.info(f"Adding comment to {repo_full_name}#{issue_number}")
        return await self._make_request('POST', url, data=data)
    
    async def update_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        title: str = None,
        body: str = None,
        state: str = None,
        assignees: List[str] = None,
        milestone: int = None,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Update an issue
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            issue_number: Issue number
            title: New title (optional)
            body: New body (optional)
            state: New state - 'open' or 'closed' (optional)
            assignees: List of usernames to assign (optional)
            milestone: Milestone number (optional)
            labels: List of label names (optional)
            
        Returns:
            Dict: Updated issue data
        """
        url = f"/repos/{repo_full_name}/issues/{issue_number}"
        data = {}
        
        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if state is not None:
            data["state"] = state
        if assignees is not None:
            data["assignees"] = assignees
        if milestone is not None:
            data["milestone"] = milestone
        if labels is not None:
            data["labels"] = labels
        
        logger.info(f"Updating issue {repo_full_name}#{issue_number}")
        return await self._make_request('PATCH', url, data=data)
    
    async def get_issue_comments(
        self,
        repo_full_name: str,
        issue_number: int,
        since: str = None,
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get comments for an issue
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            issue_number: Issue number
            since: Only comments updated at or after this time (ISO 8601 format)
            per_page: Number of comments per page (max 100)
            
        Returns:
            List[Dict]: List of comment data
        """
        url = f"/repos/{repo_full_name}/issues/{issue_number}/comments"
        params = {"per_page": per_page}
        
        if since:
            params["since"] = since
        
        comments = []
        page = 1
        
        while True:
            params["page"] = page
            response = await self._make_request('GET', url, params=params)
            
            if not response:
                break
                
            comments.extend(response)
            
            # Check if there are more pages
            if len(response) < per_page:
                break
                
            page += 1
        
        return comments
    
    async def create_issue(
        self,
        repo_full_name: str,
        title: str,
        body: str = None,
        assignees: List[str] = None,
        milestone: int = None,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new issue
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            title: Issue title
            body: Issue body (optional)
            assignees: List of usernames to assign (optional)
            milestone: Milestone number (optional)
            labels: List of label names (optional)
            
        Returns:
            Dict: Created issue data
        """
        url = f"/repos/{repo_full_name}/issues"
        data = {"title": title}
        
        if body is not None:
            data["body"] = body
        if assignees is not None:
            data["assignees"] = assignees
        if milestone is not None:
            data["milestone"] = milestone
        if labels is not None:
            data["labels"] = labels
        
        logger.info(f"Creating issue in {repo_full_name}")
        return await self._make_request('POST', url, data=data)
    
    async def list_repository_labels(self, repo_full_name: str) -> List[Dict[str, Any]]:
        """
        List all labels in a repository
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            
        Returns:
            List[Dict]: List of label data
        """
        url = f"/repos/{repo_full_name}/labels"
        return await self._make_request('GET', url)
    
    async def create_repository_label(
        self,
        repo_full_name: str,
        name: str,
        color: str,
        description: str = None
    ) -> Dict[str, Any]:
        """
        Create a new label in a repository
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            name: Label name
            color: Label color (hex code without #)
            description: Label description (optional)
            
        Returns:
            Dict: Created label data
        """
        url = f"/repos/{repo_full_name}/labels"
        data = {
            "name": name,
            "color": color
        }
        
        if description is not None:
            data["description"] = description
        
        logger.info(f"Creating label '{name}' in {repo_full_name}")
        return await self._make_request('POST', url, data=data)
    
    async def get_user(self, username: str) -> Dict[str, Any]:
        """
        Get user information
        
        Args:
            username: GitHub username
            
        Returns:
            Dict: User data
        """
        url = f"/users/{username}"
        return await self._make_request('GET', url)
    
    async def get_repository(self, repo_full_name: str) -> Dict[str, Any]:
        """
        Get repository information
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            
        Returns:
            Dict: Repository data
        """
        url = f"/repos/{repo_full_name}"
        return await self._make_request('GET', url)


class GitHubOperationsExecutor:
    """
    High-level executor for GitHub operations from dispatcher results
    """
    
    def __init__(self, api_client: GitHubAPIClient = None):
        """Initialize with optional API client"""
        self.api_client = api_client
    
    async def execute_operations(
        self,
        operations: List[Dict[str, Any]],
        repo_full_name: str,
        issue_number: int
    ) -> List[Dict[str, Any]]:
        """
        Execute a list of GitHub operations
        
        Args:
            operations: List of operation dictionaries
            repo_full_name: Repository full name
            issue_number: Issue number
            
        Returns:
            List[Dict]: Results from each operation
        """
        results = []
        
        # Use provided client or create a new one
        if self.api_client:
            client = self.api_client
            results = await self._execute_with_client(client, operations, repo_full_name, issue_number)
        else:
            async with GitHubAPIClient() as client:
                results = await self._execute_with_client(client, operations, repo_full_name, issue_number)
        
        return results
    
    async def _execute_with_client(
        self,
        client: GitHubAPIClient,
        operations: List[Dict[str, Any]],
        repo_full_name: str,
        issue_number: int
    ) -> List[Dict[str, Any]]:
        """Execute operations with provided client"""
        results = []
        
        for operation in operations:
            try:
                result = await self._execute_single_operation(
                    client, operation, repo_full_name, issue_number
                )
                results.append({
                    'operation': operation,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Failed to execute operation {operation}: {str(e)}")
                results.append({
                    'operation': operation,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _execute_single_operation(
        self,
        client: GitHubAPIClient,
        operation: Dict[str, Any],
        repo_full_name: str,
        issue_number: int
    ) -> Dict[str, Any]:
        """Execute a single operation"""
        operation_type = operation.get('type', '')
        
        if operation_type == 'add_labels':
            labels = operation.get('labels', [])
            return await client.add_labels_to_issue(repo_full_name, issue_number, labels)
        
        elif operation_type == 'remove_label':
            label = operation.get('label', '')
            success = await client.remove_label_from_issue(repo_full_name, issue_number, label)
            return {'success': success}
        
        elif operation_type == 'add_comment':
            body = operation.get('body', '')
            return await client.add_comment_to_issue(repo_full_name, issue_number, body)
        
        elif operation_type == 'update_issue':
            update_data = {k: v for k, v in operation.items() if k != 'type'}
            return await client.update_issue(repo_full_name, issue_number, **update_data)
        
        elif operation_type == 'close_issue':
            return await client.update_issue(repo_full_name, issue_number, state='closed')
        
        elif operation_type == 'reopen_issue':
            return await client.update_issue(repo_full_name, issue_number, state='open')
        
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")


# Example usage and testing
if __name__ == "__main__":
    async def test_github_api():
        """Test GitHub API client functionality"""
        try:
            async with GitHubAPIClient() as client:
                # Test getting repository info
                repo_info = await client.get_repository("octocat/Hello-World")
                print(f"Repository: {repo_info['full_name']}")
                
                # Test operations executor
                executor = GitHubOperationsExecutor(client)
                
                # Example operations
                test_operations = [
                    {
                        'type': 'add_comment',
                        'body': '🤖 This is a test comment from the GitHub Issue Agent'
                    }
                ]
                
                # Note: This would require a real issue to test
                # results = await executor.execute_operations(
                #     test_operations, "octocat/Hello-World", 1
                # )
                # print(f"Operation results: {results}")
                
        except Exception as e:
            print(f"Test failed: {str(e)}")
    
    # Run test
    # asyncio.run(test_github_api())
