import os
import json
import requests
import argparse
from datetime import datetime
from tqdm import tqdm

def get_github_headers(token):
    """Return request headers with GitHub API authentication"""
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

def get_issues(source_owner, source_repo, token, start_issue=None, end_issue=None, state="all"):
    """
    Retrieve issues from the source repository within the specified range
    
    Args:
        source_owner: Owner of the source repository
        source_repo: Name of the source repository
        token: GitHub access token
        start_issue: Starting issue number (optional)
        end_issue: Ending issue number (optional)
        state: Issue state, can be "open", "closed", or "all"
        
    Returns:
        List containing issue data
    """
    headers = get_github_headers(token)
    issues = []
    page = 1
    per_page = 100
    
    print(f"Retrieving issues from {source_owner}/{source_repo}...")
    
    while True:
        url = f"https://api.github.com/repos/{source_owner}/{source_repo}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "direction": "desc"  # Sort by creation time in ascending order
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Failed to retrieve issues: {response.status_code}")
            print(response.text)
            break
            
        page_issues = response.json()
        print(page_issues)
        
        if not page_issues:
            break
            
        for issue in page_issues:
            # Skip pull requests
            if "pull_request" in issue:
                continue
                
            issue_number = issue["number"]
            print(issue_number)
            
            # Check if issue is within the specified range
            if start_issue and issue_number < start_issue:
                continue
            if end_issue and issue_number > end_issue:
                continue
                
            # Retrieve comments for the issue
            issue["comments_data"] = get_issue_comments(
                source_owner, source_repo, issue_number, token
            )
            
            issues.append(issue)
            
        # Early exit if we've gone past the end issue number
        if end_issue and page_issues[-1]["number"] >= end_issue:
            break
            
        page += 1
        break # For testing purposes, remove this line to paginate through all pages
    
    return issues

def get_issue_comments(owner, repo, issue_number, token):
    """Retrieve all comments for the specified issue"""
    headers = get_github_headers(token)
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    
    response = requests.get(comments_url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve comments for issue #{issue_number}: {response.status_code}")
        return []

def create_issues(target_owner, target_repo, token, issues, add_migration_info=True):
    """
    Create issues in the target repository
    
    Args:
        target_owner: Owner of the target repository
        target_repo: Name of the target repository
        token: GitHub access token
        issues: List of issues to create
        add_migration_info: Whether to add migration information to the issue description
        
    Returns:
        Number of successfully created issues
    """
    headers = get_github_headers(token)
    created_count = 0
    
    url = f"https://api.github.com/repos/{target_owner}/{target_repo}/issues"
    
    print(f"Importing {len(issues)} issues to {target_owner}/{target_repo}...")
    
    for issue in tqdm(issues):
        # Prepare issue data
        original_issue_url = issue["html_url"]
        original_author = issue["user"]["login"]
        created_at = issue["created_at"]
        
        # Prepare issue body, optionally with migration info
        body = issue["body"] or ""
        
        if add_migration_info:
            migration_info = f"\n\n---\n*This issue was migrated from {original_issue_url}*\n"
            migration_info += f"*Original author: {original_author}*\n"
            migration_info += f"*Created at: {created_at}*"
            body += migration_info
            
        # Prepare data for creating issue
        issue_data = {
            "title": issue["title"],
            "body": body,
            "labels": [label["name"] for label in issue["labels"] if isinstance(label, dict)]
        }
        
        # Check if original issue is closed
        is_closed = issue["state"] == "closed"
        
        # Create issue
        response = requests.post(url, headers=headers, json=issue_data)
        
        if response.status_code == 201:
            new_issue = response.json()
            new_issue_number = new_issue["number"]
            created_count += 1
            
            # Add comments
            if issue["comments_data"]:
                add_comments(target_owner, target_repo, new_issue_number, token, issue["comments_data"], add_migration_info)
                
            # Close issue if original was closed
            if is_closed:
                close_issue(target_owner, target_repo, new_issue_number, token)
                
        else:
            print(f"Failed to create issue: {response.status_code}")
            print(response.text)
            
    return created_count

def add_comments(owner, repo, issue_number, token, comments, add_migration_info=True):
    """Add comments to an issue"""
    headers = get_github_headers(token)
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    
    for comment in comments:
        original_author = comment["user"]["login"]
        created_at = comment["created_at"]
        body = comment["body"] or ""
        
        if add_migration_info:
            body += f"\n\n---\n*Original comment author: {original_author}*\n"
            body += f"*Comment date: {created_at}*"
            
        comment_data = {"body": body}
        requests.post(comments_url, headers=headers, json=comment_data)

def close_issue(owner, repo, issue_number, token):
    """Close the specified issue"""
    headers = get_github_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    
    data = {"state": "closed"}
    requests.patch(url, headers=headers, json=data)

def main():
    parser = argparse.ArgumentParser(description="Migrate issues between GitHub repositories")
    parser.add_argument("--source", required=True, help="Source repository (format: owner/repo)")
    parser.add_argument("--target", required=True, help="Target repository (format: owner/repo)")
    parser.add_argument("--token", required=True, help="GitHub API access token")
    parser.add_argument("--start", type=int, help="Starting issue number")
    parser.add_argument("--end", type=int, help="Ending issue number")
    parser.add_argument("--state", default="all", choices=["open", "closed", "all"], help="Issue state to migrate")
    parser.add_argument("--output", help="Path to JSON file for exporting issue data (optional)")
    parser.add_argument("--input", help="Import issues from JSON file (optional, replaces source repository retrieval)")
    parser.add_argument("--no-migration-info", action="store_true", help="Don't add migration info to imported issues")
    
    args = parser.parse_args()
    
    source_owner, source_repo = args.source.split("/")
    target_owner, target_repo = args.target.split("/")
    
    # Retrieve issues
    if args.input:
        print(f"Loading issues from file: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            issues = json.load(f)
    else:
        issues = get_issues(
            source_owner, 
            source_repo, 
            args.token,
            args.start,
            args.end,
            args.state
        )
        
        print(f"Retrieved {len(issues)} issues")
        
        # Save issues data if output file is specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(issues, f, indent=2, ensure_ascii=False)
            print(f"Issues data saved to {args.output}")
    
    # Create issues
    if issues:
        created = create_issues(
            target_owner, 
            target_repo, 
            args.token, 
            issues,
            not args.no_migration_info
        )
        print(f"\nSuccessfully imported {created} issues to {target_owner}/{target_repo}")

if __name__ == "__main__":
    main()