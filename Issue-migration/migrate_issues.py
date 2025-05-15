import os
import json
import requests
import argparse
import sys
import time
from datetime import datetime
from tqdm import tqdm

def get_github_headers(token):
    """Return request headers with GitHub API authentication"""
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

def check_rate_limit(response, exit_on_limit=False):
    """
    Check GitHub API rate limit status and handle accordingly
    
    Args:
        response: GitHub API response
        exit_on_limit: Whether to exit script when hitting rate limit
    
    Returns:
        Boolean indicating if request was successful
    """
    # Extract rate limit information from headers
    limit = int(response.headers.get('X-RateLimit-Limit', 5000))
    remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
    reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
    
    # Calculate time until reset
    reset_time = datetime.fromtimestamp(reset_timestamp)
    now = datetime.now()
    time_to_reset = (reset_time - now).total_seconds()
    
    # Handle rate limiting
    if remaining < 100:  # Warning threshold
        print(f"\nWARNING: GitHub API rate limit running low - {remaining} requests remaining")
        print(f"Rate limit will reset in {time_to_reset/60:.1f} minutes at {reset_time.strftime('%H:%M:%S')}")
    
    # Handle rate limit exceeded
    if response.status_code == 403 and "API rate limit exceeded" in response.text:
        print("\nERROR: GitHub API rate limit exceeded!")
        print(f"Rate limit will reset in {time_to_reset/60:.1f} minutes at {reset_time.strftime('%H:%M:%S')}")
        
        if exit_on_limit:
            print("Exiting due to rate limit...")
            sys.exit(1)
        else:
            wait_time = min(time_to_reset + 10, 3600)  # Wait until reset (max 1 hour)
            print(f"Waiting {wait_time/60:.1f} minutes for rate limit to reset...")
            time.sleep(wait_time)
            return False
    
    # Add small delay between requests to avoid hitting rate limits
    time.sleep(0.25)  # 250ms delay between requests
    return True

def extract_key_information(issues):
    """Extract key information from issues for summary output"""
    summary = []
    for issue in issues:
        key_info = {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"] or "",
            "created_at": issue["created_at"],
            "state": issue["state"],
            "labels": [label["name"] for label in issue["labels"] if isinstance(label, dict)],
            "author": issue["user"]["login"],
            "comments": []
        }
        
        # Extract comment information
        if "comments_data" in issue and issue["comments_data"]:
            for comment in issue["comments_data"]:
                comment_info = {
                    "author": comment["user"]["login"],
                    "body": comment["body"] or "",
                    "created_at": comment["created_at"]
                }
                key_info["comments"].append(comment_info)
        
        summary.append(key_info)
        
    return summary

def get_issues(source_owner, source_repo, token, start_issue=None, end_issue=None, state="all", recent=None, max_retries=3, skip_labels=None):
    """
    Retrieve issues from the source repository within the specified range
    
    Args:
        source_owner: Owner of the source repository
        source_repo: Name of the source repository
        token: GitHub access token
        start_issue: Starting issue number (optional)
        end_issue: Ending issue number (optional)
        state: Issue state, can be "open", "closed", or "all"
        recent: Number of most recent issues to retrieve (optional)
        max_retries: Maximum number of retries for rate limited requests
        skip_labels: List of labels to skip (optional)
        
    Returns:
        List containing issue data
    """
    headers = get_github_headers(token)
    issues = []
    page = 1
    per_page = 100
    lowest_issue_number = float('inf')  # Track the lowest issue number seen
    total_issues_processed = 0
    progress_bar = None
    
    tqdm.write(f"Retrieving issues from {source_owner}/{source_repo}...")
    
    # Get repository information to show total issues count
    repo_url = f"https://api.github.com/repos/{source_owner}/{source_repo}"
    repo_response = requests.get(repo_url, headers=headers)
    
    if repo_response.status_code == 200:
        repo_info = repo_response.json()
        total_issues_count = repo_info.get('open_issues_count', 0)
        if state == "all":
            # For "all" state, we need to estimate total issues as GitHub only reports open issues
            total_issues_count = total_issues_count * 2  # Rough estimation
        
        if end_issue and start_issue:
            estimated_total = min(end_issue - start_issue + 1, total_issues_count)
        elif end_issue:
            estimated_total = min(end_issue, total_issues_count)
        elif start_issue:
            estimated_total = min(total_issues_count, 1000)  # Reasonable default
        elif recent:
            estimated_total = min(recent, total_issues_count)
        else:
            estimated_total = min(total_issues_count, 1000)  # Reasonable default
            
        # Add leave=True to ensure the progress bar stays in place
        progress_bar = tqdm(total=estimated_total, desc="Fetching issues", unit="issue", leave=True)
    
    while True:
        url = f"https://api.github.com/repos/{source_owner}/{source_repo}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "direction": "desc"  # Sort by creation time descending (newest first)
        }
        
        # Display current page information without disrupting progress bar
        if not progress_bar:
            tqdm.write(f"Fetching page {page} (up to {per_page} issues per page)")
        
        # Try with retries for rate limits
        for retry in range(max_retries):
            response = requests.get(url, headers=headers, params=params)
            
            # Check rate limit and wait if necessary
            if not check_rate_limit(response):
                if retry < max_retries - 1:
                    continue  # Retry after waiting
                else:
                    tqdm.write(f"Failed after {max_retries} retries due to rate limiting")
                    if progress_bar:
                        progress_bar.close()
                    return issues
            
            # Break out of retry loop if successful
            break
        
        if response.status_code != 200:
            tqdm.write(f"Failed to retrieve issues: {response.status_code}")
            tqdm.write(response.text)
            if progress_bar:
                progress_bar.close()
            break
            
        page_issues = response.json()
        
        if not page_issues:
            if progress_bar:
                progress_bar.close()
            break
        
        # Track how many issues from this page were actually added
        page_added = 0
            
        for issue in page_issues:
            # Skip pull requests
            if "pull_request" in issue:
                continue
                
            issue_number = issue["number"]
            lowest_issue_number = min(lowest_issue_number, issue_number)
            
            # Check if issue is within the specified range
            if end_issue and issue_number > end_issue:
                continue
            if start_issue and issue_number < start_issue:
                continue

            # Skip issues with specific labels if requested
            if skip_labels and issue.get("labels"):
                # Extract label names from the issue
                issue_label_names = [label["name"] for label in issue["labels"] if isinstance(label, dict)]
                
                # Skip if any label matches the skip_labels list
                if any(label in skip_labels for label in issue_label_names):
                    tqdm.write(f"Skipping issue #{issue_number} because it has label(s) matching skip criteria")
                    continue
            
            # Progress information for current issue (only if no progress bar)
            if not progress_bar:
                tqdm.write(f"Processing issue #{issue_number}: {issue['title'][:50]}...")
                
            # Retrieve comments for the issue with rate limit awareness
            issue["comments_data"] = get_issue_comments(
                source_owner, source_repo, issue_number, token, max_retries
            )
            
            issues.append(issue)
            page_added += 1
            total_issues_processed += 1
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"issue": f"#{issue_number}", "comments": len(issue["comments_data"])})
            
            # If we've reached the requested number of recent issues, return
            if recent and len(issues) >= recent:
                if progress_bar:
                    progress_bar.close()
                return issues
        
        # Print page summary if not using progress bar
        if not progress_bar:
            tqdm.write(f"Page {page}: Added {page_added} issues (Total: {total_issues_processed})")
                
        # Pagination control logic
        if start_issue and lowest_issue_number > start_issue and (not recent or len(issues) < recent):
            # If we haven't reached the start issue yet and still need more issues, continue pagination
            page += 1
        elif recent and len(issues) < recent:
            # If collecting recent issues and haven't collected enough, continue pagination
            page += 1
        else:
            # Otherwise, exit the loop
            if progress_bar:
                progress_bar.close()
            break
    
    return issues

def get_issue_comments(owner, repo, issue_number, token, max_retries=3):
    """Retrieve all comments for the specified issue"""
    headers = get_github_headers(token)
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    
    # Try with retries for rate limits
    for retry in range(max_retries):
        response = requests.get(comments_url, headers=headers)
        
        # Check rate limit and wait if necessary
        if not check_rate_limit(response):
            if retry < max_retries - 1:
                continue  # Retry after waiting
            else:
                tqdm.write(f"Failed to get comments for issue #{issue_number} after {max_retries} retries")
                return []
        
        if response.status_code == 200:
            comments = response.json()
            # Show number of comments retrieved but use tqdm.write to avoid disrupting progress bar
            if len(comments) > 0:
                # Use tqdm.write instead of print to avoid disrupting the progress bar
                if len(comments) > 5:  # Only log if significant number of comments
                    tqdm.write(f"  Retrieved {len(comments)} comments for issue #{issue_number}")
            return comments
        else:
            tqdm.write(f"Failed to retrieve comments for issue #{issue_number}: {response.status_code}")
            tqdm.write(response.text)
            return []

def create_issues(target_owner, target_repo, token, issues, add_migration_info=True, max_retries=3):
    """
    Create issues in the target repository
    
    Args:
        target_owner: Owner of the target repository
        target_repo: Name of the target repository
        token: GitHub access token
        issues: List of issues to create
        add_migration_info: Whether to add migration information to the issue description
        max_retries: Maximum number of retries for rate limited requests
        
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
        
        # Create issue with retry capability
        for retry in range(max_retries):
            response = requests.post(url, headers=headers, json=issue_data)
            
            # Check rate limit and wait if necessary
            if not check_rate_limit(response):
                if retry < max_retries - 1:
                    continue  # Retry after waiting
                else:
                    print(f"Failed to create issue after {max_retries} retries")
                    break
            
            if response.status_code == 201:
                new_issue = response.json()
                new_issue_number = new_issue["number"]
                created_count += 1
                
                # Add comments
                if issue["comments_data"]:
                    add_comments(target_owner, target_repo, new_issue_number, token, issue["comments_data"], add_migration_info, max_retries)
                    
                # Close issue if original was closed
                if is_closed:
                    close_issue(target_owner, target_repo, new_issue_number, token, max_retries)
                    
                break  # Exit retry loop on success
            else:
                print(f"Failed to create issue: {response.status_code}")
                print(response.text)
                
                # Only retry on rate limit errors
                if response.status_code != 403 or "API rate limit exceeded" not in response.text:
                    break  # Don't retry non-rate-limit errors
            
    return created_count

def add_comments(owner, repo, issue_number, token, comments, add_migration_info=True, max_retries=3):
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
        
        # Try with retries for rate limits
        for retry in range(max_retries):
            response = requests.post(comments_url, headers=headers, json=comment_data)
            
            # Check rate limit and wait if necessary
            if not check_rate_limit(response):
                if retry < max_retries - 1:
                    continue  # Retry after waiting
                else:
                    print(f"Failed to add comment after {max_retries} retries")
                    break  # Give up after max retries
            
            # Break on success or non-rate-limit error
            if response.status_code == 201:
                break
            elif response.status_code != 403 or "API rate limit exceeded" not in response.text:
                print(f"Failed to add comment: {response.status_code}")
                print(response.text)
                break

def close_issue(owner, repo, issue_number, token, max_retries=3):
    """Close the specified issue"""
    headers = get_github_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    
    data = {"state": "closed"}
    
    # Try with retries for rate limits
    for retry in range(max_retries):
        response = requests.patch(url, headers=headers, json=data)
        
        # Check rate limit and wait if necessary
        if not check_rate_limit(response):
            if retry < max_retries - 1:
                continue  # Retry after waiting
            else:
                print(f"Failed to close issue #{issue_number} after {max_retries} retries")
                return
        
        # Break on success or non-rate-limit error
        if response.status_code == 200:
            break
        elif response.status_code != 403 or "API rate limit exceeded" not in response.text:
            print(f"Failed to close issue #{issue_number}: {response.status_code}")
            print(response.text)
            break

def process_in_batches(source_owner, source_repo, token, start_issue, batch_size, end_issue=None, state="all", skip_labels=None):
    """Process issues in batches to manage rate limits"""
    all_issues = []
    current_start = start_issue
    
    # Calculate total range and create overall progress bar
    total_range = "unknown"
    overall_progress = None
    if end_issue:
        total_range = end_issue - start_issue + 1
        # Add leave=True to ensure the progress bar stays in place
        overall_progress = tqdm(total=total_range, desc="Overall progress", unit="issues", leave=True)
    
    batch_count = 0
    
    while True:
        batch_count += 1
        current_end = current_start + batch_size - 1
        if end_issue and current_end > end_issue:
            current_end = end_issue
        
        batch_range = current_end - current_start + 1
        
        # Use tqdm.write to avoid disrupting progress bar
        tqdm.write(f"\nBatch {batch_count}: Processing issues #{current_start} to #{current_end} ({batch_range} issues)")
        
        batch = get_issues(
            source_owner, source_repo, token,
            current_start, current_end, state,
            skip_labels=skip_labels
        )
        
        if not batch:
            tqdm.write(f"No issues found in batch {batch_count} (range: #{current_start}-#{current_end})")
            
            # If we've reached the specified end or there are no more issues, break
            if end_issue and current_end >= end_issue:
                break
            
            # If batch is empty but we haven't reached the end, try the next batch
            current_start = current_end + 1
            
            # Update overall progress
            if overall_progress:
                overall_progress.update(batch_range)
            
            continue
        
        batch_size_actual = len(batch)
        all_issues.extend(batch)
        
        # Calculate and display percentage complete
        total_retrieved = len(all_issues)
        if end_issue:
            percent_complete = min(100.0, (current_end - start_issue + 1) / total_range * 100)
            if overall_progress:
                overall_progress.update(batch_size_actual)
            tqdm.write(f"Batch {batch_count} complete: Retrieved {batch_size_actual} issues " 
                      f"(Total: {total_retrieved}, Progress: {percent_complete:.1f}%)")
        else:
            tqdm.write(f"Batch {batch_count} complete: Retrieved {batch_size_actual} issues (Total: {total_retrieved})")
        
        # If we've reached the specified end, break
        if end_issue and current_end >= end_issue:
            break
            
        current_start = current_end + 1
    
    # Close overall progress bar if it exists
    if overall_progress:
        overall_progress.close()
    
    return all_issues

def save_checkpoint(issues, checkpoint_file):
    """Save checkpoint of issues processed so far"""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved to {checkpoint_file}")

def main():
    parser = argparse.ArgumentParser(description="Migrate issues between GitHub repositories")
    parser.add_argument("--source", required=True, help="Source repository (format: owner/repo)")
    parser.add_argument("--target", help="Target repository (format: owner/repo)")
    parser.add_argument("--token", required=True, help="GitHub API access token")
    parser.add_argument("--start", type=int, help="Starting issue number")
    parser.add_argument("--end", type=int, help="Ending issue number")
    parser.add_argument("--recent", type=int, help="Number of most recent issues to retrieve")
    parser.add_argument("--state", default="all", choices=["open", "closed", "all"], help="Issue state to migrate")
    parser.add_argument("--output", help="Path to JSON file for exporting issue data (optional)")
    parser.add_argument("--summary-output", help="Path to JSON file for exporting key issue information (optional)")
    parser.add_argument("--input", help="Import issues from JSON file (optional, replaces source repository retrieval)")
    parser.add_argument("--no-migration-info", action="store_true", help="Don't add migration info to imported issues")
    parser.add_argument("--save-only", action="store_true", help="Only save issues to file, don't migrate")
    parser.add_argument("--batch-size", type=int, default=50, 
                      help="Process issues in batches of this size to manage rate limits")
    parser.add_argument("--checkpoint", action="store_true", 
                      help="Save checkpoint files after each batch")
    parser.add_argument("--resume", help="Resume from a checkpoint file")
    parser.add_argument("--skip-labels", 
                      help="Comma-separated list of labels to skip (e.g., 'bug,enhancement')")
    
    args = parser.parse_args()
    
    source_owner, source_repo = args.source.split("/")
    
    # Parse skip_labels if provided
    skip_labels = None
    if args.skip_labels:
        skip_labels = [label.strip() for label in args.skip_labels.split(',')]
        print(f"Will skip issues with the following labels: {', '.join(skip_labels)}")
    
    # Validate arguments
    if not args.save_only and not args.target:
        parser.error("--target is required unless --save-only is specified")
    
    # Set target repository information (if provided)
    target_owner, target_repo = None, None
    if args.target:
        target_owner, target_repo = args.target.split("/")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint file: {args.resume}")
        with open(args.resume, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        print(f"Loaded {len(issues)} issues from checkpoint")
    
    # Retrieve issues
    elif args.input:
        print(f"Loading issues from file: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            issues = json.load(f)
    else:
        # Process in batches if batch_size is specified and not using 'recent'
        if args.batch_size and not args.recent and args.start:
            issues = process_in_batches(
                source_owner, source_repo, args.token,
                args.start, args.batch_size, args.end, args.state,
                skip_labels=skip_labels
            )
            
            # Save checkpoint after batch processing if requested
            if args.checkpoint and issues:
                checkpoint_file = "issues_checkpoint.json"
                if args.output:
                    base_name = os.path.splitext(args.output)[0]
                    checkpoint_file = f"{base_name}_checkpoint.json"
                save_checkpoint(issues, checkpoint_file)
        else:
            # Regular processing
            issues = get_issues(
                source_owner, 
                source_repo, 
                args.token,
                args.start,
                args.end,
                args.state,
                args.recent,
                skip_labels=skip_labels
            )
        
        print(f"Retrieved {len(issues)} issues")
        
        # Save full issue data if output file is specified
        if args.output and issues:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(issues, f, indent=2, ensure_ascii=False)
            print(f"Issues data saved to {args.output}")
        
        # Save summary information if summary output file is specified
        if args.summary_output and issues:
            summary = extract_key_information(issues)
            with open(args.summary_output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Issues summary saved to {args.summary_output}")
    
    # Create issues (if not in save-only mode and target repo is provided)
    if issues and not args.save_only and args.target:
        created = create_issues(
            target_owner, 
            target_repo, 
            args.token, 
            issues,
            not args.no_migration_info
        )
        print(f"\nSuccessfully imported {created} issues to {target_owner}/{target_repo}")
    elif args.save_only:
        print("Issues saved to file. Migration skipped.")

if __name__ == "__main__":
    main()