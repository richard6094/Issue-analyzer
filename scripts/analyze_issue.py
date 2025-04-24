import os
import json
import requests
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class AzureChatOpenAIError(Exception):
    """Exception raised for errors in Azure OpenAI chat operations."""
    pass

def get_azure_ad_token():
    """Returns a function that gets an Azure AD token."""
    credential = DefaultAzureCredential()
    # This returns the token when called by the LangChain internals
    return lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token

def get_azure_chat_model(model_id="gpt-4o"):
    """Get a configured Azure OpenAI chat model through LangChain."""
    try:
        # Create Azure OpenAI chat model with Azure AD authentication
        chat_model = AzureChatOpenAI(
            deployment_name=model_id,
            api_version="2025-01-01-preview",
            azure_endpoint="https://officegithubcopilotextsubdomain.openai.azure.com/",
            azure_ad_token_provider=get_azure_ad_token(),
            temperature=0,  # Use deterministic output for analysis
            model_kwargs={"response_format": {"type": "json_object"}}  # Ensure JSON output
        )
        return chat_model
    except Exception as e:
        raise AzureChatOpenAIError(e) from e

def ensure_label_exists(repo_owner, repo_name, github_token, label_name, color, description=None):
    """
    Ensure a label exists with the specified color and description.
    If the label doesn't exist, create it. If it exists but with different settings, update it.
    """
    # Check if label exists
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/labels/{label_name}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    
    # If label exists
    if response.status_code == 200:
        current_label = response.json()
        # Check if we need to update
        if current_label["color"] != color or (description and current_label.get("description") != description):
            print(f"Updating label '{label_name}' with new color: {color}")
            update_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/labels/{label_name}"
            data = {"color": color}
            if description:
                data["description"] = description
            
            update_response = requests.patch(update_url, headers=headers, json=data)
            return update_response.status_code == 200
        return True
    
    # If label doesn't exist, create it
    elif response.status_code == 404:
        print(f"Creating label '{label_name}' with color: {color}")
        create_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/labels"
        data = {
            "name": label_name,
            "color": color
        }
        if description:
            data["description"] = description
            
        create_response = requests.post(create_url, headers=headers, json=data)
        return create_response.status_code == 201
    
    # Some other error
    else:
        print(f"Error checking label: {response.status_code}")
        print(response.text)
        return False

def get_issue_comments(repo_owner, repo_name, issue_number, github_token):
    """Get all comments for a specific issue."""
    comments_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(comments_url, headers=headers)
    
    if response.status_code == 200:
        comments = response.json()
        print(f"Retrieved {len(comments)} comments for issue #{issue_number}")
        return comments
    else:
        print(f"Error fetching comments: {response.status_code}")
        print(response.text)
        return []

def analyze_single_issue(issue_data, repo_owner, repo_name, github_token, chat_model, parser, prompt, include_comments=False):
    """
    Analyze a single issue and apply labels if needed.
    
    Args:
        issue_data: The issue data from GitHub API
        repo_owner: Repository owner
        repo_name: Repository name
        github_token: GitHub API token
        chat_model: LangChain model for analysis
        parser: Output parser
        prompt: Analysis prompt template
        include_comments: Whether to include comments in the analysis
        
    Returns:
        bool: True if the issue is a regression, False otherwise
    """
    issue_number = issue_data["number"]
    issue_title = issue_data["title"]
    issue_body = issue_data["body"] or ""
    
    # Check if issue already has regression label
    has_regression_label = False
    for label in issue_data["labels"]:
        if isinstance(label, dict) and label.get("name") == "regression":
            has_regression_label = True
            break
        elif isinstance(label, str) and label == "regression":
            has_regression_label = True
            break
    
    if has_regression_label:
        print(f"Skipping issue #{issue_number} as it already has regression label.")
        return False
        
    print(f"\nProcessing issue #{issue_number}: {issue_title}")
    print(f"Body length: {len(issue_body)} characters")
    
    # Prepare issue content for analysis
    issue_content = f"Issue Title: {issue_title}\n\nIssue Description: {issue_body}"
    
    # Include comments if requested
    if include_comments:
        print(f"Including comments in analysis for issue #{issue_number}")
        comments = get_issue_comments(repo_owner, repo_name, issue_number, github_token)
        if comments:
            comments_text = "\n\n".join([f"Comment by {comment['user']['login']}: {comment['body']}" for comment in comments])
            issue_content += f"\n\nComments:\n{comments_text}"
            print(f"Added {len(comments)} comments to analysis")
    
    try:
        # Execute the chain with issue content
        result = prompt.pipe(chat_model).pipe(parser).invoke({"issue": issue_content})
        
        # Print response
        print(f"Analysis result for issue #{issue_number}:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # Add labels based on analysis
        if result["is_regression"]:
            # GitHub API to add labels to the issue
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/labels"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            data = {"labels": ["regression"]}
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                print(f"Successfully added 'regression' label to issue #{issue_number}")
                # Add a comment explaining the label
                comment_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
                source = "issue content and comments" if include_comments else "issue content"
                comment_data = {"body": f"This issue was automatically labeled as a regression based on LLM analysis of the {source}.\n\nReason: {result['reason']}"}
                requests.post(comment_url, headers=headers, json=comment_data)
            else:
                print(f"Failed to add label. Status code: {response.status_code}")
            
            return True
        else:
            print(f"Issue #{issue_number} is not identified as a regression.")
            return False
            
    except Exception as e:
        print(f"Error processing issue #{issue_number}: {str(e)}")
        return False
        
def analyze_issues():
    """Process all issues in the repository."""
    try:
        # Get environment variables
        github_token = os.environ["GITHUB_TOKEN"]
        repo_full_name = os.environ["GITHUB_REPOSITORY"]
        repo_owner, repo_name = repo_full_name.split("/")
        
        ensure_label_exists(
            repo_owner, 
            repo_name, 
            github_token, 
            "regression", 
            "d73a4a",  #Red color for regression
            "Functionality that previously worked no longer works"
        )

        # Determine event type from environment variable or event file
        # event_name = os.environ.get("GITHUB_EVENT_NAME", "")
        # is_comment_event = event_name == "issue_comment"
        
        # Check for specific issue number
        specific_issue = os.environ.get("ISSUE_NUMBER")
        
        # Initialize model and parsing components
        model_id = "gpt-4o"
        chat_model = get_azure_chat_model(model_id)
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing software issues. Your task is to determine if an issue is a regression. A regression is a bug where functionality that previously worked no longer works due to a recent change. Analyze the issue carefully, including any comments that might provide additional context. Provide a JSON response."),
            ("human", "Analyze the following issue and determine if it's a regression issue. Response must be JSON with a 'is_regression' boolean and a 'reason' string.\n\n{issue}")
        ])
        
        # Process a specific issue if specified
        if specific_issue:
            issue_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{specific_issue}"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            response = requests.get(issue_url, headers=headers)
            
            if response.status_code == 200:
                issue_data = response.json()
                analyze_single_issue(
                    issue_data, 
                    repo_owner, 
                    repo_name, 
                    github_token, 
                    chat_model, 
                    parser, 
                    prompt,
                    include_comments=True
                )
            else:
                print(f"Error fetching specific issue: {response.status_code}")
                print(response.text)
            
            return
        
        # Process all open issues
        print(f"Fetching all open issues from {repo_owner}/{repo_name}")
        page = 1
        per_page = 30
        processed = 0
        
        while True:
            # Fetch issues with pagination
            issues_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues?state=open&per_page={per_page}&page={page}"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(issues_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Error fetching issues: {response.status_code}")
                print(response.text)
                break
                
            issues = response.json()
            
            # Break if no more issues
            if not issues:
                break
                
            print(f"Processing page {page} with {len(issues)} issues")
            
            # Process each issue
            for issue in issues:
                # Skip pull requests (they appear in issues API)
                if "pull_request" in issue:
                    continue
                    
                analyze_single_issue(issue, repo_owner, repo_name, github_token, chat_model, parser, prompt)
                processed += 1
                
            # Move to next page
            page += 1
                
        print(f"\nCompleted processing {processed} issues")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    analyze_issues()