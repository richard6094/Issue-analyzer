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

def analyze_single_issue(issue_data, repo_owner, repo_name, github_token, chat_model, parser, prompt):
    """Analyze a single issue and apply labels if needed."""
    issue_number = issue_data["number"]
    issue_title = issue_data["title"]
    issue_body = issue_data["body"] or ""
    
    # Skip issues that already have labels (optional)
    if issue_data["labels"]:
        print(f"Skipping issue #{issue_number} as it already has labels.")
        return
        
    print(f"\nProcessing issue #{issue_number}: {issue_title}")
    print(f"Body length: {len(issue_body)} characters")
    
    # Prepare issue content for analysis
    issue_content = f"Issue Title: {issue_title}\n\nIssue Description: {issue_body}"
    
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
                comment_data = {"body": f"This issue was automatically labeled as a regression based on LLM analysis.\n\nReason: {result['reason']}"}
                requests.post(comment_url, headers=headers, json=comment_data)
            else:
                print(f"Failed to add label. Status code: {response.status_code}")
        else:
            print(f"Issue #{issue_number} is not identified as a regression.")
            
    except Exception as e:
        print(f"Error processing issue #{issue_number}: {str(e)}")

def analyze_issues():
    """Process all issues in the repository."""
    try:
        # Get environment variables
        github_token = os.environ["GITHUB_TOKEN"]
        repo_full_name = os.environ["GITHUB_REPOSITORY"]
        repo_owner, repo_name = repo_full_name.split("/")
        
        # Check for specific issue number
        specific_issue = os.environ.get("ISSUE_NUMBER")
        
        # Initialize model and parsing components
        model_id = "gpt-4o"
        chat_model = get_azure_chat_model(model_id)
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing software issues. Your task is to determine if an issue is a regression. A regression is a bug where functionality that previously worked no longer works due to a recent change. Analyze the issue carefully and provide a JSON response."),
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
                analyze_single_issue(issue_data, repo_owner, repo_name, github_token, chat_model, parser, prompt)
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