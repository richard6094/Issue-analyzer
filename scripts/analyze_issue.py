import os
import json
import requests
from openai import OpenAI

def analyze_issue():
    # Get environment variables
    api_key = os.environ["OPENAI_API_KEY"]
    github_token = os.environ["GITHUB_TOKEN"]
    issue_number = os.environ["ISSUE_NUMBER"]
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ["ISSUE_BODY"]
    repo_owner = os.environ["REPO_OWNER"] 
    repo_name = os.environ["REPO_NAME"].split("/")[-1]

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Prepare issue content for analysis
    issue_content = f"Issue Title: {issue_title}\n\nIssue Description: {issue_body}"

    # Call LLM to analyze the issue
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing software issues. Your task is to determine if an issue is a regression. A regression is a bug where functionality that previously worked no longer works due to a recent change. Analyze the issue carefully and provide a JSON response."},
            {"role": "user", "content": f"Analyze the following issue and determine if it's a regression issue. Response must be JSON with a 'is_regression' boolean and a 'reason' string.\n\n{issue_content}"}
        ],
        response_format={"type": "json_object"}
    )

    # Extract the analysis result
    result = json.loads(response.choices[0].message.content)
    
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
            print(f"Reason: {result['reason']}")
            # Add a comment explaining the label
            comment_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
            comment_data = {"body": f"This issue was automatically labeled as a regression based on LLM analysis.\n\nReason: {result['reason']}"}
            requests.post(comment_url, headers=headers, json=comment_data)
        else:
            print(f"Failed to add label. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    else:
        print("Issue is not identified as a regression.")
        print(f"Reason: {result['reason']}")

if __name__ == "__main__":
    analyze_issue()