import os
import requests

def process_regression_feedback():
    """Process user feedback on regression analysis."""
    # Get environment variables
    github_token = os.environ["GITHUB_TOKEN"]
    issue_number = os.environ["ISSUE_NUMBER"]
    
    # Repository information
    repo_full_name = os.environ["GITHUB_REPOSITORY"]
    repo_owner, repo_name = repo_full_name.split("/")
    
    print(f"Processing feedback for issue #{issue_number}")
    
    # API headers
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Get issue comments to find our feedback request
    comments_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
    response = requests.get(comments_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching comments: {response.status_code}")
        return
    
    comments = response.json()
    
    # Find our regression analysis comment
    feedback_comment = None
    for comment in comments:
        if "## Regression Analysis Needs Your Input" in comment.get("body", ""):
            feedback_comment = comment
            break
    
    if not feedback_comment:
        print("No regression analysis comment found.")
        return
    
    comment_body = feedback_comment.get("body", "")
    
    # Check if a choice was made
    yes_selected = "- [x] Yes, this is a regression" in comment_body
    no_selected = "- [x] No, this is not a regression" in comment_body
    
    if not yes_selected and not no_selected:
        # User clicked the button without making a selection
        followup_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
        followup_data = {
            "body": "⚠️ **Please select either 'Yes' or 'No' option in the previous comment before submitting your answer.**"
        }
        requests.post(followup_url, headers=headers, json=followup_data)
        return
    
    # Process the selection
    if yes_selected:
        print(f"User confirmed issue #{issue_number} is a regression")
        
        # Add regression label only
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/labels"
        data = {"labels": ["regression"]}
        label_response = requests.post(url, headers=headers, json=data)
        
        if label_response.status_code == 200:
            print(f"Successfully added 'regression' label to issue #{issue_number}")
        else:
            print(f"Failed to add regression label. Status code: {label_response.status_code}")
    else:
        print(f"User confirmed issue #{issue_number} is NOT a regression - no label will be added")
    
    # Update the original comment to show it's been processed
    comment_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/comments/{feedback_comment['id']}"
    
    # Replace the submission button with a processed message
    updated_comment = comment_body.replace(
        "[🔄 Submit My Answer]", 
        "✅ **Answer Submitted** *(response recorded)*"
    )
    
    update_data = {"body": updated_comment}
    update_response = requests.patch(comment_url, headers=headers, json=update_data)
    
    if update_response.status_code == 200:
        print("Successfully updated comment to show response was recorded")
    else:
        print(f"Failed to update comment. Status code: {update_response.status_code}")
    
    # Add a simple thank you comment
    thank_you_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
    if yes_selected:
        thank_you_data = {
            "body": "Thank you for confirming this is a regression issue. The issue has been labeled accordingly."
        }
    else:
        thank_you_data = {
            "body": "Thank you for your feedback. No regression label has been applied to this issue."
        }
    requests.post(thank_you_url, headers=headers, json=thank_you_data)

if __name__ == "__main__":
    process_regression_feedback()