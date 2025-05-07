import os
import re
import requests
import time

def process_regression_feedback():
    """Process user feedback on regression analysis."""
    # Get environment variables
    github_token = os.environ["GITHUB_TOKEN"]
    comment_id = os.environ["COMMENT_ID"]
    comment_body = os.environ["COMMENT_BODY"]
    issue_number = os.environ["ISSUE_NUMBER"]
    
    print(f"Processing feedback for issue #{issue_number}")
    
    # Extract repository information from the comment body using regex
    match = re.search(r'<!-- comment_id:(\d+):([^:]+):([^:]+) -->', comment_body)
    if not match:
        repo_full_name = os.environ["GITHUB_REPOSITORY"]
        repo_owner, repo_name = repo_full_name.split("/")
        print("Using environment variables for repository information")
    else:
        comment_issue_number = match.group(1)
        repo_owner = match.group(2)
        repo_name = match.group(3)
    
    print(f"Processing feedback for {repo_owner}/{repo_name} issue #{issue_number}")
    
    # API headers
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check which options are selected
    yes_selected = "- [x] Yes, this is a regression" in comment_body
    no_selected = "- [x] No, this is not a regression" in comment_body
    
    # If both options are selected, keep the most recent one
    if yes_selected and no_selected:
        print("Both options are selected. Implementing single-select behavior.")
        
        # Get the comment history to determine which was selected last
        comment_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/comments/{comment_id}"
        history_response = requests.get(comment_url, headers=headers)
        
        if history_response.status_code != 200:
            print(f"Error fetching comment details: {history_response.status_code}")
            return
            
        # The edited_at timestamp will help us determine which option was selected most recently
        # Since we can't directly know which option was selected last, we'll use a heuristic:
        # We'll look at the position of both checkboxes in the comment text.
        # The checkbox that appears last in the comment (higher index) is likely the one that was checked most recently.
        
        yes_position = comment_body.find("- [x] Yes")
        no_position = comment_body.find("- [x] No")
        
        # Determine which option was likely selected last (higher index = later in edit sequence)
        if yes_position > no_position:
            # The "Yes" option appears later in the text, so it was likely selected last
            print("Deselecting 'No' option and keeping 'Yes' option")
            updated_body = comment_body.replace("- [x] No", "- [ ] No")
            keep_yes = True
        else:
            # The "No" option appears later in the text, so it was likely selected last
            print("Deselecting 'Yes' option and keeping 'No' option")
            updated_body = comment_body.replace("- [x] Yes", "- [ ] Yes")
            keep_yes = False
            
        # Update the comment with the corrected selection
        update_data = {"body": updated_body}
        update_response = requests.patch(comment_url, headers=headers, json=update_data)
        
        if update_response.status_code == 200:
            print("Successfully updated comment to implement single-select behavior")
            # Update our local variables to reflect the change
            yes_selected = keep_yes
            no_selected = not keep_yes
        else:
            print(f"Failed to update comment. Status code: {update_response.status_code}")
            return
    
    # If neither option is selected, do nothing
    elif not yes_selected and not no_selected:
        print("No option selected yet.")
        return
    
    # At this point, exactly one option is selected - process the selection
    if yes_selected:
        print(f"User confirmed issue #{issue_number} is a regression")
        
        # Add regression label
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/labels"
        data = {"labels": ["regression"]}
        label_response = requests.post(url, headers=headers, json=data)
        
        if label_response.status_code == 200:
            print(f"Successfully added 'regression' label to issue #{issue_number}")
        else:
            print(f"Failed to add label. Status code: {label_response.status_code}")
    
    # Update the original comment to show it's been processed
    comment_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/comments/{comment_id}"
    
    # Disable further selection by replacing the unselected checkbox with plain text
    updated_body = comment_body
    if yes_selected:
        # Keep Yes checked and remove No checkbox
        if "- [ ] No" in updated_body:
            updated_body = updated_body.replace("- [ ] No", "- No")
        elif "- [x] No" in updated_body:  
            updated_body = updated_body.replace("- [x] No", "- No")
    else:  # no_selected
        # Keep No checked and remove Yes checkbox
        if "- [ ] Yes" in updated_body:
            updated_body = updated_body.replace("- [ ] Yes", "- Yes")
        elif "- [x] Yes" in updated_body:
            updated_body = updated_body.replace("- [x] Yes", "- Yes")
    
    # Add a "processed" message to the note
    updated_body = updated_body.replace(
        "> Note: Only 'Yes' responses will result in adding the regression label.", 
        "> ✅ **Selection recorded**: Your choice has been processed.\n> Note: Only 'Yes' responses will result in adding the regression label."
    )
    
    update_data = {"body": updated_body}
    update_response = requests.patch(comment_url, headers=headers, json=update_data)
    
    if update_response.status_code == 200:
        print("Successfully updated comment to show selection was processed")
    else:
        print(f"Failed to update comment. Status code: {update_response.status_code}")
    
    # Add a confirmation comment
    thank_you_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
    if yes_selected:
        thank_you_data = {
            "body": "Thank you for confirming this is a regression issue. The issue has been labeled accordingly."
        }
    else:
        thank_you_data = {
            "body": "Thank you for your feedback confirming this is not a regression issue."
        }
    requests.post(thank_you_url, headers=headers, json=thank_you_data)

if __name__ == "__main__":
    process_regression_feedback()