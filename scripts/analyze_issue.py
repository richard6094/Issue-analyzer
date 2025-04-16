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
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        return chat_model
    except Exception as e:
        raise AzureChatOpenAIError(e) from e

def analyze_issue():
    # Get environment variables
    github_token = os.environ["GITHUB_TOKEN"]
    issue_number = os.environ["ISSUE_NUMBER"]
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ["ISSUE_BODY"]
    repo_owner = os.environ["REPO_OWNER"] 
    repo_name = os.environ["REPO_NAME"].split("/")[-1]
    
    # Specify the Azure model to use
    model_id = "gpt-4o"
    
    # Prepare issue content for analysis
    issue_content = f"Issue Title: {issue_title}\n\nIssue Description: {issue_body}"

    try:
        # Get LangChain model
        chat_model = get_azure_chat_model(model_id)
        
        # Create parser to handle JSON output
        parser = JsonOutputParser()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing software issues. Your task is to determine if an issue is a regression. A regression is a bug where functionality that previously worked no longer works due to a recent change. Analyze the issue carefully and provide a JSON response."),
            ("human", "Analyze the following issue and determine if it's a regression issue. Response must be JSON with a 'is_regression' boolean and a 'reason' string.\n\n{issue}")
        ])
        
        # Create chain: prompt -> model -> parser
        chain = prompt | chat_model | parser
        
        # Execute the chain with issue content
        result = chain.invoke({"issue": issue_content})
        
        # Print response
        print("API 响应内容:")
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
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
              

if __name__ == "__main__":
    analyze_issue()