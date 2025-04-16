import os
import json
import asyncio
import sys
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

class AzureChatOpenAIError(Exception):
    """Exception raised for errors in Azure OpenAI chat operations."""
    pass

def get_azure_ad_token():
    """Returns a function that gets an Azure AD token."""
    credential = DefaultAzureCredential()
    # This returns the token when called by the LangChain internals
    return lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token

def get_azure_chat_model(
    model_id="gpt-4o",
    streaming=True,
):
    try:
        # Create Azure OpenAI chat model with Azure AD authentication
        chat_model = AzureChatOpenAI(
            deployment_name=model_id,
            api_version="2025-01-01-preview",
            azure_endpoint="https://officegithubcopilotextsubdomain.openai.azure.com/",
            azure_ad_token_provider=get_azure_ad_token(),  # Pass the function that returns a token
            streaming=streaming,  # Enable streaming for real-time output
        )
        return chat_model
    except Exception as e:
        raise AzureChatOpenAIError(e) from e

async def get_llm_response_stream(user_input, history, system_role):
    """
    Get streaming response from LLM with conversation history context.
    
    Args:
        user_input: User's current message
        history: List of previous messages as (role, content) tuples
        system_role: System message defining the assistant's role
    
    Returns:
        Async generator that yields response chunks
    """
    try:
        # Get the LangChain chat model with streaming enabled
        chat_model = get_azure_chat_model(streaming=True)
        
        # Create prompt template with history placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_role),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create streaming chain
        chain = prompt | chat_model | StrOutputParser()
        
        # Convert history to message objects
        formatted_history = []
        for msg in history:
            if msg[0] == "human":
                formatted_history.append(HumanMessage(content=msg[1]))
            elif msg[0] == "ai":
                formatted_history.append(AIMessage(content=msg[1]))
        
        # Return streaming response
        return chain.astream({
            "history": formatted_history,
            "input": user_input
        })
        
    except Exception as e:
        print(f"\n出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

async def chat_session_async(system_role="You are a helpful AI assistant that provides clear and accurate responses."):
    """
    Run an interactive chat session with history context and streaming output.
    
    Args:
        system_role: Description of the AI's role/persona
    """
    print(f"\n💬 开始对话 - AI角色: {system_role}")
    print("输入 'exit' 或 'quit' 结束对话\n")
    
    # Initialize conversation history
    history = []
    
    while True:
        # Get user input
        user_input = input("\n👤 你: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("\n👋 对话已结束")
            break
        
        try:
            # Get AI response stream
            print("\n🤖 AI: ", end="", flush=True)
            full_response = ""
            
            # Process the stream
            async for chunk in await get_llm_response_stream(user_input, history, system_role):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            # Add newline after streaming completes
            print()
            
            # Update history
            history.append(("human", user_input))
            history.append(("ai", full_response))
            
            # Limit history length if needed
            if len(history) > 20:  # Keep last 10 exchanges (20 messages)
                history = history[-20:]
                
        except Exception as e:
            print(f"\n处理响应时发生错误: {str(e)}")
            
def chat_session(system_role="You are a helpful AI assistant that provides clear and accurate responses."):
    """
    Synchronous wrapper for the async chat session.
    
    Args:
        system_role: Description of the AI's role/persona
    """
    asyncio.run(chat_session_async(system_role))

if __name__ == "__main__":
    # Ask for AI role customization
    print("欢迎使用AI助手!")
    custom_role = input("请定义AI助手的角色 (直接回车使用默认角色): ")
    
    if custom_role:
        chat_session(system_role=custom_role)
    else:
        chat_session()