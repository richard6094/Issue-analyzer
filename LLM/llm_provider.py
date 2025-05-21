"""
LLM providers using LangChain architecture.
"""
from typing import Dict, Any, List, Optional, Union
import os
import json

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable

# Define function to get model based on provider
def get_llm(
    provider: str = "openai",
    model: str = None,
    temperature: float = 0,
    **kwargs
) -> BaseChatModel:
    """
    Get a LangChain chat model based on the specified provider.
    
    Args:
        provider: The provider to use ("openai" or "azure")
        model: Model name/deployment name to use
        temperature: Temperature setting for generation (0-1)
        **kwargs: Additional arguments for the model
        
    Returns:
        A LangChain chat model
    """
    if provider.lower() == "azure":
        # Azure OpenAI
        return get_azure_llm(
            deployment=model or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            temperature=temperature,
            **kwargs
        )
    else:
        # Standard OpenAI
        return get_openai_llm(
            model=model or "gpt-4o",
            temperature=temperature,
            **kwargs
        )

def get_openai_llm(
    model: str = "gpt-4o",
    api_key: str = None,
    temperature: float = 0,
    **kwargs
) -> ChatOpenAI:
    """
    Get a LangChain chat model for OpenAI.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4o")
        api_key: OpenAI API key
        temperature: Temperature setting for generation (0-1)
        **kwargs: Additional arguments for the ChatOpenAI model
        
    Returns:
        A ChatOpenAI instance
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Provide it as an argument or set the OPENAI_API_KEY environment variable.")
    
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        **kwargs
    )

def get_azure_llm(
    deployment: str = None,
    endpoint: str = None,
    api_version: str = "2025-01-01-preview",
    api_key: str = None,
    temperature: float = 0,
    **kwargs
) -> AzureChatOpenAI:
    """
    Get a LangChain chat model for Azure OpenAI.
    
    Args:
        deployment: Azure OpenAI deployment name
        endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        api_key: Azure OpenAI API key
        temperature: Temperature setting for generation (0-1)
        **kwargs: Additional arguments for the AzureChatOpenAI model
        
    Returns:
        An AzureChatOpenAI instance
    """
    endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint is required. Provide it as an argument or set the AZURE_OPENAI_ENDPOINT environment variable.")
    if not deployment:
        raise ValueError("Azure OpenAI deployment name is required. Provide it as an argument or set the AZURE_OPENAI_DEPLOYMENT environment variable.")
    
    auth_args = {}
    if api_key:
        auth_args["api_key"] = api_key
    else:
        # Use Azure AD authentication
        try:
            from azure.identity import DefaultAzureCredential
            
            # Create a function that returns a token when called
            # This is what LangChain expects for azure_ad_token_provider
            def get_azure_ad_token():
                credential = DefaultAzureCredential()
                return credential.get_token("https://cognitiveservices.azure.com/.default").token
            
            auth_args["azure_ad_token_provider"] = get_azure_ad_token
        except ImportError:
            raise ImportError("To use Azure AD authentication, install the azure-identity package.")
    
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_version=api_version,
        temperature=temperature,
        **auth_args,
        **kwargs
    )

# Utility function for generating text
def generate_text(
    prompt: str,
    llm: BaseChatModel = None,
    provider: str = "openai",
    **kwargs
) -> str:
    """
    Generate text response for a single prompt using LangChain.
    
    Args:
        prompt: Text prompt to generate completion for
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Generated text as a string
    """
    model = llm or get_llm(provider=provider, **kwargs)
    chain = model | StrOutputParser()
    return chain.invoke(prompt)

# Utility function for generating chat responses
def generate_chat_response(
    messages: List[Dict[str, str]],
    llm: BaseChatModel = None,
    provider: str = "openai",
    return_langchain_messages: bool = False,
    **kwargs
) -> Union[Dict[str, Any], BaseMessage]:
    """
    Generate a response for a chat conversation using LangChain.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        return_langchain_messages: If True, return LangChain message objects
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Dictionary with response and metadata or LangChain message
    """
    model = llm or get_llm(provider=provider, **kwargs)
    
    # Convert dict messages to LangChain messages
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "")
        
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        else:
            langchain_messages.append(ChatMessage(role=role, content=content))
    
    # Generate response
    result = model.invoke(langchain_messages)
    
    if return_langchain_messages:
        return result
    
    # Convert to dictionary format (similar to OpenAI API response)
    response = {
        "content": result.content,
        "role": "assistant",
        "model": getattr(model, "model", "unknown"),
    }
    
    # Add any model-specific metadata that might be available
    model_generation_info = None
    if hasattr(result, "generation_info"):
        model_generation_info = result.generation_info
        response.update(model_generation_info)
    
    # Add token usage if available (might not be available for all models)
    if hasattr(model, "last_token_usage"):
        response["usage"] = model.last_token_usage
    
    return response

# Utility function for generating structured outputs
def generate_structured_output(
    prompt: str,
    output_schema: Dict[str, Any],
    llm: BaseChatModel = None,
    provider: str = "openai",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate structured output based on a schema using LangChain.
    
    Args:
        prompt: Text prompt to generate structured output for
        output_schema: JSON schema defining the expected output structure
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Dictionary containing the structured output
    """
    model = llm or get_llm(provider=provider, **kwargs)
    
    # We need to add response_format param for JSON output if using OpenAI
    model_kwargs = kwargs.copy()
    if "response_format" not in model_kwargs:
        model_kwargs["response_format"] = {"type": "json_object"}
    
    # Convert the schema to string with properly escaped braces
    schema_str = json.dumps(output_schema, indent=2)
    
    # Create system prompt with schema information
    system_message = f"""You will be given a prompt. Respond with a JSON object that matches this schema:
{schema_str}

Ensure your response is valid JSON that conforms to the schema. 
Only return the JSON object, no other text."""
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
    
    try:
        # Method 1: Simple direct approach first
        result = model.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ])
        response_text = result.content
        
        # Parse JSON content
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it contains extra text
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(response_text[json_start:json_end])
            except:
                pass
            
            # If that fails, try Method 2 with the chain approach
            pass
    except Exception as e:
        # If direct approach fails, log and continue to method 2
        print(f"Direct approach failed with error: {str(e)}")
    
    # Method 2: Use a chain with better error handling
    try:
        # Create chain: model -> output parser
        chain = model | StrOutputParser()
        
        # Execute with direct messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        response_text = chain.invoke(messages)
        
        # Parse JSON content
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it contains extra text
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(response_text[json_start:json_end])
            except:
                pass
            
            # Return error if we couldn't parse the JSON
            return {
                "error": "Failed to parse JSON response",
                "content": response_text
            }
    except Exception as e:
        return {
            "error": f"Failed to generate structured output: {str(e)}",
            "content": prompt
        }