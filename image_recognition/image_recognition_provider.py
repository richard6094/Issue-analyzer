"""
Image recognition provider using LangChain architecture with GPT-4o Vision capabilities.
"""
from typing import Dict, Any, List, Optional, Union
import os
import json
import re
from urllib.parse import urlparse

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
from langchain_core.prompts import ChatPromptTemplate

# Regular expression for matching image URLs in markdown format
IMAGE_URL_REGEX = r"!\[.*?\]\((.*?)\)"

def extract_image_urls(content: str) -> List[str]:
    """
    Extract image URLs from markdown content.
    
    Args:
        content: Markdown content possibly containing image references
        
    Returns:
        List of image URLs found in the content
    """
    urls = re.findall(IMAGE_URL_REGEX, content)
    return urls

def is_valid_image_url(url: str) -> bool:
    """
    Check if a URL is likely a valid image URL.
    
    Args:
        url: URL to check
        
    Returns:
        Boolean indicating if the URL appears to be a valid image URL
    """
    # Check if URL has an image extension
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    # Check common image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
    has_image_extension = any(path.endswith(ext) for ext in valid_extensions)
    
    # GitHub attachment URLs don't always have extensions
    is_github_attachment = (
        "user-content" in url or 
        "github.com/user-attachments" in url or
        "githubusercontent.com" in url
    )
    
    return has_image_extension or is_github_attachment

def create_image_content_item(url: str) -> Dict[str, Any]:
    """
    Create a content item for an image URL in the format expected by OpenAI's API.
    
    Args:
        url: URL of the image
        
    Returns:
        Dictionary with the image content item
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": url
        }
    }

def get_image_recognition_model(
    provider: str = "azure",
    model: str = "gpt-4o",
    **kwargs
) -> BaseChatModel:
    """
    Get a chat model with vision capabilities based on the specified provider.
    
    Args:
        provider: The provider to use ("openai" or "azure")
        model: Model name/deployment name to use
        **kwargs: Additional arguments for the model
        
    Returns:
        A LangChain chat model with vision capabilities
    """
    if provider.lower() == "azure":
        # Azure OpenAI
        return get_azure_vision_model(
            deployment=model or os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o"),
            **kwargs
        )
    else:
        # Standard OpenAI
        return get_openai_vision_model(
            model=model or "gpt-4o",
            **kwargs
        )

def get_openai_vision_model(
    model: str = "gpt-4o",
    api_key: str = None,
    **kwargs
) -> ChatOpenAI:
    """
    Get a LangChain chat model for OpenAI with vision capabilities.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4o")
        api_key: OpenAI API key
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
        **kwargs
    )

def get_azure_vision_model(
    deployment: str = None,
    endpoint: str = None,
    api_version: str = "2025-01-01-preview",
    api_key: str = None,
    **kwargs
) -> AzureChatOpenAI:
    """
    Get a LangChain chat model for Azure OpenAI with vision capabilities.
    
    Args:
        deployment: Azure OpenAI deployment name
        endpoint: Azure OpenAI endpoint URL
        api_version: Azure OpenAI API version
        api_key: Azure OpenAI API key
        **kwargs: Additional arguments for the AzureChatOpenAI model
        
    Returns:
        An AzureChatOpenAI instance
    """
    endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = deployment or os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o")
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint is required. Provide it as an argument or set the AZURE_OPENAI_ENDPOINT environment variable.")
    if not deployment:
        raise ValueError("Azure OpenAI deployment name is required. Provide it as an argument or set the AZURE_OPENAI_VISION_DEPLOYMENT environment variable.")
    
    auth_args = {}
    if api_key:
        auth_args["api_key"] = api_key
    else:
        # Use Azure AD authentication
        try:
            from azure.identity import DefaultAzureCredential
            
            # Create a function that returns a token when called
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
        **auth_args,
        **kwargs
    )

def analyze_image(
    image_url: str,
    prompt: str = "Analyze this image and describe what you see in detail. Include any visible text, objects, people, and notable features.",
    llm: BaseChatModel = None,
    provider: str = "azure",
    **kwargs
) -> str:
    """
    Analyze a single image URL and return a detailed description.
    
    Args:
        image_url: URL of the image to analyze
        prompt: Text prompt to guide the analysis
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Detailed description of the image
    """
    if not is_valid_image_url(image_url):
        return f"Error: Invalid or unsupported image URL: {image_url}"
    
    # Get or create the LLM
    model = llm or get_image_recognition_model(provider=provider, **kwargs)
    
    # Create a message with both text and image
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            create_image_content_item(image_url)
        ]
    )
    
    # Invoke the model
    try:
        response = model.invoke([message])
        return response.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def process_text_with_images(
    content: str,
    llm: BaseChatModel = None,
    provider: str = "azure",
    keep_images: bool = False,
    **kwargs
) -> str:
    """
    Process markdown content that may include image references, replacing each image with a detailed description.
    
    Args:
        content: Markdown content with potential image references
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        keep_images: If True, keep the original image references in addition to descriptions
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Processed content with image descriptions
    """
    # Extract image URLs
    image_urls = extract_image_urls(content)
    
    if not image_urls:
        return content  # No images to process
    
    # Get or create the LLM
    model = llm or get_image_recognition_model(provider=provider, **kwargs)
    
    # Make a copy of the original content to modify
    processed_content = content
    
    # Process each image and replace in the content
    for image_url in image_urls:
        if not is_valid_image_url(image_url):
            print(f"Skipping invalid image URL: {image_url}")
            continue
        
        # Check if this URL exists in the content to prevent processing duplicates
        markdown_pattern = f"![Image]({image_url})"
        if markdown_pattern not in processed_content:
            print(f"Image URL pattern not found in content: {markdown_pattern}")
            continue
            
        # Get the image description
        try:
            print(f"Analyzing image: {image_url}")
            description = analyze_image(
                image_url=image_url, 
                prompt="Describe this image in detail. Include any visible text, objects, and overall context.",
                llm=model, 
                **kwargs
            )
            
            if not description or description.startswith("Error"):
                print(f"Warning: Failed to get description for {image_url}: {description}")
                continue
                
            print(f"Got description of length {len(description)}")
            
            # Format the replacement
            if keep_images:
                # Keep the original image reference and add description
                replacement = f"![Image]({image_url})\n\n**Image Description:** {description}\n\n"
            else:
                # Replace image reference with description only
                replacement = f"**Image Description:** {description}\n\n"
            
            # Replace in content (simple string replacement first)
            if markdown_pattern in processed_content:
                processed_content = processed_content.replace(markdown_pattern, replacement)
            else:
                # Try with more complex regex pattern if simple replacement fails
                # This handles potential variations in the markdown syntax
                image_pattern = f"!\\[[^\\]]*\\]\\({re.escape(image_url)}\\)"
                processed_content = re.sub(image_pattern, replacement, processed_content)
                
            print(f"Replaced image reference for {image_url}")
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
    
    # Verify that changes were made
    if processed_content == content:
        print("Warning: No changes were made to the content. Check image URLs and model responses.")
    
    return processed_content

def analyze_issue_with_images(
    issue_title: str,
    issue_body: str,
    llm: BaseChatModel = None,
    provider: str = "azure",
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a GitHub issue that may contain images, providing a comprehensive analysis.
    
    Args:
        issue_title: Title of the GitHub issue
        issue_body: Body of the GitHub issue, potentially containing image references
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Dictionary with analysis results including image descriptions
    """
    # Get or create the LLM
    model = llm or get_image_recognition_model(provider=provider, **kwargs)
    
    # Extract image URLs from the issue body
    image_urls = extract_image_urls(issue_body)
    
    # Process any images in the issue body
    processed_body = issue_body
    image_descriptions = []
    
    for image_url in image_urls:
        if is_valid_image_url(image_url):
            # Get image description
            description = analyze_image(image_url, llm=model, **kwargs)
            image_descriptions.append({
                "url": image_url,
                "description": description
            })
            
            # Replace in the body
            image_pattern = re.escape(f"![Image]({image_url})").replace("\\!", "!").replace("\\[", "[").replace("\\]", "]").replace("\\(", "(").replace("\\)", ")")
            processed_body = re.sub(image_pattern, f"![Image]({image_url})\n\n**Image Description:** {description}\n\n", processed_body)
    
    # Create prompt for combined analysis if there are images
    if image_descriptions:
        # Enhanced system prompt to get direct analysis without introductory phrases
        system_prompt = """You are an expert at analyzing software issues that include both text and images.
        
        Provide a direct, concise, and comprehensive analysis of the issue based on both the text content and any images provided.
        
        Focus on:
        1. Identifying the problem
        2. Analyzing potential causes
        3. Highlighting visual evidence from the images that supports your analysis
        
        IMPORTANT:
        - Do NOT include introductory phrases like "Based on the information provided" or "Looking at the screenshot"
        - Do NOT write in first person or use phrases like "I will analyze" or "I can see"
        - Start DIRECTLY with your analysis, using headers and structured sections
        - Be factual, objective, and avoid filler language"""
        
        # Create content with text and images for comprehensive analysis
        human_content = [{"type": "text", "text": f"Analyze this issue and provide a concise, comprehensive breakdown:\n\nIssue Title: {issue_title}\n\nIssue Body: {processed_body}"}]
        
        # Add up to 5 images (API limit) directly to the content for unified analysis
        for i, img_data in enumerate(image_descriptions[:5]):
            human_content.append(create_image_content_item(img_data["url"]))
        
        # Create message and invoke model
        message = HumanMessage(content=human_content)
        
        try:
            response = model.invoke([
                SystemMessage(content=system_prompt),
                message
            ])
            
            comprehensive_analysis = response.content
        except Exception as e:
            comprehensive_analysis = f"Error performing comprehensive analysis: {str(e)}"
    else:
        comprehensive_analysis = "No images found in the issue for visual analysis."
    
    # Return results
    return {
        "processed_body": processed_body,
        "image_descriptions": image_descriptions,
        "comprehensive_analysis": comprehensive_analysis,
        "has_images": len(image_descriptions) > 0
    }