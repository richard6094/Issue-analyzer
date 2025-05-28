"""
Image recognition provider using LangChain architecture with GPT-4o Vision capabilities.
"""
from typing import Dict, Any, List, Optional, Union
import os
import json
import re
import uuid  # Added import for uuid module
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
# Regular expression for matching image URLs in HTML format
HTML_IMAGE_URL_REGEX = r'<img[^>]*src=[\'"]([^\'"]+)[\'"][^>]*>'

def extract_image_urls(content: str) -> List[str]:
    """
    Extract image URLs from markdown content or HTML img tags.
    
    Args:
        content: Markdown or HTML content possibly containing image references
        
    Returns:
        List of image URLs found in the content
    """
    # Extract Markdown image URLs
    markdown_urls = re.findall(IMAGE_URL_REGEX, content)
    
    # Extract HTML image URLs
    html_urls = re.findall(HTML_IMAGE_URL_REGEX, content)
    
    # Combine both types of URLs
    urls = markdown_urls + html_urls
    
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
        # Check if deployment is already in kwargs to avoid duplicate parameters
        if 'deployment' not in kwargs:
            kwargs['deployment'] = model or os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o")
        return get_azure_vision_model(**kwargs)
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
    print_descriptions: bool = True,
    **kwargs
) -> str:
    """
    Process content that may include image references (both Markdown and HTML),
    replacing each image with a detailed description.
    
    Args:
        content: Content with potential image references
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        keep_images: If True, keep the original image references in addition to descriptions
        print_descriptions: If True, print the image descriptions to console
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
                
            # print the description if requested
            if print_descriptions:
                print("\n" + "=" * 40)
                print("Image description:")
                print("-" * 40)
                print(description)
                print("=" * 40 + "\n")
            
            print(f"Got description of length {len(description)}")
            
            # Format the replacement
            if keep_images:
                # Keep the original image reference and add description
                replacement = f"![Image]({image_url})\n\n**Image Description:** {description}\n\n"
            else:
                # Replace image reference with description only
                replacement = f"**Image Description:** {description}\n\n"
            
            # Try to replace the content based on the image format
            replaced = False
            
            # Try Markdown format first
            markdown_pattern = f"![Image]({image_url})"
            if markdown_pattern in processed_content:
                processed_content = processed_content.replace(markdown_pattern, replacement)
                replaced = True
            else:
                # Try with more complex Markdown regex pattern
                image_pattern = f"!\\[[^\\]]*\\]\\({re.escape(image_url)}\\)"
                if re.search(image_pattern, processed_content):
                    processed_content = re.sub(image_pattern, replacement, processed_content)
                    replaced = True
            
            # If not replaced yet, try HTML format
            if not replaced:
                # Create HTML pattern based on the URL
                html_pattern = f'<img[^>]*src=[\'"]?{re.escape(image_url)}[\'"]?[^>]*>'
                if re.search(html_pattern, processed_content):
                    processed_content = re.sub(html_pattern, replacement, processed_content)
                    replaced = True
                    
            if replaced:
                print(f"Replaced image reference for {image_url}")
            else:
                print(f"Image URL pattern not found in content for: {image_url}")
                
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
    max_context_length: int = 1000,  # 限制上下文长度
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze images in a GitHub issue, using text content primarily as background context to enhance image understanding.
    The focus is on providing detailed descriptions of the images rather than analyzing the issue as a whole.
    
    Args:
        issue_title: Title of the GitHub issue
        issue_body: Body of the GitHub issue, potentially containing image references
        llm: Optional pre-configured LangChain chat model
        provider: Provider to use if llm not provided ("openai" or "azure")
        max_context_length: Maximum length of surrounding text context to include
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Dictionary with image descriptions as the primary output, with the issue text serving as context
    """
    # Get or create the LLM
    model = llm or get_image_recognition_model(provider=provider, **kwargs)
    
    # Extract image URLs from the issue body
    image_urls = extract_image_urls(issue_body)
    
    # If no images found, return early
    if not image_urls:
        return {
            "has_images": False,
            "image_descriptions": [],
            "message": "No images found in the issue content."
        }
    
    # Filter for valid image URLs
    valid_image_urls = [url for url in image_urls if is_valid_image_url(url)]
    if not valid_image_urls:
        return {
            "has_images": False,
            "image_descriptions": [],
            "message": "No valid image URLs found in the issue content."
        }
    
    # Extract relevant text context for each image
    image_contexts = extract_image_contexts(issue_body, valid_image_urls, max_context_length)
    
    # Process each image with issue context to enhance understanding
    image_analyses = []
    
    for i, image_url in enumerate(valid_image_urls):
        try:
            # Get the surrounding text context for this image
            surrounding_text = image_contexts.get(image_url, "")
            
            # Create a context-aware prompt that includes issue information
            context_prompt = f"""
            Analyze this image in detail, focusing on the visual content itself.

            For context, this image is from a GitHub issue with the following details:
            Title: {issue_title}
            
            Surrounding text context:
            {surrounding_text}
            
            Use this context only to better understand what's in the image, but focus your description on what you actually see in the image.
            
            Include:
            1. Any visible text, error messages, or code snippets
            2. UI elements, buttons, windows, or application interfaces
            3. Visual indicators of problems (error states, highlighting, unexpected UI states)
            4. Charts, diagrams, or visual data representations
            5. Terminal/console output
            
            Describe ONLY what you can see in the image. Don't make assumptions beyond what's visible.
            """
            
            # Get detailed image description
            description = analyze_image(
                image_url=image_url, 
                prompt=context_prompt,
                llm=model, 
                **kwargs
            )
            
            if not description or description.startswith("Error"):
                print(f"Warning: Failed to get description for {image_url}: {description}")
                continue
            
            # Add to our list of image analyses
            image_analyses.append({
                "url": image_url,
                "description": description,
                "index": i + 1,
                "context": surrounding_text[:100] + "..." if len(surrounding_text) > 100 else surrounding_text  # 添加上下文摘要
            })
            
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
    
    # Return results focusing on the image descriptions
    return {
        "has_images": True,
        "image_count": len(image_analyses),
        "image_descriptions": image_analyses,
        "issue_title": issue_title  # Include the title for reference
    }

def extract_image_contexts(text: str, image_urls: List[str], max_context_length: int = 1000) -> Dict[str, str]:
    """
    Extract the surrounding text context for each image URL in the text.
    
    Args:
        text: The full text containing image references
        image_urls: List of image URLs to find context for
        max_context_length: Maximum length of context to extract
        
    Returns:
        Dictionary mapping image URLs to their surrounding text context
    """
    contexts = {}
    
    for url in image_urls:
        # Find the markdown image reference
        pattern = f"!\\[[^\\]]*\\]\\({re.escape(url)}\\)"
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            # If exact pattern not found, try a more lenient search
            contexts[url] = ""
            continue
            
        # Get the position of the image reference
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract text before the image
            context_start = max(0, start_pos - max_context_length // 2)
            before_text = text[context_start:start_pos].strip()
            
            # Extract text after the image
            context_end = min(len(text), end_pos + max_context_length // 2)
            after_text = text[end_pos:context_end].strip()
            
            # Combine context, prioritizing this match if multiple exist
            contexts[url] = f"{before_text}\n\n{after_text}"
            
    return contexts

def process_issue_images(
    issue_number: int, 
    issue_title: str, 
    issue_body: str, 
    metadata: Dict = None, 
    provider: str = "azure",
    max_context_length: int = 1000,
    endpoint: str = None, 
    deployment: str = None, 
    api_key: str = None, 
    **kwargs 
) -> List[Dict]:
    """
    Unified function to process images in a GitHub issue, extract surrounding context,
    and generate rich image descriptions. This function is designed to be used by both 
    direct analysis scenarios and vector database creation.
    
    Args:
        issue_number: The issue number
        issue_title: The title of the issue
        issue_body: The full text content of the issue
        metadata: Additional metadata to include in the chunks (optional)
        provider: Provider to use for image recognition ("azure" or "openai")
        max_context_length: Maximum length of context to extract around each image
        endpoint: Azure OpenAI endpoint URL (for Azure provider)
        deployment: Azure OpenAI deployment name (for Azure provider)
        api_key: API key for authentication (if needed)
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        List of image description chunks with metadata, ready for vector database storage
        or direct usage
    """
    # Extract image URLs from the issue text
    image_urls = extract_image_urls(issue_body)
    
    # If no images found, return empty list
    if not image_urls:
        return []
    
    # Filter valid image URLs
    valid_image_urls = [url for url in image_urls if is_valid_image_url(url)]
    if not valid_image_urls:
        return []
    
    print(f"Processing {len(valid_image_urls)} valid images in issue #{issue_number}")
    
    # Get or create a vision model for image analysis
    try:
        # Create vision model with provided configuration
        model_kwargs = kwargs.copy()
        if provider.lower() == "azure":
            model_kwargs.update({
                "endpoint": endpoint,
                "deployment": deployment,
                "api_key": api_key
            })
        
        # Create vision model with all parameters
        vision_model = get_image_recognition_model(
            provider=provider,
            **model_kwargs
        )
        
        image_chunks = []
        
        # Extract relevant text context for each image
        image_contexts = extract_image_contexts(issue_body, valid_image_urls, max_context_length)
        
        # Process each image with context
        for i, image_url in enumerate(valid_image_urls):
            try:
                # Get the surrounding text context for this image
                surrounding_text = image_contexts.get(image_url, "")
                
                # Create a context-aware prompt
                context_prompt = f"""
                Analyze this image in detail, focusing on the visual content itself.
                
                For context, this image is from a GitHub issue with the following details:
                Title: {issue_title}
                
                Surrounding text context:
                {surrounding_text}
                
                Use this context only to better understand what's in the image, but focus your description on what you actually see in the image.
                
                Include:
                1. Any visible text, error messages, or code snippets
                2. UI elements, buttons, windows, or application interfaces
                3. Visual indicators of problems (error states, highlighting, unexpected UI states)
                4. Charts, diagrams, or visual data representations
                5. Terminal/console output
                
                Describe ONLY what you can see in the image. Don't make assumptions beyond what's visible.
                """
                
                print(f"Analyzing image {i+1}/{len(valid_image_urls)} in issue #{issue_number}")
                
                # Get detailed image description
                description = analyze_image(
                    image_url=image_url,
                    prompt=context_prompt,
                    llm=vision_model
                )
                
                if not description or description.startswith("Error"):
                    print(f"Failed to analyze image {image_url}: {description}")
                    continue
                
                # Create a chunk ID
                chunk_id = str(uuid.uuid4())
                
                # Create a formatted text
                formatted_text = f"Image URL: {image_url}\n\nImage Description: {description}"
                
                # Prepare base metadata
                base_metadata = {
                    "issue_number": issue_number,
                    "issue_title": issue_title,
                    "chunk_type": "image_description",
                    "position": -1,  # Use -1 to indicate this isn't part of the main text flow
                    "is_image": True,
                    "image_url": image_url,
                    "source_issue": issue_number,
                    "chunk_id": chunk_id,
                    "context_snippet": surrounding_text[:100] + "..." if len(surrounding_text) > 100 else surrounding_text
                }
                
                # Merge with additional metadata if provided
                if metadata:
                    base_metadata.update(metadata)
                
                # Create a chunk for the image description
                image_chunks.append({
                    "text": formatted_text,
                    "metadata": base_metadata
                })
                
                print(f"Successfully processed image {i+1} in issue #{issue_number}")
                
            except Exception as e:
                print(f"Error processing image {image_url} in issue #{issue_number}: {str(e)}")
        
        return image_chunks
        
    except Exception as e:
        print(f"Error setting up image processing for issue #{issue_number}: {str(e)}")
        return []