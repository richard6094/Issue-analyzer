"""
Demonstration of the image recognition features.

This demo shows how to use the image recognition module to:
1. Analyze a single image URL
2. Process text with embedded image references
3. Analyze an issue with images
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import image recognition functions
from image_recognition import (
    analyze_image,
    process_text_with_images,
    analyze_issue_with_images,
    get_image_recognition_model
)

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Image Recognition Demo")
    
    parser.add_argument("--image-url", type=str,
                        help="URL of the image to analyze")
    
    parser.add_argument("--provider", type=str, choices=["azure", "openai"], default=None,
                        help="Provider to use (azure or openai)")
    
    parser.add_argument("--endpoint", type=str,
                        help="Azure OpenAI endpoint URL")
    
    parser.add_argument("--deployment", type=str,
                        help="Azure OpenAI deployment name or model name")
    
    parser.add_argument("--api-key", type=str,
                        help="API key for the chosen provider")
    
    parser.add_argument("--keep-images", action="store_true",
                        help="Keep original image references in the output")
    
    return parser.parse_args()

def main():
    """Run the image recognition demo."""
    # Load environment variables for API keys
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    print_section("Image Recognition Demo")
    
    # Example image URLs (GitHub user-attachments format)
    # Use command line argument if provided, otherwise use default
    sample_image_url = args.image_url or "https://github.com/user-attachments/assets/a7f73cb4-9256-44b7-a0c1-68cfaca65505"
    print(f"Using image URL: {sample_image_url}")
    
    # Set provider based on arguments or available credentials
    provider = args.provider
    if provider is None:
        provider = "azure" if os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_ENDPOINT") else "openai"
    print(f"Using provider: {provider}")
    
    # Create model kwargs based on provided arguments
    model_kwargs = {}
    
    if args.endpoint:
        print(f"Using custom endpoint: {args.endpoint}")
        model_kwargs["endpoint"] = args.endpoint
    
    # Handle deployment/model parameter
    model = None
    if args.deployment:
        print(f"Using custom deployment/model: {args.deployment}")
        model = args.deployment
    
    if args.api_key:
        print("Using provided API key")
        model_kwargs["api_key"] = args.api_key
    
    # Example 1: Analyze a single image
    print_section("Example 1: Single Image Analysis")
    
    try:
        image_description = analyze_image(
            image_url=sample_image_url,
            prompt="Describe this image in detail. Include any text visible in the image, objects, and overall context.",
            provider=provider,
            model=model,  # Pass as model parameter instead of in kwargs
            **model_kwargs
        )
        
        print("Image analysis result:")
        print(image_description)
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
    
    # Example 2: Process text with embedded images
    print_section("Example 2: Process Text with Embedded Images")
    
    sample_text = f"""
# Sample Issue Report

We're experiencing a problem with our application.
When I click the button shown in the screenshot below, the application crashes.

![Image]({sample_image_url})

I've tried restarting the application, but the issue persists.
Can someone help me understand what's happening?
"""
    
    print("Original text with image reference:")
    print(sample_text)
    print("\n")
    
    try:
        processed_text = process_text_with_images(
            content=sample_text,
            provider=provider,
            model=model,  # Pass as model parameter instead of in kwargs
            keep_images=args.keep_images,  # Use the command line argument
            **model_kwargs
        )
        
        print("Processed text with image descriptions:")
        print(processed_text)
    except Exception as e:
        print(f"Error processing text with images: {str(e)}")
    
    # Example 3: Analyze a full issue with images
    print_section("Example 3: Analyze Issue with Images")
    
    issue_title = "Application crashes when clicking the submit button"
    issue_body = f"""
When I try to submit a form in the application, it crashes immediately.

Steps to reproduce:
1. Open the application
2. Navigate to the form page
3. Fill out the required fields
4. Click the submit button shown in the screenshot below

![Image]({sample_image_url})

The application version is 1.2.3 running on Windows 10.
This was working fine in version 1.2.2.
"""
    
    print("Issue title:", issue_title)
    print("Issue body:")
    print(issue_body)
    print("\n")
    
    try:
        issue_analysis = analyze_issue_with_images(
            issue_title=issue_title,
            issue_body=issue_body,
            provider=provider,
            model=model,  # Pass as model parameter instead of in kwargs
            **model_kwargs
        )
        
        print("Issue analysis results:")
        print(json.dumps(issue_analysis, indent=2))
        
        print("\nComprehensive analysis:")
        print(issue_analysis["comprehensive_analysis"])
    except Exception as e:
        print(f"Error analyzing issue with images: {str(e)}")

if __name__ == "__main__":
    main()