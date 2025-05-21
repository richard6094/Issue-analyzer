"""
Example script demonstrating how to use the embedding providers.
"""
import argparse
import os
import sys
from typing import List

# Add the parent directory to sys.path to allow importing the embeddings module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import OpenAIEmbedding, AzureOpenAIEmbedding


def get_embedding_provider(args):
    """
    Create and return the appropriate embedding provider based on arguments.
    """
    if args.use_azure_openai:
        # Use Azure OpenAI embeddings
        return AzureOpenAIEmbedding(
            endpoint=args.azure_endpoint,
            deployment=args.azure_deployment,
            api_version=args.azure_api_version
        )
    else:
        # Use OpenAI embeddings
        return OpenAIEmbedding(
            api_key=args.openai_api_key,
            model=args.openai_model
        )


def main():
    """
    Main function to demonstrate embedding provider usage.
    """
    parser = argparse.ArgumentParser(description="Demonstrate embeddings module")
    parser.add_argument("--text", type=str, default="Hello, world!",
                      help="Text to generate embeddings for")
    
    # OpenAI options
    parser.add_argument("--openai-api-key", type=str,
                      help="OpenAI API key (if not set as environment variable)")
    parser.add_argument("--openai-model", type=str, default="text-embedding-3-small",
                      help="OpenAI model to use for embeddings")
    
    # Azure OpenAI options
    parser.add_argument("--use-azure-openai", action="store_true",
                      help="Use Azure OpenAI embeddings instead of regular OpenAI")
    parser.add_argument("--azure-endpoint", type=str,
                      help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-key", type=str,
                      help="Azure OpenAI API key")
    parser.add_argument("--azure-deployment", type=str, default="text-embedding-3-small",
                      help="Azure OpenAI deployment name")
    parser.add_argument("--azure-api-version", type=str, default="2024-02-01",
                      help="Azure OpenAI API version")
    parser.add_argument("--use-entra-id", action="store_true", default=False,
                      help="Use Azure Entra ID authentication instead of API key")
    
    args = parser.parse_args()
    
    try:
        # Get the appropriate embedding provider
        embedding_provider = get_embedding_provider(args)
        
        # Generate embedding for the given text
        print(f"Generating embedding for: '{args.text}'")
        embedding = embedding_provider.get_embedding(args.text)
        
        # Display embedding details
        print(f"Embedding length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Last 5 values: {embedding[-5:]}")
        
        # Try with multiple texts
        texts = [args.text, "This is another example.", "And one more for good measure."]
        print("\nGenerating embeddings for multiple texts:")
        for text in texts:
            print(f" - '{text}'")
        
        embeddings = embedding_provider.get_embeddings(texts)
        for i, emb in enumerate(embeddings):
            print(f"Embedding {i+1} length: {len(emb)}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())