"""
Azure OpenAI embedding provider implementation.
"""
from typing import List, Dict, Any, Optional
import os

try:
    from openai import AzureOpenAI
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    raise ImportError("Required packages are missing. Install them with 'pip install openai azure-identity'.")

from .embedding_provider import EmbeddingProvider


class AzureOpenAIEmbedding(EmbeddingProvider):
    """
    Azure OpenAI embedding provider implementation.
    Uses Azure OpenAI API to generate embeddings with Entra ID authentication.
    """
    
    def __init__(
        self, 
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2024-02-01"
    ):
        """
        Initialize the Azure OpenAI embedding provider with Entra ID authentication.
        
        Args:
            endpoint: Azure OpenAI endpoint URL. If not provided, will try to get from AZURE_OPENAI_ENDPOINT environment variable.
            deployment: Azure OpenAI deployment name. If not provided, will try to get from AZURE_OPENAI_DEPLOYMENT environment variable.
            api_version: Azure OpenAI API version. Default is "2024-02-01".
        """
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Provide it as an argument or set the AZURE_OPENAI_ENDPOINT environment variable.")
        
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise ValueError("Azure OpenAI deployment name is required. Provide it as an argument or set the AZURE_OPENAI_DEPLOYMENT environment variable.")
        
        self.api_version = api_version
        
        # Initialize client with Entra ID authentication
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Azure OpenAI.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment
        )
        
        # Sort the embeddings by index to ensure correct order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using Azure OpenAI.
        
        Args:
            text: Text string to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0]