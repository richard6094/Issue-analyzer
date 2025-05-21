"""
OpenAI embedding provider implementation.
"""
from typing import List, Dict, Any, Optional
import os

try:
    import openai
except ImportError:
    raise ImportError("The 'openai' package is required. Install it with 'pip install openai'.")

from .embedding_provider import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """
    OpenAI embedding provider implementation.
    Uses the OpenAI API to generate embeddings.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
            model: Model name to use for embeddings. Default is "text-embedding-3-small".
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as an argument or set the OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = openai.Client(api_key=self.api_key)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        
        # Sort the embeddings by index to ensure correct order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using OpenAI.
        
        Args:
            text: Text string to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0]