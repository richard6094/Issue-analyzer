"""
Base embedding provider interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text string to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        pass