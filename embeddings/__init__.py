"""
Embeddings module for generating vector embeddings using different providers.
"""

from .embedding_provider import EmbeddingProvider
from .openai_embedding import OpenAIEmbedding
from .azure_openai_embedding import AzureOpenAIEmbedding

__all__ = ['EmbeddingProvider', 'OpenAIEmbedding', 'AzureOpenAIEmbedding']