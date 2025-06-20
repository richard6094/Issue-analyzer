"""
Token optimization utilities for embedding models.

This module provides accurate token counting, chunk size optimization,
and query protection for various embedding models.
"""

import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    name: str
    max_tokens: int
    encoding_name: Optional[str] = None
    tokenizer_model: Optional[str] = None
    
# Comprehensive model configurations
MODEL_CONFIGS = {
    # OpenAI Embedding Models
    "text-embedding-ada-002": ModelConfig("text-embedding-ada-002", 8191, "cl100k_base"),
    "text-embedding-3-small": ModelConfig("text-embedding-3-small", 8191, "cl100k_base"),
    "text-embedding-3-large": ModelConfig("text-embedding-3-large", 8191, "cl100k_base"),
    
    # Azure OpenAI (same as OpenAI models)
    "text-embedding-ada-002-azure": ModelConfig("text-embedding-ada-002", 8191, "cl100k_base"),
    "text-embedding-3-small-azure": ModelConfig("text-embedding-3-small", 8191, "cl100k_base"),
    "text-embedding-3-large-azure": ModelConfig("text-embedding-3-large", 8191, "cl100k_base"),
    
    # Common fallbacks
    "default": ModelConfig("default", 8191, "cl100k_base"),
    "sentence-transformers": ModelConfig("sentence-transformers", 512, None),  # Most ST models have 512 token limit
}

class TokenCounter(ABC):
    """Abstract base class for token counting."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        pass

class TikTokenCounter(TokenCounter):
    """Accurate token counter using tiktoken."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for accurate token counting. Install with: pip install tiktoken")
        
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.encoding.decode(tokens)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        if max_tokens <= 0:
            return ""
        
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.decode(truncated_tokens)

class ApproximateTokenCounter(TokenCounter):
    """Fallback token counter using approximation."""
    
    def __init__(self, tokens_per_word: float = 1.3):
        """
        Initialize with tokens per word ratio.
        
        Args:
            tokens_per_word: Average tokens per word (1.3 is a reasonable default for English)
        """
        self.tokens_per_word = tokens_per_word
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count based on words."""
        if not text:
            return 0
        
        # Simple word-based counting with adjustments for punctuation
        words = len(text.split())
        # Add some tokens for punctuation and subword tokenization
        punctuation_factor = len(re.findall(r'[.,!?;:]', text)) * 0.1
        return int(words * self.tokens_per_word + punctuation_factor)
    
    def encode(self, text: str) -> List[int]:
        """Mock encoding - returns word indices."""
        words = text.split()
        return list(range(len(words)))
    
    def decode(self, tokens: List[int]) -> str:
        """Mock decoding - cannot accurately reconstruct text."""
        return f"[Decoded {len(tokens)} tokens]"
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Approximate text truncation."""
        if max_tokens <= 0:
            return ""
        
        words = text.split()
        estimated_tokens_per_word = self.tokens_per_word
        max_words = int(max_tokens / estimated_tokens_per_word)
        
        if max_words >= len(words):
            return text
        
        return " ".join(words[:max_words])

class TokenOptimizer:
    """Main class for token optimization and management."""
    
    def __init__(self, 
                 model_name: str = "text-embedding-3-small",
                 safety_margin: float = 0.95,
                 min_chunk_tokens: int = 50,
                 preferred_overlap_tokens: int = 100):
        """
        Initialize token optimizer.
        
        Args:
            model_name: Name of the embedding model
            safety_margin: Safety margin for token limits (0.95 = use 95% of max tokens)
            min_chunk_tokens: Minimum tokens per chunk
            preferred_overlap_tokens: Preferred overlap between chunks in tokens
        """
        self.model_name = model_name
        self.safety_margin = safety_margin
        self.min_chunk_tokens = min_chunk_tokens
        self.preferred_overlap_tokens = preferred_overlap_tokens
        
        # Get model configuration
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])
        
        # Calculate effective token limits
        self.max_tokens = self.model_config.max_tokens
        self.effective_max_tokens = int(self.max_tokens * safety_margin)
        
        # Initialize token counter
        self.token_counter = self._create_token_counter()
        
        print(f"TokenOptimizer initialized for {model_name}")
        print(f"Model max tokens: {self.max_tokens}")
        print(f"Effective max tokens (with {safety_margin:.1%} safety): {self.effective_max_tokens}")
        print(f"Token counter: {type(self.token_counter).__name__}")
    
    def _create_token_counter(self) -> TokenCounter:
        """Create appropriate token counter based on model and availability."""
        if TIKTOKEN_AVAILABLE and self.model_config.encoding_name:
            try:
                return TikTokenCounter(self.model_config.encoding_name)
            except Exception as e:
                print(f"Warning: Failed to create tiktoken counter: {e}")
                print("Falling back to approximate counting")
        
        return ApproximateTokenCounter()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.token_counter.count_tokens(text)
    
    def validate_query_length(self, query: str) -> Tuple[bool, int, str]:
        """
        Validate if query length is within model limits.
        
        Args:
            query: Query text to validate
            
        Returns:
            Tuple of (is_valid, token_count, message)
        """
        token_count = self.count_tokens(query)
        
        if token_count <= self.effective_max_tokens:
            return True, token_count, f"Query is valid ({token_count} tokens)"
        else:
            excess = token_count - self.effective_max_tokens
            message = f"Query too long: {token_count} tokens (exceeds limit by {excess} tokens)"
            return False, token_count, message
    
    def truncate_query_if_needed(self, query: str) -> Tuple[str, bool, str]:
        """
        Truncate query if it exceeds token limits.
        
        Args:
            query: Original query text
            
        Returns:
            Tuple of (processed_query, was_truncated, message)
        """
        is_valid, token_count, message = self.validate_query_length(query)
        
        if is_valid:
            return query, False, message
        
        # Truncate to effective max tokens
        if hasattr(self.token_counter, 'truncate_to_tokens'):
            truncated_query = self.token_counter.truncate_to_tokens(query, self.effective_max_tokens)
            new_token_count = self.count_tokens(truncated_query)
            truncate_message = f"Query truncated from {token_count} to {new_token_count} tokens"
            return truncated_query, True, truncate_message
        else:
            # Fallback truncation for approximate counter
            words = query.split()
            estimated_words = int(self.effective_max_tokens / 1.3)  # Approximate words
            if estimated_words < len(words):
                truncated_query = " ".join(words[:estimated_words])
                truncate_message = f"Query truncated (approximately) from {token_count} to ~{self.effective_max_tokens} tokens"
                return truncated_query, True, truncate_message
            else:
                return query, False, "Unable to truncate effectively"
    
    def calculate_optimal_chunk_size(self, text_length: int) -> int:
        """
        Calculate optimal chunk size based on text length and model limits.
        
        Args:
            text_length: Length of text in tokens
            
        Returns:
            Optimal chunk size in tokens
        """
        # If text is shorter than effective max, use it as single chunk
        if text_length <= self.effective_max_tokens:
            return text_length
        
        # Calculate number of chunks needed
        chunks_needed = math.ceil(text_length / self.effective_max_tokens)
        
        # Adjust for overlap
        overlap_total = (chunks_needed - 1) * self.preferred_overlap_tokens
        available_content_tokens = text_length - overlap_total
        
        # Calculate chunk size ensuring we don't exceed limits
        optimal_chunk_size = min(
            self.effective_max_tokens,
            max(
                self.min_chunk_tokens,
                int(available_content_tokens / chunks_needed)
            )
        )
        
        return optimal_chunk_size
    
    def smart_chunk_text(self, text: str, overlap_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Intelligently chunk text with optimal token sizes.
        
        Args:
            text: Text to chunk
            overlap_tokens: Override default overlap tokens
            
        Returns:
            List of chunk dictionaries with text, start_pos, end_pos, and token_count
        """
        if not text.strip():
            return []
        
        overlap_tokens = overlap_tokens or self.preferred_overlap_tokens
        total_tokens = self.count_tokens(text)
        
        # If text fits in single chunk, return as-is
        if total_tokens <= self.effective_max_tokens:
            return [{
                "text": text,
                "start_pos": 0,
                "end_pos": len(text),
                "token_count": total_tokens,
                "chunk_index": 0
            }]
        
        # Calculate optimal chunk size
        optimal_chunk_size = self.calculate_optimal_chunk_size(total_tokens)
        
        chunks = []
        start_pos = 0
        chunk_index = 0
        
        # For tiktoken, we can work with token-level precision
        if isinstance(self.token_counter, TikTokenCounter):
            return self._smart_chunk_with_tiktoken(text, optimal_chunk_size, overlap_tokens)
        else:
            return self._smart_chunk_approximate(text, optimal_chunk_size, overlap_tokens)
    
    def _smart_chunk_with_tiktoken(self, text: str, chunk_size: int, overlap_tokens: int) -> List[Dict[str, Any]]:
        """Precise chunking using tiktoken."""
        tokens = self.token_counter.encode(text)
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Extract chunk tokens and decode
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.token_counter.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "start_token": start_idx,
                "end_token": end_idx,
                "token_count": len(chunk_tokens),
                "chunk_index": chunk_index
            })
            
            # Move start position for next chunk (with overlap)
            start_idx = end_idx - overlap_tokens
            chunk_index += 1
            
            # Avoid infinite loop with very small texts
            if start_idx >= end_idx - overlap_tokens:
                break
        
        return chunks
    
    def _smart_chunk_approximate(self, text: str, chunk_size: int, overlap_tokens: int) -> List[Dict[str, Any]]:
        """Approximate chunking for fallback counter."""
        # Split by sentences or paragraphs for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "start_pos": len("".join([c["text"] for c in chunks])),
                    "end_pos": len("".join([c["text"] for c in chunks])) + len(current_chunk),
                    "token_count": current_tokens,
                    "chunk_index": chunk_index
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if there's content
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "start_pos": len("".join([c["text"] for c in chunks])),
                "end_pos": len("".join([c["text"] for c in chunks])) + len(current_chunk),
                "token_count": current_tokens,
                "chunk_index": chunk_index
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens worth of text for overlap."""
        if isinstance(self.token_counter, TikTokenCounter):
            tokens = self.token_counter.encode(text)
            if len(tokens) <= overlap_tokens:
                return text
            overlap_token_ids = tokens[-overlap_tokens:]
            return self.token_counter.decode(overlap_token_ids)
        else:
            # Approximate overlap by words
            words = text.split()
            estimated_words = max(1, int(overlap_tokens / 1.3))
            return " ".join(words[-estimated_words:])
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization configuration stats."""
        return {
            "model_name": self.model_name,
            "model_max_tokens": self.max_tokens,
            "effective_max_tokens": self.effective_max_tokens,
            "safety_margin": self.safety_margin,
            "min_chunk_tokens": self.min_chunk_tokens,
            "preferred_overlap_tokens": self.preferred_overlap_tokens,
            "token_counter_type": type(self.token_counter).__name__,
            "tiktoken_available": TIKTOKEN_AVAILABLE,
            "encoding_name": getattr(self.model_config, 'encoding_name', 'N/A')
        }

def create_token_optimizer(model_name: str = "text-embedding-3-small", **kwargs) -> TokenOptimizer:
    """
    Factory function to create a token optimizer.
    
    Args:
        model_name: Name of the embedding model
        **kwargs: Additional arguments for TokenOptimizer
        
    Returns:
        Configured TokenOptimizer instance
    """
    return TokenOptimizer(model_name=model_name, **kwargs)

# Convenience functions for common operations
def count_tokens(text: str, model_name: str = "text-embedding-3-small") -> int:
    """Quick token counting."""
    optimizer = create_token_optimizer(model_name)
    return optimizer.count_tokens(text)

def validate_query(query: str, model_name: str = "text-embedding-3-small") -> Tuple[bool, int, str]:
    """Quick query validation."""
    optimizer = create_token_optimizer(model_name)
    return optimizer.validate_query_length(query)

def smart_chunk(text: str, model_name: str = "text-embedding-3-small", **kwargs) -> List[Dict[str, Any]]:
    """Quick smart chunking."""
    optimizer = create_token_optimizer(model_name)
    return optimizer.smart_chunk_text(text, **kwargs)
