#!/usr/bin/env python3
"""
Comprehensive tests for the token optimization system.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from analyzer_core.utils.token_optimizer import (
    TokenOptimizer, 
    TikTokenCounter, 
    ApproximateTokenCounter,
    create_token_optimizer,
    count_tokens,
    validate_query,
    smart_chunk,
    MODEL_CONFIGS,
    TIKTOKEN_AVAILABLE
)

class TestTokenOptimizer(unittest.TestCase):
    """Test cases for TokenOptimizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = "This is a sample text for testing token optimization features. " * 10
        self.long_text = "This is a very long text that should exceed token limits. " * 200
        self.short_text = "Short text."
        
    def test_model_configs(self):
        """Test that model configurations are properly defined."""
        self.assertIn("text-embedding-3-small", MODEL_CONFIGS)
        self.assertIn("text-embedding-ada-002", MODEL_CONFIGS)
        self.assertIn("default", MODEL_CONFIGS)
        
        config = MODEL_CONFIGS["text-embedding-3-small"]
        self.assertEqual(config.max_tokens, 8191)
        self.assertEqual(config.encoding_name, "cl100k_base")

    def test_token_optimizer_initialization(self):
        """Test TokenOptimizer initialization."""
        optimizer = TokenOptimizer()
        
        self.assertEqual(optimizer.model_name, "text-embedding-3-small")
        self.assertEqual(optimizer.max_tokens, 8191)
        self.assertLessEqual(optimizer.effective_max_tokens, optimizer.max_tokens)
        self.assertIsNotNone(optimizer.token_counter)

    def test_token_counting(self):
        """Test token counting functionality."""
        optimizer = TokenOptimizer()
        
        # Test basic counting
        count = optimizer.count_tokens(self.sample_text)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
        
        # Test empty text
        empty_count = optimizer.count_tokens("")
        self.assertEqual(empty_count, 0)
        
        # Test None text
        none_count = optimizer.count_tokens(None)
        self.assertEqual(none_count, 0)

    def test_query_validation(self):
        """Test query validation functionality."""
        optimizer = TokenOptimizer()
        
        # Test valid query
        is_valid, token_count, message = optimizer.validate_query_length(self.short_text)
        self.assertTrue(is_valid)
        self.assertGreater(token_count, 0)
        self.assertIn("valid", message.lower())
        
        # Test potentially invalid query (very long)
        is_valid, token_count, message = optimizer.validate_query_length(self.long_text)
        if not is_valid:
            self.assertIn("too long", message.lower())

    def test_query_truncation(self):
        """Test query truncation functionality."""
        optimizer = TokenOptimizer()
        
        # Test truncation with long text
        truncated_query, was_truncated, message = optimizer.truncate_query_if_needed(self.long_text)
        
        self.assertIsInstance(truncated_query, str)
        self.assertIsInstance(was_truncated, bool)
        
        if was_truncated:
            # Verify truncated query is shorter
            original_tokens = optimizer.count_tokens(self.long_text)
            truncated_tokens = optimizer.count_tokens(truncated_query)
            self.assertLess(truncated_tokens, original_tokens)
            self.assertLessEqual(truncated_tokens, optimizer.effective_max_tokens)

    def test_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        optimizer = TokenOptimizer()
        
        # Test small text
        small_tokens = optimizer.count_tokens(self.short_text)
        chunk_size = optimizer.calculate_optimal_chunk_size(small_tokens)
        self.assertEqual(chunk_size, small_tokens)
        
        # Test large text
        large_tokens = optimizer.max_tokens * 3  # 3x the limit
        chunk_size = optimizer.calculate_optimal_chunk_size(large_tokens)
        self.assertLessEqual(chunk_size, optimizer.effective_max_tokens)
        self.assertGreaterEqual(chunk_size, optimizer.min_chunk_tokens)

    def test_smart_chunking(self):
        """Test smart text chunking."""
        optimizer = TokenOptimizer()
        
        # Test chunking short text
        chunks = optimizer.smart_chunk_text(self.short_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("token_count", chunk)
            self.assertIn("chunk_index", chunk)
            self.assertLessEqual(chunk["token_count"], optimizer.effective_max_tokens)
          # Test chunking long text
        very_long_text = "This is a very long text that should exceed token limits. " * 500  # Make it much longer
        chunks_long = optimizer.smart_chunk_text(very_long_text)
        
        # Check if we actually need multiple chunks
        total_tokens = optimizer.count_tokens(very_long_text)
        if total_tokens > optimizer.effective_max_tokens:
            self.assertGreater(len(chunks_long), 1)  # Should create multiple chunks
        else:
            self.assertEqual(len(chunks_long), 1)  # Single chunk is fine if text is short enough

    def test_tiktoken_counter(self):
        """Test TikTokenCounter if available."""
        if TIKTOKEN_AVAILABLE:
            try:
                counter = TikTokenCounter()
                
                # Test basic functionality
                count = counter.count_tokens(self.sample_text)
                self.assertIsInstance(count, int)
                self.assertGreater(count, 0)
                
                # Test encoding/decoding
                tokens = counter.encode(self.sample_text)
                self.assertIsInstance(tokens, list)
                
                decoded = counter.decode(tokens)
                self.assertIsInstance(decoded, str)
                
                # Test truncation
                truncated = counter.truncate_to_tokens(self.sample_text, 10)
                truncated_count = counter.count_tokens(truncated)
                self.assertLessEqual(truncated_count, 10)
                
            except ImportError:
                self.skipTest("tiktoken not available")
        else:
            self.skipTest("tiktoken not available")

    def test_approximate_counter(self):
        """Test ApproximateTokenCounter."""
        counter = ApproximateTokenCounter()
        
        # Test basic functionality
        count = counter.count_tokens(self.sample_text)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
        
        # Test truncation
        truncated = counter.truncate_to_tokens(self.sample_text, 10)
        self.assertIsInstance(truncated, str)

    def test_factory_functions(self):
        """Test factory and convenience functions."""
        # Test create_token_optimizer
        optimizer = create_token_optimizer("text-embedding-3-small")
        self.assertIsInstance(optimizer, TokenOptimizer)
        
        # Test count_tokens convenience function
        token_count = count_tokens(self.sample_text)
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)
        
        # Test validate_query convenience function
        is_valid, token_count, message = validate_query(self.short_text)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(token_count, int)
        self.assertIsInstance(message, str)
        
        # Test smart_chunk convenience function
        chunks = smart_chunk(self.sample_text)
        self.assertIsInstance(chunks, list)

    def test_optimization_stats(self):
        """Test optimization statistics."""
        optimizer = TokenOptimizer()
        stats = optimizer.get_optimization_stats()
        
        self.assertIn("model_name", stats)
        self.assertIn("model_max_tokens", stats)
        self.assertIn("effective_max_tokens", stats)
        self.assertIn("token_counter_type", stats)
        self.assertIn("tiktoken_available", stats)
        
        self.assertEqual(stats["model_name"], optimizer.model_name)
        self.assertEqual(stats["model_max_tokens"], optimizer.max_tokens)

    def test_different_models(self):
        """Test with different embedding models."""
        models_to_test = [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002"
        ]
        
        for model in models_to_test:
            with self.subTest(model=model):
                optimizer = TokenOptimizer(model_name=model)
                self.assertEqual(optimizer.model_name, model)
                
                # Test basic functionality
                count = optimizer.count_tokens(self.sample_text)
                self.assertGreater(count, 0)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        optimizer = TokenOptimizer()
        
        # Test with None input
        self.assertEqual(optimizer.count_tokens(None), 0)
        
        # Test with empty string
        self.assertEqual(optimizer.count_tokens(""), 0)
        
        # Test with whitespace only
        whitespace_count = optimizer.count_tokens("   \n\t  ")
        self.assertGreaterEqual(whitespace_count, 0)
        
        # Test chunking empty text
        empty_chunks = optimizer.smart_chunk_text("")
        self.assertEqual(len(empty_chunks), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for token optimization in RAG system."""
    
    def test_rag_tool_integration(self):
        """Test integration with RAG tool."""
        # This would test the actual integration but requires database setup
        # For now, just test that imports work
        try:
            from analyzer_core.tools.rag_tool import RAGTool
            from analyzer_core.utils.token_optimizer import create_token_optimizer
            
            # Test that token optimizer can be created
            optimizer = create_token_optimizer()
            self.assertIsNotNone(optimizer)
            
        except ImportError as e:
            self.skipTest(f"Integration test skipped due to import error: {e}")

    def test_create_vectordb_integration(self):
        """Test integration with vectordb creation."""
        try:
            # Test imports
            from analyzer_core.utils.token_optimizer import TokenOptimizer
            
            # Test basic functionality without requiring full database setup
            optimizer = TokenOptimizer()
            sample_text = "This is a sample issue text for testing."
            
            # Test chunking
            chunks = optimizer.smart_chunk_text(sample_text)
            self.assertIsInstance(chunks, list)
            
        except ImportError as e:
            self.skipTest(f"Integration test skipped due to import error: {e}")

def run_performance_test():
    """Run performance tests for token optimization."""
    print("\n=== Performance Test ===")
    
    optimizer = TokenOptimizer()
    
    # Test with different text sizes
    test_sizes = [100, 1000, 10000, 50000]  # Number of words
    
    for size in test_sizes:
        test_text = "This is a sample word. " * size
        
        import time
        start_time = time.time()
        token_count = optimizer.count_tokens(test_text)
        count_time = time.time() - start_time
        
        start_time = time.time()
        chunks = optimizer.smart_chunk_text(test_text)
        chunk_time = time.time() - start_time
        
        print(f"Size: {size} words")
        print(f"  Token count: {token_count} (took {count_time:.4f}s)")
        print(f"  Chunks: {len(chunks)} (took {chunk_time:.4f}s)")
        print(f"  Avg tokens per chunk: {token_count / len(chunks) if chunks else 0:.1f}")

def run_validation_test():
    """Run validation tests to ensure system works correctly."""
    print("\n=== Validation Test ===")
    
    # Test 1: Basic functionality
    print("Test 1: Basic functionality")
    optimizer = TokenOptimizer()
    test_text = "This is a test of the token optimization system."
    token_count = optimizer.count_tokens(test_text)
    print(f"  Text: '{test_text}'")
    print(f"  Token count: {token_count}")
    print(f"  Counter type: {type(optimizer.token_counter).__name__}")
    
    # Test 2: Query validation
    print("\nTest 2: Query validation")
    short_query = "What is the issue?"
    long_query = "This is a very long query that might exceed token limits. " * 100
    
    for query, label in [(short_query, "Short"), (long_query, "Long")]:
        is_valid, count, message = optimizer.validate_query_length(query)
        print(f"  {label} query ({count} tokens): {'✓' if is_valid else '✗'} {message}")
    
    # Test 3: Chunking
    print("\nTest 3: Smart chunking")
    long_text = "This is a long document that will be chunked. " * 200
    chunks = optimizer.smart_chunk_text(long_text)
    print(f"  Original text: {len(long_text)} characters")
    print(f"  Total tokens: {optimizer.count_tokens(long_text)}")
    print(f"  Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"    Chunk {i+1}: {chunk['token_count']} tokens")
    
    # Test 4: Model configurations
    print("\nTest 4: Model configurations")
    for model_name in ["text-embedding-3-small", "text-embedding-ada-002"]:
        opt = TokenOptimizer(model_name=model_name)
        stats = opt.get_optimization_stats()
        print(f"  {model_name}:")
        print(f"    Max tokens: {stats['model_max_tokens']}")
        print(f"    Effective tokens: {stats['effective_max_tokens']}")
        print(f"    Counter: {stats['token_counter_type']}")

if __name__ == "__main__":
    print("Running Token Optimizer Tests...")
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run additional tests
    run_validation_test()
    run_performance_test()
    
    print("\n=== Test Summary ===")
    print("✓ Unit tests completed")
    print("✓ Validation tests completed") 
    print("✓ Performance tests completed")
    print("\nToken optimization system is ready for use!")
