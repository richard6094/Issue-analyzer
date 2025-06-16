#!/usr/bin/env python3
"""
Debug script to check RAG database paths in different contexts
"""
import os
import sys

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

print("=" * 60)
print("DEBUG: RAG Database Path Resolution")
print("=" * 60)

print(f"Current script: {__file__}")
print(f"Current working directory: {os.getcwd()}")
print(f"Current script parent_dir: {parent_dir}")

# Test import and path resolution
try:
    from RAG.rag_helper import DEFAULT_DB_PATH, default_rag_helper
    print(f"SUCCESS: RAG module imported")
    print(f"DEFAULT_DB_PATH: {DEFAULT_DB_PATH}")
    print(f"Database exists: {os.path.exists(DEFAULT_DB_PATH)}")
    
    # Check both possible database locations
    chroma_db_path = os.path.join(parent_dir, "chroma_db")
    local_chroma_db_path = os.path.join(parent_dir, "local_chroma_db_LLM")
    
    print(f"\nPossible database locations:")
    print(f"chroma_db: {chroma_db_path} (exists: {os.path.exists(chroma_db_path)})")
    print(f"local_chroma_db_LLM: {local_chroma_db_path} (exists: {os.path.exists(local_chroma_db_path)})")
    
    # Test the actual function call
    try:
        from RAG.rag_helper import query_vectordb_for_regression
        print(f"\nTesting query_vectordb_for_regression...")
        result = query_vectordb_for_regression("test issue", "test body", n_results=1)
        print(f"Query result length: {len(result) if result else 0}")
        print(f"Query successful: {result != 'No similar issues found in the database.'}")
    except Exception as e:
        print(f"ERROR in query: {str(e)}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"ERROR: Failed to import RAG module: {str(e)}")
    import traceback
    traceback.print_exc()

print("=" * 60)
