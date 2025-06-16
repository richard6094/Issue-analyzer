#!/usr/bin/env python3
"""
Debug RAG paths to understand why different scripts see different database paths
"""

import os
import sys

print("=== Debug RAG Database Paths ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print(f"Python path: {sys.path}")

# Add parent directory to sys.path (like both scripts do)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

print(f"Added parent_dir to sys.path: {parent_dir}")

# Test 1: Import the way analyze_issue.py does it
print("\n=== TEST 1: analyze_issue.py approach ===")
try:
    from RAG.rag_helper import query_vectordb_for_regression, DEFAULT_DB_PATH
    print(f"SUCCESS: Imported query_vectordb_for_regression")
    print(f"DEFAULT_DB_PATH: {DEFAULT_DB_PATH}")
    print(f"DEFAULT_DB_PATH exists: {os.path.exists(DEFAULT_DB_PATH)}")
    
    # Test the function call
    print("Testing query_vectordb_for_regression...")
    result = query_vectordb_for_regression(
        issue_title="Test issue",
        issue_body="Test body",
        n_results=1
    )
    print(f"Query result type: {type(result)}")
    print(f"Query result length: {len(result) if result else 0}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Import the way intelligent_dispatch_action.py does it
print("\n=== TEST 2: intelligent_dispatch_action.py approach ===")
try:
    # Clear any cached imports to simulate fresh import
    import importlib
    if 'RAG.rag_helper' in sys.modules:
        importlib.reload(sys.modules['RAG.rag_helper'])
    
    from RAG.rag_helper import default_rag_helper
    print(f"SUCCESS: Imported default_rag_helper")
    print(f"default_rag_helper.db_path: {default_rag_helper.db_path}")
    print(f"default_rag_helper.db_path exists: {os.path.exists(default_rag_helper.db_path)}")
    
    # Test the method call
    print("Testing default_rag_helper.query_for_regression_analysis...")
    results, context = default_rag_helper.query_for_regression_analysis(
        issue_title="Test issue",
        issue_body="Test body",
        n_results=1
    )
    print(f"Query results type: {type(results)}")
    print(f"Query results length: {len(results) if results else 0}")
    print(f"Query context type: {type(context)}")
    print(f"Query context length: {len(context) if context else 0}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check database files
print("\n=== TEST 3: Database file checks ===")
possible_db_paths = [
    os.path.join(parent_dir, "chroma_db"),
    os.path.join(parent_dir, "local_chroma_db_LLM"),
    os.path.join(os.getcwd(), "chroma_db"),
    os.path.join(os.getcwd(), "local_chroma_db_LLM")
]

for db_path in possible_db_paths:
    exists = os.path.exists(db_path)
    print(f"Path: {db_path}")
    print(f"  Exists: {exists}")
    if exists:
        try:
            files = os.listdir(db_path)
            print(f"  Contents: {files}")
            sqlite_file = os.path.join(db_path, "chroma.sqlite3")
            if os.path.exists(sqlite_file):
                size = os.path.getsize(sqlite_file)
                print(f"  chroma.sqlite3 size: {size} bytes")
        except Exception as e:
            print(f"  Error listing contents: {e}")
    print()

print("=== Debug Complete ===")
