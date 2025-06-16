#!/usr/bin/env python3
"""
Simple test to check RAG database access
"""

import os
import sys

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

print(f"Parent dir: {parent_dir}")

# Check database path
from RAG.rag_helper import DEFAULT_DB_PATH
print(f"DEFAULT_DB_PATH: {DEFAULT_DB_PATH}")
print(f"Database exists: {os.path.exists(DEFAULT_DB_PATH)}")

if os.path.exists(DEFAULT_DB_PATH):
    sqlite_file = os.path.join(DEFAULT_DB_PATH, "chroma.sqlite3")
    print(f"SQLite file exists: {os.path.exists(sqlite_file)}")
    if os.path.exists(sqlite_file):
        print(f"SQLite file size: {os.path.getsize(sqlite_file)} bytes")

# Test the function that works
try:
    from RAG.rag_helper import query_vectordb_for_regression
    print("Successfully imported query_vectordb_for_regression")
    
    result = query_vectordb_for_regression("test", "test", 1)
    print(f"Function result: {type(result)}, length: {len(result) if result else 0}")
    print(f"First 100 chars: {result[:100] if result else 'None'}")
    
except Exception as e:
    print(f"Error calling function: {e}")
    import traceback
    traceback.print_exc()
