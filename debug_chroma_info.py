#!/usr/bin/env python3
"""
Debug ChromaDB collections and version info
"""

import os
import sys

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def check_chromadb_info():
    """Check ChromaDB version and available collections"""
    print("=== ChromaDB Information ===")
    
    try:
        import chromadb
        print(f"ChromaDB version: {chromadb.__version__}")
    except Exception as e:
        print(f"Error importing chromadb: {e}")
        return
    
    # Test database connection
    try:
        from RAG.rag_helper import DEFAULT_DB_PATH
        print(f"Database path: {DEFAULT_DB_PATH}")
        print(f"Database exists: {os.path.exists(DEFAULT_DB_PATH)}")
        
        if os.path.exists(DEFAULT_DB_PATH):
            # Try to connect to database
            client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
            print("Successfully connected to ChromaDB client")
            
            # List collections
            collections = client.list_collections()
            print(f"Available collections: {len(collections)}")
            
            for collection in collections:
                print(f"  - Collection: {collection.name}")
                print(f"    ID: {collection.id}")
                try:
                    count = collection.count()
                    print(f"    Count: {count}")
                except Exception as count_error:
                    print(f"    Count error: {count_error}")
                
                # Try to get metadata
                try:
                    metadata = collection.metadata
                    print(f"    Metadata: {metadata}")
                except Exception as meta_error:
                    print(f"    Metadata error: {meta_error}")
                print()
                
        else:
            print("Database directory does not exist")
            
    except Exception as e:
        print(f"Error accessing database: {e}")
        import traceback
        traceback.print_exc()

def check_azure_openai_info():
    """Check Azure OpenAI embedding configuration"""
    print("=== Azure OpenAI Configuration ===")
    
    try:
        from embeddings.azure_openai_embedding import AzureOpenAIEmbedding
        
        # Try to create embedding instance
        embedding = AzureOpenAIEmbedding(
            deployment_name="text-embedding-3-small",
            api_version="2024-07-01-preview"
        )
        print("Successfully created AzureOpenAIEmbedding instance")
        
        # Test embedding generation
        test_text = "This is a test"
        try:
            embeddings = embedding.embed_documents([test_text])
            print(f"Embedding generation works, dimension: {len(embeddings[0]) if embeddings else 'Unknown'}")
        except Exception as embed_error:
            print(f"Embedding generation error: {embed_error}")
            
    except Exception as e:
        print(f"Error with Azure OpenAI embedding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chromadb_info()
    print()
    check_azure_openai_info()
