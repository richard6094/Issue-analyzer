import os
import json
import argparse
import sys
from typing import Dict, Optional, List, Any
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError  # Add proper error import

def format_search_result(result: Dict, index: int) -> str:
    """
    Format a single search result for display.
    
    Args:
        result: Search result dictionary
        index: Result index
        
    Returns:
        Formatted result string
    """
    doc = result["document"]
    metadata = result["metadata"]
    
    # Format the result
    formatted = f"--- Result {index+1} ---\n"
    formatted += f"Issue: #{metadata['issue_number']} - {metadata['issue_title']}\n"
    formatted += f"Type: {metadata['chunk_type']}\n"
    formatted += f"State: {metadata['state']}\n"
    
    # Add labels if available
    if 'labels' in metadata and metadata['labels']:
        formatted += f"Labels: {metadata['labels']}\n"
    
    # Add author
    if 'author' in metadata:
        formatted += f"Author: {metadata['author']}\n"
    
    # Add text snippet (first 300 chars)
    text_preview = doc[:300] + "..." if len(doc) > 300 else doc
    formatted += f"\nContent: {text_preview}\n"
    
    return formatted

def verify_database_exists(db_path: str) -> bool:
    """
    Verify if the database directory exists
    
    Args:
        db_path: Database path
        
    Returns:
        Boolean indicating if database exists
    """
    path = Path(db_path)
    return path.exists() and path.is_dir()

def list_available_collections(db_path: str) -> List[str]:
    """
    List available collections in the database
    
    Args:
        db_path: Database path
        
    Returns:
        List of collection names
    """
    if not verify_database_exists(db_path):
        return []
    
    client = chromadb.PersistentClient(path=db_path)
    try:
        return client.list_collections()
    except Exception:
        return []

def query_database(
    db_path: str,
    collection_name: str,
    query: str,
    n_results: int = 5,
    filters: Optional[Dict] = None,
    include_related: bool = True,
    use_openai: bool = False,
    openai_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query the vector database.
    
    Args:
        db_path: Path to ChromaDB directory
        collection_name: Name of the collection to query
        query: The search query
        n_results: Number of results to return
        filters: Optional filters to apply
        include_related: Whether to include related chunks
        use_openai: Whether to use OpenAI embeddings
        openai_api_key: OpenAI API key
        
    Returns:
        List of result dictionaries
    """
    # First check if database directory exists
    if not verify_database_exists(db_path):
        print(f"Error: Database directory '{db_path}' does not exist.")
        print(f"Please create the vector database first using create_issue_vectordb.py script:")
        print(f"python RAG/create_issue_vectordb.py --input Issue-migration/issues_summaries.json --db-path {db_path}")
        return []
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Choose embedding function
    if use_openai:
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key is required for OpenAI embeddings.")
            print("Please provide --openai-api-key parameter or set OPENAI_API_KEY environment variable.")
            return []
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
    else:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    
    # Get collection (enhanced error handling)
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except (ValueError, NotFoundError):
        # List available collections
        available_collections = list_available_collections(db_path)
        
        print(f"Error: Collection '{collection_name}' does not exist.")
        if available_collections:
            print(f"Available collections: {', '.join([c.name for c in available_collections])}")
        else:
            print("No collections available in the database.")
        
        print("\nPlease create the vector database first:")
        print(f"python RAG/create_issue_vectordb.py --input Issue-migration/issues_summaries.json --db-path {db_path} --collection {collection_name}")
        return []
    
    # Apply filters if provided
    where = filters if filters else {}
    
    # Execute query
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
    except Exception as e:
        print(f"Query error: {str(e)}")
        return []
    
    # If results are empty
    if len(results["documents"][0]) == 0:
        return []
    
    # Handle related chunks if requested
    if include_related and results["ids"][0]:
        # Extract related chunk IDs
        related_ids = []
        
        for i, metadata in enumerate(results["metadatas"][0]):
            # Parse related chunks from JSON string
            related_chunks = json.loads(metadata.get("related_chunks", "[]"))
            
            for rel in related_chunks:
                related_ids.append(rel["id"])
            
            # Also include previous and next chunks
            if metadata.get("previous_chunk_id"):
                related_ids.append(metadata["previous_chunk_id"])
            if metadata.get("next_chunk_id"):
                related_ids.append(metadata["next_chunk_id"])
        
        # Remove duplicates and already included results
        related_ids = [rid for rid in list(set(related_ids)) if rid not in results["ids"][0]]
        
        # If we have related IDs, get them from the collection
        if related_ids:
            try:
                related_results = collection.get(
                    ids=related_ids
                )
                
                # Format the related results
                for key in ["documents", "metadatas", "ids"]:
                    if key in results and key in related_results:
                        results[key][0].extend(related_results[key])
            except Exception as e:
                print(f"Error retrieving related chunks: {str(e)}")
    
    # Format results for return
    formatted_results = []
    for i in range(len(results["documents"][0])):
        formatted_results.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "id": results["ids"][0][i]
        })
    
    return formatted_results

def main():
    """Main function for the query script"""
    parser = argparse.ArgumentParser(description="Query the GitHub Issues vector database")
    parser.add_argument("--query", type=str, required=True,
                      help="The search query")
    parser.add_argument("--db-path", type=str, default="./chroma_db",
                      help="Path to ChromaDB directory")
    parser.add_argument("--collection", type=str, default="github_issues",
                      help="Name of the collection to query")
    parser.add_argument("--results", type=int, default=5,
                      help="Number of results to return")
    parser.add_argument("--use-openai", action="store_true",
                      help="Use OpenAI embeddings (requires API key)")
    parser.add_argument("--openai-api-key", type=str,
                      help="OpenAI API key (optional if set as environment variable)")
    parser.add_argument("--issue-number", type=int,
                      help="Filter by issue number")
    parser.add_argument("--state", type=str, choices=["open", "closed"],
                      help="Filter by issue state")
    parser.add_argument("--chunk-type", type=str,
                      help="Filter by chunk type (e.g., error, solution, environment)")
    parser.add_argument("--no-related", action="store_true",
                      help="Don't include related chunks in results")
    
    args = parser.parse_args()
    
    # Build filters
    filters = {}
    if args.issue_number:
        filters["issue_number"] = args.issue_number
    if args.state:
        filters["state"] = args.state
    if args.chunk_type:
        filters["chunk_type"] = args.chunk_type
    
    # Execute query
    results = query_database(
        db_path=args.db_path,
        collection_name=args.collection,
        query=args.query,
        n_results=args.results,
        filters=filters,
        include_related=not args.no_related,
        use_openai=args.use_openai,
        openai_api_key=args.openai_api_key
    )
    
    # Display results
    if not results:
        print(f"\nNo results found matching query: '{args.query}'")
        return
    
    print(f"\nFound {len(results)} results for query: '{args.query}'")
    print("=" * 80)
    
    for i, result in enumerate(results):
        print(format_search_result(result, i))
        print("=" * 80)

if __name__ == "__main__":
    main()