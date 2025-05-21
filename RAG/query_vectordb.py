import os
import json
import argparse
import sys
from typing import Dict, Optional, List, Any
from pathlib import Path

# Add parent directory to sys.path to allow importing from embeddings package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError

# Import the embedding module
from embeddings.azure_openai_embedding import AzureOpenAIEmbedding

def format_search_result(result: Dict, index: int, related_results: List[Dict] = None, full_related: bool = False) -> str:
    """
    Format a single search result for display, including related chunks.
    
    Args:
        result: Primary search result dictionary
        index: Result index
        related_results: List of related result dictionaries
        full_related: Whether to display full content of related chunks
        
    Returns:
        Formatted result string
    """
    doc = result["document"]
    metadata = result["metadata"]
    
    # Format the primary result
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
    
    # Add related chunks if available
    if related_results and len(related_results) > 0:
        formatted += f"\nRelated chunks ({len(related_results)}):\n"
        for i, related in enumerate(related_results):
            formatted += f"  - [{related['metadata']['chunk_type']}] "
            
            # Either show full content or truncated preview based on full_related parameter
            if full_related:
                # Display the full content with proper formatting
                formatted += f"\n    {related['document'].replace('\n', '\n    ')}\n"
            else:
                # Show truncated preview
                related_preview = related['document'][:150] + "..." if len(related['document']) > 150 else related['document']
                formatted += f"{related_preview}\n"
    
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
    openai_api_key: Optional[str] = None,
    use_azure_openai: bool = False,
    azure_openai_endpoint: Optional[str] = None,
    azure_openai_deployment: Optional[str] = None,
    include_all_issue_chunks: bool = False
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
        use_azure_openai: Whether to use Azure OpenAI embeddings
        azure_openai_endpoint: Azure OpenAI endpoint URL
        azure_openai_deployment: Azure OpenAI deployment name
        include_all_issue_chunks: Whether to include all chunks from the same issue
        
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
    if use_azure_openai:
        # Use Azure OpenAI embedding function
        endpoint = azure_openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = azure_openai_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        
        if not endpoint or not deployment:
            print("Error: Azure OpenAI endpoint and deployment are required for Azure OpenAI embeddings.")
            print("Please provide --azure-openai-endpoint and --azure-openai-deployment parameters")
            print("or set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT environment variables.")
            return []
        
        try:
            azure_embedding = AzureOpenAIEmbedding(
                endpoint=endpoint,
                deployment=deployment
            )
            
            # Create a custom embedding function that uses AzureOpenAIEmbedding
            class AzureOpenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
                def __call__(self, texts):
                    return azure_embedding.get_embeddings(texts)
            
            embedding_function = AzureOpenAIEmbeddingFunction()
            print(f"Using Azure OpenAI embeddings with deployment {deployment}")
        except Exception as e:
            print(f"Error initializing Azure OpenAI embeddings: {str(e)}")
            print("Falling back to default embeddings.")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    elif use_openai:
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
    
    # Request more results than needed to account for deduplication
    query_n_results = n_results * 3
    
    # Execute query
    try:
        results = collection.query(
            query_texts=[query],
            n_results=query_n_results,  # Request more results to account for deduplication
            where=where
        )
    except Exception as e:
        print(f"Query error: {str(e)}")
        return []
    
    # If results are empty
    if len(results["documents"][0]) == 0:
        return []

    # Format primary results first
    formatted_results = []
    for i in range(len(results["documents"][0])):
        primary_result = {
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "id": results["ids"][0][i],
            "related_results": []  # Will store related chunks here
        }
        formatted_results.append(primary_result)
    
    # Apply deduplication based on issue number
    original_count = len(formatted_results)
    formatted_results = deduplicate_results(formatted_results)
    if len(formatted_results) < original_count:
        print(f"Deduplication removed {original_count - len(formatted_results)} results from duplicate issues")
    
    # Limit to requested number of results
    formatted_results = formatted_results[:n_results]
        
    # Handle related chunks if requested
    if include_related and formatted_results:
        # Process each primary result to find related chunks
        for result_idx, result in enumerate(formatted_results):
            metadata = result["metadata"]
            related_ids = []
            
            # Parse related chunks from JSON string
            related_chunks = json.loads(metadata.get("related_chunks", "[]"))
            for rel in related_chunks:
                related_ids.append(rel["id"])
            
            # Also include previous and next chunks
            if metadata.get("previous_chunk_id"):
                related_ids.append(metadata["previous_chunk_id"])
            if metadata.get("next_chunk_id"):
                related_ids.append(metadata["next_chunk_id"])
            
            # If include_all_issue_chunks is True, find all chunks from the same issue
            if include_all_issue_chunks:
                issue_number = metadata.get("issue_number")
                if issue_number is not None:
                    print(f"Retrieving all chunks for issue #{issue_number}")
                    # Query for all chunks of this issue
                    try:
                        issue_chunks = collection.get(
                            where={"issue_number": issue_number}
                        )
                        
                        # Add all IDs to the related IDs list (except this chunk's ID)
                        if "ids" in issue_chunks and issue_chunks["ids"]:
                            current_id = result["id"]
                            for chunk_id in issue_chunks["ids"]:
                                if chunk_id != current_id:
                                    related_ids.append(chunk_id)
                            print(f"Found {len(issue_chunks['ids'])-1} additional chunks for issue #{issue_number}")
                    except Exception as e:
                        print(f"Error retrieving all chunks for issue #{issue_number}: {str(e)}")
                
            # Remove duplicates
            related_ids = list(set(related_ids))
            
            # If we have related IDs, get them from the collection
            if related_ids:
                try:
                    related_results = collection.get(
                        ids=related_ids
                    )
                    
                    # Create related result objects and attach to parent result
                    for j in range(len(related_results["documents"])):
                        related_result = {
                            "document": related_results["documents"][j],
                            "metadata": related_results["metadatas"][j],
                            "id": related_results["ids"][j]
                        }
                        formatted_results[result_idx]["related_results"].append(related_result)
                except Exception as e:
                    print(f"Error retrieving related chunks: {str(e)}")
    
    return formatted_results

def deduplicate_results(results):
    """
    Deduplicate results based on issue number to ensure diversity.
    Each result will be from a different issue.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of deduplicated results
    """
    if not results:
        return []
    
    deduplicated = []
    seen_issue_numbers = set()
    
    for result in results:
        issue_number = result["metadata"]["issue_number"]
        
        # Skip if we've already included this issue number
        if issue_number in seen_issue_numbers:
            continue
        
        # Add to deduplicated results and track the issue number
        deduplicated.append(result)
        seen_issue_numbers.add(issue_number)
    
    return deduplicated

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
    parser.add_argument("--use-azure-openai", action="store_true",
                      help="Use Azure OpenAI embeddings (requires endpoint and deployment)")
    parser.add_argument("--azure-openai-endpoint", type=str,
                      help="Azure OpenAI endpoint URL (optional if set as AZURE_OPENAI_ENDPOINT environment variable)")
    parser.add_argument("--azure-openai-deployment", type=str,
                      help="Azure OpenAI deployment name (optional if set as AZURE_OPENAI_DEPLOYMENT environment variable)")
    parser.add_argument("--issue-number", type=int,
                      help="Filter by issue number")
    parser.add_argument("--state", type=str, choices=["open", "closed"],
                      help="Filter by issue state")
    parser.add_argument("--chunk-type", type=str,
                      help="Filter by chunk type (e.g., error, solution, environment)")
    parser.add_argument("--no-related", action="store_true",
                      help="Don't include related chunks in results")
    parser.add_argument("--full-related", action="store_true",
                      help="Output full content of related chunks instead of truncated previews")
    parser.add_argument("--include-all-issue-chunks", action="store_true",
                      help="Include all chunks from the same issue in results")
    
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
        openai_api_key=args.openai_api_key,
        use_azure_openai=args.use_azure_openai,
        azure_openai_endpoint=args.azure_openai_endpoint,
        azure_openai_deployment=args.azure_openai_deployment,
        include_all_issue_chunks=args.include_all_issue_chunks
    )
    
    # Display results
    if not results:
        print(f"\nNo results found matching query: '{args.query}'")
        return
    
    # Calculate total chunks (primary + related)
    total_chunks = sum(1 + len(result.get("related_results", [])) for result in results)
    primary_count = len(results)
    
    print(f"\nFound {primary_count} primary results for query: '{args.query}' (total chunks: {total_chunks})")
    print("=" * 80)
    
    for i, result in enumerate(results):
        # Format result with its related chunks integrated
        if args.full_related:
            # Output full content of related chunks
            print(format_search_result(result, i, result.get("related_results", []), full_related=True))
        else:
            # Output truncated previews of related chunks
            print(format_search_result(result, i, result.get("related_results", []), full_related=False))
        print("=" * 80)

if __name__ == "__main__":
    main()