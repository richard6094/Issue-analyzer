"""
Vector database helper module providing simplified interfaces for RAG functionality.
"""
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Union

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import query function from query_vectordb
from RAG.query_vectordb import query_database, format_search_result

# Constants for default configuration
DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "github_issues"
DEFAULT_RESULTS_COUNT = 3
DEFAULT_AZURE_ENDPOINT = "https://officegithubcopilotextsubdomain.openai.azure.com/"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_VISION_MODEL = "gpt-4o"

class RagQueryHelper:
    """
    Helper class providing simplified interfaces to query the vector database.
    """
    
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        use_azure_openai: bool = True,
        azure_openai_endpoint: Optional[str] = DEFAULT_AZURE_ENDPOINT,
        azure_openai_deployment: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        azure_vision_deployment: Optional[str] = DEFAULT_VISION_MODEL,
        verbose: bool = False
    ):
        """
        Initialize the RAG query helper with default settings.
        
        Args:
            db_path: Path to ChromaDB directory
            collection_name: Name of the collection to query
            use_azure_openai: Whether to use Azure OpenAI embeddings
            azure_openai_endpoint: Azure OpenAI endpoint URL
            azure_openai_deployment: Azure OpenAI deployment name for embeddings
            azure_vision_deployment: Azure OpenAI deployment name for vision models
            verbose: Whether to print detailed output during query
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.use_azure_openai = use_azure_openai
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_deployment = azure_openai_deployment
        self.azure_vision_deployment = azure_vision_deployment
        self.verbose = verbose
    
    def query(
        self,
        query_text: str,
        n_results: int = DEFAULT_RESULTS_COUNT,
        chunk_type: Optional[str] = "error",  # Default to error chunks
        include_all_issue_chunks: bool = True,
        include_related: bool = True,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database with the given text.
        
        Args:
            query_text: The search query text, can include image URLs/HTML tags
            n_results: Number of results to return
            chunk_type: Type of chunks to filter for (e.g., 'error', 'solution')
            include_all_issue_chunks: Whether to include all chunks from the same issue
            include_related: Whether to include related chunks
            filters: Additional filters to apply
            
        Returns:
            List of result dictionaries
        """
        # Prepare filters
        if filters is None:
            filters = {}
            
        # Add chunk type filter if specified
        if chunk_type:
            filters["chunk_type"] = chunk_type
            
        # Execute the query
        if self.verbose:
            print(f"Querying vector database with: '{query_text}'")
            print(f"Using filters: {filters}")
            
        results = query_database(
            db_path=self.db_path,
            collection_name=self.collection_name,
            query=query_text,
            n_results=n_results,
            filters=filters,
            include_related=include_related,
            use_azure_openai=self.use_azure_openai,
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_deployment=self.azure_openai_deployment,
            azure_vision_deployment=self.azure_vision_deployment,
            include_all_issue_chunks=include_all_issue_chunks
        )
        
        if self.verbose and results:
            print(f"Found {len(results)} primary results")
            total_related = sum(len(result.get("related_results", [])) for result in results)
            print(f"Total related chunks: {total_related}")
            
        return results
    
    def query_for_regression_analysis(
        self, 
        issue_title: str,
        issue_body: str,
        n_results: int = DEFAULT_RESULTS_COUNT
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Specialized query function for regression analysis.
        Searches for similar issues with focus on error patterns and solutions.
        
        Args:
            issue_title: The title of the issue to analyze
            issue_body: The body of the issue to analyze
            n_results: Number of results to return
            
        Returns:
            Tuple containing:
                - List of result dictionaries
                - Context string formatted for LLM prompt
        """
        # Create a query from the title and body
        query_text = f"{issue_title}\n\n{issue_body}"
        
        # Query for error chunks first (most relevant for regression analysis)
        error_results = self.query(
            query_text=query_text,
            n_results=n_results,
            chunk_type="error",
            include_all_issue_chunks=True
        )
        
        # Also query for solution chunks that might be relevant
        solution_results = self.query(
            query_text=query_text,
            n_results=n_results // 2,  # Fewer solution results
            chunk_type="solution",
            include_all_issue_chunks=True
        )
        
        # Combine results
        all_results = error_results + solution_results
        
        # Format results into a context string for LLM prompt
        context = self._format_results_for_prompt(all_results)
        
        return all_results, context
    
    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """
        Format results into a context string suitable for inclusion in an LLM prompt.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No similar issues found in the database."
            
        context_parts = ["### SIMILAR ISSUES FROM KNOWLEDGE BASE:"]
        
        for i, result in enumerate(results):
            # Get basic metadata
            metadata = result["metadata"]
            issue_number = metadata.get("issue_number", "Unknown")
            issue_title = metadata.get("issue_title", "Untitled")
            chunk_type = metadata.get("chunk_type", "Unknown")
            
            # Add issue header
            context_parts.append(f"\n## ISSUE #{issue_number}: {issue_title}")
            context_parts.append(f"TYPE: {chunk_type}")
            
            # Add main content
            doc = result["document"]
            context_parts.append(f"\n{doc}")
            
            # Add selected related chunks (focus on errors and solutions)
            related_results = result.get("related_results", [])
            if related_results:
                # Filter for most useful related chunks
                useful_related = [
                    r for r in related_results 
                    if r["metadata"].get("chunk_type") in ["error", "solution", "reproduction_steps"]
                ]
                
                if useful_related:
                    context_parts.append("\nRELATED INFORMATION:")
                    
                    for rel in useful_related[:3]:  # Limit to 3 related chunks
                        rel_type = rel["metadata"].get("chunk_type", "information")
                        context_parts.append(f"\n- {rel_type.upper()}: {rel['document'][:300]}")
            
            # Add separator between issues
            context_parts.append("\n" + "-" * 40)
        
        return "\n".join(context_parts)

# Create a default instance for easy import
default_rag_helper = RagQueryHelper(
    db_path=DEFAULT_DB_PATH,
    collection_name=DEFAULT_COLLECTION,
    use_azure_openai=True,
    azure_openai_endpoint=DEFAULT_AZURE_ENDPOINT,
    azure_openai_deployment=DEFAULT_EMBEDDING_MODEL,
    azure_vision_deployment=DEFAULT_VISION_MODEL
)

def query_vectordb_for_regression(
    issue_title: str,
    issue_body: str,
    n_results: int = DEFAULT_RESULTS_COUNT
) -> str:
    """
    Simple function to query the vector database for regression analysis.
    
    Args:
        issue_title: The title of the issue to analyze
        issue_body: The body of the issue to analyze
        n_results: Number of results to return
        
    Returns:
        Context string formatted for LLM prompt
    """
    _, context = default_rag_helper.query_for_regression_analysis(
        issue_title=issue_title,
        issue_body=issue_body,
        n_results=n_results
    )
    
    return context