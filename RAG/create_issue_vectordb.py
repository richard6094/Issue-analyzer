import os
import json
import uuid
import argparse
import threading
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError
from tqdm import tqdm
import numpy as np

# Add imports for Azure OpenAI integration
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from openai import OpenAI

# Import embedding providers
from embeddings.embedding_provider import EmbeddingProvider
from embeddings.openai_embedding import OpenAIEmbedding
from embeddings.azure_openai_embedding import AzureOpenAIEmbedding

# Import image recognition providers
from image_recognition.image_recognition_provider import (
    extract_image_urls,
    process_issue_images,
    analyze_issue_with_images,
    get_image_recognition_model,
    is_valid_image_url
)

class ProgressTracker:
    """
    Track and display real-time progress during database creation.
    """
    def __init__(self, update_interval=5):
        """
        Initialize the progress tracker.
        
        Args:
            update_interval: How often to update the display (in seconds)
        """
        self.start_time = time.time()
        self.update_interval = update_interval
        self.stop_event = threading.Event()
        self.status = {
            "total_issues": 0,
            "processed_issues": 0,
            "total_chunks": 0,
            "success_chunks": 0,
            "error_chunks": 0,
            "current_stage": "Initializing",
            "current_issue": None,
            "eta": None,
            "chunk_types": {},
            "stage_times": {
                "loading": 0,
                "chunking": 0,
                "embedding": 0,
                "storing": 0
            }
        }
        self.thread = None
        # Flag to control initial display
        self.first_update = True
        # Flag to track if terminal supports ANSI escape codes
        self.supports_ansi = True
    
    def start(self):
        """Start the progress tracking thread."""
        if self.thread is not None:
            return
        
        # Detect terminal capabilities
        self.supports_ansi = os.name == 'posix' or 'ANSICON' in os.environ or 'WT_SESSION' in os.environ
        
        # Add initial lines only once
        if self.supports_ansi:
            print("\n\n\n\n\n\n")
            
        self.thread = threading.Thread(target=self._update_display)
        self.thread.daemon = True
        self.thread.start()
        print("Progress tracking started")
    
    def stop(self):
        """Stop the progress tracking thread."""
        if self.thread is None:
            return
            
        self.stop_event.set()
        self.thread.join()
        self.thread = None
        # Final display
        self._display_progress()
        print("\nProgress tracking stopped")
    
    def update_stage(self, stage_name, issue_number=None):
        """Update the current processing stage."""
        self.status["current_stage"] = stage_name
        if issue_number is not None:
            self.status["current_issue"] = issue_number
    
    def set_total_issues(self, count):
        """Set the total number of issues to process."""
        self.status["total_issues"] = count
    
    def increment_processed_issues(self):
        """Increment the count of processed issues."""
        self.status["processed_issues"] += 1
        # Update ETA based on processed issues
        if self.status["processed_issues"] > 0 and self.status["total_issues"] > 0:
            elapsed = time.time() - self.start_time
            issues_left = self.status["total_issues"] - self.status["processed_issues"]
            time_per_issue = elapsed / self.status["processed_issues"]
            eta_seconds = issues_left * time_per_issue
            self.status["eta"] = eta_seconds
    
    def add_chunks(self, count, chunk_types=None):
        """
        Add to the total chunk count.
        
        Args:
            count: Number of chunks to add
            chunk_types: Dictionary of chunk types and their counts
        """
        self.status["total_chunks"] += count
        
        # Update chunk type distribution
        if chunk_types:
            for chunk_type, type_count in chunk_types.items():
                if chunk_type in self.status["chunk_types"]:
                    self.status["chunk_types"][chunk_type] += type_count
                else:
                    self.status["chunk_types"][chunk_type] = type_count
    
    def increment_success_chunks(self, count=1):
        """Increment the count of successfully processed chunks."""
        self.status["success_chunks"] += count
    
    def increment_error_chunks(self, count=1):
        """Increment the count of chunks with errors."""
        self.status["error_chunks"] += count
    
    def update_stage_time(self, stage, elapsed):
        """Update the time spent in a processing stage."""
        if stage in self.status["stage_times"]:
            self.status["stage_times"][stage] += elapsed
    
    def _format_time(self, seconds):
        """Format seconds into a readable time string."""
        if seconds is None:
            return "Unknown"
            
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _display_progress(self):
        """Display the current progress information."""
        elapsed = time.time() - self.start_time
        
        # Choose display method based on terminal capabilities
        if self.supports_ansi:
            # Clear previous lines (6 lines for the header, status, progress info, and spacing)
            print("\033[K\033[F" * 6, end="")
            
            # Print header
            print("\033[K\033[1m===== Real-time Progress Report =====\033[0m")
            
            # Print current status
            issue_info = f" (Issue #{self.status['current_issue']})" if self.status["current_issue"] else ""
            print(f"\033[KStage: {self.status['current_stage']}{issue_info}")
            
            # Print issue progress
            if self.status["total_issues"] > 0:
                issue_percent = self.status["processed_issues"] / self.status["total_issues"] * 100
                print(f"\033[KIssues: {self.status['processed_issues']} / {self.status['total_issues']} ({issue_percent:.1f}%)")
            else:
                print(f"\033[KIssues: {self.status['processed_issues']} processed")
            
            # Print chunk progress
            if self.status["total_chunks"] > 0:
                success_percent = self.status["success_chunks"] / self.status["total_chunks"] * 100
                print(f"\033[KChunks: {self.status['success_chunks']} / {self.status['total_chunks']} ({success_percent:.1f}%) - {self.status['error_chunks']} errors")
            else:
                print(f"\033[KChunks: {self.status['success_chunks']} processed - {self.status['error_chunks']} errors")
            
            # Print timing information
            eta_str = self._format_time(self.status["eta"]) if self.status["eta"] else "Calculating..."
            print(f"\033[KTime: Elapsed {self._format_time(elapsed)} - ETA: {eta_str}")
            
            # Add an empty line for better visibility
            print("")  # This empty line will be cleared on next update
        else:
            # For terminals that don't support ANSI escape codes, just print a new update
            if self.first_update or (time.time() % 10) < 1:  # Limit full updates to every ~10 seconds
                print("\n===== Real-time Progress Report =====")
                issue_info = f" (Issue #{self.status['current_issue']})" if self.status["current_issue"] else ""
                print(f"Stage: {self.status['current_stage']}{issue_info}")
                
                if self.status["total_issues"] > 0:
                    issue_percent = self.status["processed_issues"] / self.status["total_issues"] * 100
                    print(f"Issues: {self.status['processed_issues']} / {self.status['total_issues']} ({issue_percent:.1f}%)")
                else:
                    print(f"Issues: {self.status['processed_issues']} processed")
                
                if self.status["total_chunks"] > 0:
                    success_percent = self.status["success_chunks"] / self.status["total_chunks"] * 100
                    print(f"Chunks: {self.status['success_chunks']} / {self.status['total_chunks']} ({success_percent:.1f}%) - {self.status['error_chunks']} errors")
                else:
                    print(f"Chunks: {self.status['success_chunks']} processed - {self.status['error_chunks']} errors")
                
                eta_str = self._format_time(self.status["eta"]) if self.status["eta"] else "Calculating..."
                print(f"Time: Elapsed {self._format_time(elapsed)} - ETA: {eta_str}\n")
                
        self.first_update = False
    
    def _update_display(self):
        """Background thread to update the display periodically."""
        # Don't add initial lines here, already added in start()
        while not self.stop_event.is_set():
            self._display_progress()
            time.sleep(self.update_interval)

class AzureChatOpenAIError(Exception):
    """Exception raised for errors in Azure OpenAI chat operations."""
    pass

def get_azure_ad_token():
    """Returns a function that gets an Azure AD token."""
    credential = DefaultAzureCredential()
    # This returns the token when called by the LangChain internals
    return lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token

def get_azure_chat_model(model_id="gpt-4o"):
    """Get a configured Azure OpenAI chat model through LangChain."""
    try:
        # Create Azure OpenAI chat model with Azure AD authentication
        chat_model = AzureChatOpenAI(
            deployment_name=model_id,
            api_version="2025-01-01-preview",
            azure_endpoint="https://officegithubcopilotextsubdomain.openai.azure.com/",
            azure_ad_token_provider=get_azure_ad_token(),
            temperature=0.2,  # Low temperature for more deterministic outputs
            model_kwargs={"response_format": {"type": "json_object"}}  # Ensure JSON output
        )
        return chat_model
    except Exception as e:
        raise AzureChatOpenAIError(str(e)) from e

class IssueVectorDatabase:
    """
    A class for creating and managing a vector database of GitHub issues using ChromaDB
    with intelligent semantic chunking and relationship preservation.
    """
    
    def __init__(self, 
                 issues_json_path: str, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "github_issues",
                 openai_api_key: Optional[str] = None,
                 use_openai_embeddings: bool = True,
                 use_azure_openai: bool = False,
                 azure_endpoint: Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 azure_vision_deployment: Optional[str] = "gpt-4o", 
                 chunk_overlap: int = 100,
                 max_tokens_per_chunk: int = 1000,
                 use_llm_chunking: bool = False):  # Changed default to False
        """
        Initialize the Issue Vector Database.
        
        Args:
            issues_json_path: Path to the issues JSON file
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection in ChromaDB
            openai_api_key: OpenAI API key (optional if using local embeddings)
            use_openai_embeddings: Whether to use OpenAI embeddings or SentenceTransformers
            use_azure_openai: Whether to use Azure OpenAI embeddings
            azure_endpoint: Azure OpenAI API endpoint (required if use_azure_openai is True)
            azure_deployment: Azure OpenAI deployment name for embeddings (required if use_azure_openai is True)
            azure_vision_deployment: Azure OpenAI deployment name for vision/chat models (required for image analysis)
            chunk_overlap: Number of characters to overlap between chunks
            max_tokens_per_chunk: Maximum tokens per chunk
            use_llm_chunking: Whether to use LLM-based chunking (False by default, uses pattern-based chunking)
        """
        self.issues_json_path = issues_json_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.use_openai_embeddings = use_openai_embeddings
        self.use_azure_openai = use_azure_openai
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "text-embedding-3-small"
        self.azure_vision_deployment = azure_vision_deployment or os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT") or "gpt-4o"
        self.chunk_overlap = chunk_overlap
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.use_llm_chunking = use_llm_chunking
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Choose embedding function based on configuration
        if self.use_azure_openai:
            print(f"Using Azure OpenAI embeddings with deployment {self.azure_deployment}")
            if not self.azure_endpoint:
                raise ValueError("Azure OpenAI endpoint is required for Azure OpenAI embeddings")
            
            # Create Azure OpenAI embedding provider
            embedding_provider = AzureOpenAIEmbedding(
                endpoint=self.azure_endpoint,
                deployment=self.azure_deployment
            )
            
            # Create a custom embedding function for ChromaDB that uses our Azure provider
            self.embedding_function = self._create_azure_embedding_function(embedding_provider)
        elif use_openai_embeddings:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-3-small"
            )
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            # Use local embeddings with SentenceTransformers
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
            self.collection._embedding_function = self.embedding_function
            print(f"Found existing collection '{collection_name}' with {self.collection.count()} documents")
            
        except (ValueError, NotFoundError):
            print(f"Collection '{collection_name}' not found. Creating new collection...")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "GitHub Issues semantic chunks with relationships"}
            )
            print(f"Created new collection '{collection_name}'")
    
    def _create_azure_embedding_function(self, embedding_provider: EmbeddingProvider):
        """
        Create a custom embedding function for ChromaDB that uses Azure OpenAI
        
        Args:
            embedding_provider: The Azure OpenAI embedding provider
            
        Returns:
            A function that can be used by ChromaDB for embeddings
        """
        def _azure_embeddings_function(texts=None, **kwargs):
            """
            Function that processes batches of texts and returns embeddings using Azure OpenAI
            
            Args:
                texts: List of texts to embed (legacy parameter)
                **kwargs: Keyword arguments including 'input' that may be passed by ChromaDB
                
            Returns:
                List of embeddings as numpy arrays
            """
            # Handle both old-style 'texts' parameter and new-style 'input' parameter
            input_texts = kwargs.get('input', texts)
            if input_texts is None:
                raise ValueError("No text provided for embedding")
                
            embeddings = []
            for text in input_texts:
                # Get embedding from Azure OpenAI
                embedding = embedding_provider.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
            
        return _azure_embeddings_function
    
    def load_issues(self) -> List[Dict[str, Any]]:
        """
        Load issues from the JSON file.
        
        Returns:
            List of issue dictionaries
        """
        with open(self.issues_json_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        print(f"Loaded {len(issues)} issues from {self.issues_json_path}")
        return issues
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for estimating token count.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # This is a simple approximation - not as accurate as a real tokenizer
        return text.split()
    
    def identify_semantic_boundaries(self, text: str) -> List[int]:
        """
        Identify semantic boundaries in the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of indices where semantic boundaries occur
        """
        # Look for semantic boundary indicators
        boundaries = []
        
        # Headers and sections
        section_patterns = [
            "## ", "# ", "\n\n", 
            "Expected behavior", "Current behavior",
            "Steps to reproduce", "Environment:",
            "Your Environment", "Error:", "Error Message:",
            "Console output:"
        ]
        
        for pattern in section_patterns:
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break
                boundaries.append(pos)
                start = pos + 1
        
        # Code block boundaries
        code_markers = ["```", "`"]
        for marker in code_markers:
            start = 0
            while True:
                pos = text.find(marker, start)
                if pos == -1:
                    break
                boundaries.append(pos)
                start = pos + 1
        
        # Comments often represent semantic shifts
        comment_indicators = ["Comment", "commented", "replied"]
        for indicator in comment_indicators:
            start = 0
            while True:
                pos = text.find(indicator, start)
                if pos == -1:
                    break
                boundaries.append(pos)
                start = pos + 1
                
        # Sort boundaries and remove duplicates or very close ones
        boundaries = sorted(list(set(boundaries)))
        filtered_boundaries = [boundaries[0]] if boundaries else []
        
        for i in range(1, len(boundaries)):
            if boundaries[i] - filtered_boundaries[-1] > 50:  # Minimum distance between boundaries
                filtered_boundaries.append(boundaries[i])
                
        return filtered_boundaries
    
    def identify_chunk_type(self, chunk: str) -> str:
        """
        Identify the type of a chunk based on its content.
        
        Args:
            chunk: The text chunk
            
        Returns:
            Chunk type as a string
        """
        # Identify chunk type based on content patterns
        chunk_lower = chunk.lower()
        
        if any(marker in chunk_lower for marker in ["```", "code:", "example:", "function "]):
            return "code"
        elif any(marker in chunk_lower for marker in ["error", "exception", "failed", "crash"]):
            return "error"
        elif any(marker in chunk_lower for marker in ["environment", "platform", "version", "os:"]):
            return "environment"
        elif any(marker in chunk_lower for marker in ["solution", "workaround", "fix:", "solved"]):
            return "solution"
        elif any(marker in chunk_lower for marker in ["expected behavior", "should work", "supposed to"]):
            return "expected_behavior"
        elif any(marker in chunk_lower for marker in ["current behavior", "instead", "problem"]):
            return "current_behavior"
        elif any(marker in chunk_lower for marker in ["reproduce", "steps to", "steps:"]):
            return "reproduction_steps"
        elif any(marker in chunk_lower for marker in ["comment", "replied", "response"]):
            return "comment"
        else:
            return "description"
    
    def semantic_chunk_text(self, text: str, issue_number: int, issue_data: Dict) -> List[Dict]:
        """
        Intelligently chunk text using LLM or pattern-based approach based on configuration.
        Also identifies and processes images in the text.
        
        Args:
            text: Text to chunk
            issue_number: Issue number
            issue_data: Complete issue data
            
        Returns:
            List of semantic chunks with metadata (including image chunks)
        """
        # Check for images in the text
        image_urls = extract_image_urls(text)
        
        # If text is very short, just return as a single chunk
        if len(text) < 500 and not image_urls:
            chunk_type = self.identify_chunk_type(text)
            return [{
                "text": text,
                "metadata": {
                    "issue_number": issue_number,
                    "issue_title": issue_data.get("title", ""),
                    "chunk_type": chunk_type,
                    "position": 0,
                    "state": issue_data.get("state", ""),
                    "author": issue_data.get("author", ""),
                    "labels": ",".join(issue_data.get("labels", [])),
                    "created_at": issue_data.get("created_at", ""),
                    "chunk_id": str(uuid.uuid4())
                }
            }]
        
        # Create chunks list to hold all results
        chunks = []
        
        # Process any images found in the text and add them as separate chunks
        if image_urls:
            try:
                print(f"Processing {len(image_urls)} images in issue #{issue_number}")
                
                # Base metadata for images
                base_metadata = {
                    "issue_number": issue_number,
                    "issue_title": issue_data.get("title", ""),
                    "state": issue_data.get("state", ""),
                    "author": issue_data.get("author", ""),
                    "labels": ",".join(issue_data.get("labels", [])),
                    "created_at": issue_data.get("created_at", "")
                }
                
                # Process images in the issue using the image recognition module
                image_chunks = process_issue_images(
                    issue_number=issue_number,
                    issue_title=issue_data.get("title", ""),
                    issue_body=text,
                    metadata=base_metadata,
                    provider="azure",  # Use Azure OpenAI by default
                    max_context_length=1000,  # Context length around each image
                    endpoint=self.azure_endpoint,
                    deployment=self.azure_vision_deployment, 
                    api_key=self.openai_api_key if not self.use_azure_openai else None  # Only pass API key in non-Azure mode
                )
                
                # Add image chunks to our chunk collection
                if image_chunks:
                    chunks.extend(image_chunks)
                    print(f"Added {len(image_chunks)} image description chunks for issue #{issue_number}")
                
            except Exception as e:
                print(f"Error processing images in issue #{issue_number}: {str(e)}")
        
        # Choose chunking method based on configuration for the text content
        if self.use_llm_chunking:
            segments = self.llm_segment_text(text)
        else:
            segments = self.pattern_based_segment_text(text)
        
        # Create properly formatted chunks with metadata
        for i, segment in enumerate(segments):
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "text": segment["text"],
                "metadata": {
                    "issue_number": issue_number,
                    "issue_title": issue_data.get("title", ""),
                    "chunk_type": segment["type"],
                    "position": i,  # Position within the issue
                    "state": issue_data.get("state", ""),
                    "author": issue_data.get("author", ""),
                    "labels": ",".join(issue_data.get("labels", [])),
                    "created_at": issue_data.get("created_at", ""),
                    "chunk_id": chunk_id
                }
            })
        
        return chunks

    def find_related_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Find and establish relationships between related chunks.
        
        Args:
            chunks: List of chunks with metadata
            
        Returns:
            Updated list of chunks with relationship metadata
        """
        # Group chunks by issue number
        issue_chunks = {}
        for chunk in chunks:
            issue_number = chunk["metadata"]["issue_number"]
            if issue_number not in issue_chunks:
                issue_chunks[issue_number] = []
            issue_chunks[issue_number].append(chunk)
        
        # Process each issue's chunks to find relationships
        for issue_number, issue_chunks_list in issue_chunks.items():
            # Sort chunks by position
            issue_chunks_list.sort(key=lambda x: x["metadata"]["position"])
            
            # Find relationships
            for i, chunk in enumerate(issue_chunks_list):
                # Initialize relationship lists
                chunk["metadata"]["related_chunks"] = []
                chunk["metadata"]["previous_chunk_id"] = ""
                chunk["metadata"]["next_chunk_id"] = ""
                
                # Store previous and next chunk IDs
                if i > 0:
                    chunk["metadata"]["previous_chunk_id"] = issue_chunks_list[i-1]["metadata"]["chunk_id"]
                if i < len(issue_chunks_list) - 1:
                    chunk["metadata"]["next_chunk_id"] = issue_chunks_list[i+1]["metadata"]["chunk_id"]
                
                # Find specific relationship types
                for j, other_chunk in enumerate(issue_chunks_list):
                    if i == j:  # Skip self
                        continue
                    
                    # Check for error-solution relationship
                    if chunk["metadata"]["chunk_type"] == "error" and other_chunk["metadata"]["chunk_type"] == "solution":
                        chunk["metadata"]["related_chunks"].append({
                            "id": other_chunk["metadata"]["chunk_id"],
                            "relation_type": "solution_for"
                        })
                    
                    # Check for problem-expected behavior relationship
                    if chunk["metadata"]["chunk_type"] == "current_behavior" and other_chunk["metadata"]["chunk_type"] == "expected_behavior":
                        chunk["metadata"]["related_chunks"].append({
                            "id": other_chunk["metadata"]["chunk_id"],
                            "relation_type": "expected_behavior"
                        })
                    
                    # Check for problem-reproduction relationship
                    if chunk["metadata"]["chunk_type"] == "current_behavior" and other_chunk["metadata"]["chunk_type"] == "reproduction_steps":
                        chunk["metadata"]["related_chunks"].append({
                            "id": other_chunk["metadata"]["chunk_id"],
                            "relation_type": "reproduction"
                        })
                
                # Convert related_chunks to string to store in ChromaDB metadata (which requires strings)
                chunk["metadata"]["related_chunks"] = json.dumps(chunk["metadata"]["related_chunks"])
        
        # Flatten back to a single list
        return [chunk for chunks_list in issue_chunks.values() for chunk in chunks_list]
    
    def preprocess_issue(self, issue: Dict) -> str:
        """
        Preprocess a single issue to extract its full text content.
        
        Args:
            issue: Dictionary containing issue data
            
        Returns:
            String containing the issue's full text content
        """
        if not issue or not isinstance(issue, dict):
            return ""
            
        # Ensure basic fields exist
        if "number" not in issue or "title" not in issue:
            return ""
            
        # Construct full text representation
        full_text = f"Issue #{issue.get('number')}: {issue.get('title')}\n\n"
        
        # Add issue description
        if "body" in issue and issue["body"]:
            full_text += f"Description:\n{issue['body']}\n\n"
        
        # Add label information
        if "labels" in issue and issue["labels"]:
            if isinstance(issue["labels"], list):
                labels_text = ", ".join(issue["labels"])
                full_text += f"Labels: {labels_text}\n\n"
        
        # Add author and state information
        if "author" in issue:
            full_text += f"Author: {issue['author']}\n"
        if "state" in issue:
            full_text += f"State: {issue['state']}\n"
        if "created_at" in issue:
            full_text += f"Created: {issue['created_at']}\n\n"
        
        # Add comments
        if "comments" in issue and issue["comments"] and isinstance(issue["comments"], list):
            for i, comment in enumerate(issue["comments"]):
                if isinstance(comment, dict) and "body" in comment and "author" in comment:
                    full_text += f"Comment {i+1} by {comment['author']} on {comment.get('created_at', 'unknown date')}:\n"
                    full_text += f"{comment['body']}\n\n"
        
        return full_text
    
    def process_issues(self):
        """
        Process all issues, create semantic chunks, and add them to the vector database.
        """
        # Initialize progress tracker
        progress = ProgressTracker(update_interval=2)
        progress.start()
        progress.update_stage("Loading Issues", None)
        
        # Load issues from file - measure loading time
        start_time = time.time()
        issues = self.load_issues()
        loading_time = time.time() - start_time
        progress.update_stage_time("loading", loading_time)
        
        # Track statistics
        total_chunks = 0
        all_chunks = []
        
        # Set total issues for tracking
        progress.set_total_issues(len(issues))
        
        # Process each issue
        for issue in issues:
            # Get issue number
            issue_number = issue.get("number", 0)
            
            # Update progress to the current issue
            progress.update_stage("Processing Issue", issue_number)
            
            # Skip empty issues
            if not issue:
                progress.increment_processed_issues()
                continue
            
            # Preprocess the issue
            start_time = time.time()
            full_text = self.preprocess_issue(issue)
            if not full_text:
                progress.increment_processed_issues()
                continue
            
            # Break the issue into semantic chunks
            progress.update_stage("Semantic Chunking", issue_number)
            chunks = self.semantic_chunk_text(full_text, issue_number, issue)
            chunking_time = time.time() - start_time
            progress.update_stage_time("chunking", chunking_time)
            
            # Add chunks to the total
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
            
            # Calculate chunk type distribution
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk["metadata"]["chunk_type"]
                if chunk_type in chunk_types:
                    chunk_types[chunk_type] += 1
                else:
                    chunk_types[chunk_type] = 1
            
            # Update progress with chunk information
            progress.add_chunks(len(chunks), chunk_types)
            
            # Update processed count
            progress.increment_processed_issues()
        
        # Find and establish relationships between chunks
        progress.update_stage("Finding Relationships", None)
        start_time = time.time()
        all_chunks = self.find_related_chunks(all_chunks)
        relationship_time = time.time() - start_time
        progress.update_stage_time("embedding", relationship_time)
        
        # Add all chunks to ChromaDB
        progress.update_stage("Storing in Database", None)
        start_time = time.time()
        
        # Prepare data for batch insertion
        ids = []
        documents = []
        metadatas = []
        
        for chunk in all_chunks:
            chunk_id = chunk["metadata"].pop("chunk_id")  # Extract and remove ID from metadata
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            batch_ids = ids[i:batch_end]
            batch_docs = documents[i:batch_end]
            batch_meta = metadatas[i:batch_end]
            
            # Update progress
            progress.update_stage(f"Adding Batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}", None)
            
            try:
                # Try to add the batch
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                progress.increment_success_chunks(len(batch_ids))
            except Exception as e:
                print(f"Error adding batch to database: {str(e)}")
                progress.increment_error_chunks(len(batch_ids))
        
        # Update storage time
        storing_time = time.time() - start_time
        progress.update_stage_time("storing", storing_time)
        
        # Final stage
        progress.update_stage("Complete", None)
        
        # Stop the progress tracker and display final stats
        progress.stop()
        
        print(f"\nSuccessfully added {progress.status['success_chunks']} semantic chunks to ChromaDB")
        print(f"Vector database is now available at {self.db_path}")
        
        # Display errors if any
        if progress.status["error_chunks"] > 0:
            print(f"Warning: {progress.status['error_chunks']} chunks failed to be added to the database")
        
        # Display time breakdown
        print("\nTime Breakdown:")
        total_time = sum(progress.status["stage_times"].values())
        for stage, stage_time in progress.status["stage_times"].items():
            percent = (stage_time / total_time) * 100 if total_time > 0 else 0
            print(f"  {stage.capitalize()}: {self._format_time(stage_time)} ({percent:.1f}%)")
        
    def _format_time(self, seconds):
        """Format seconds into a readable time string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def query(self, 
              query_text: str, 
              n_results: int = 5, 
              filters: Optional[Dict] = None,
              include_related: bool = True) -> Dict:
        """
        Query the vector database for similar chunks.
        
        Args:
            query_text: The text query
            n_results: Number of results to return
            filters: Optional filters to apply
            include_related: Whether to include related chunks in results
            
        Returns:
            Dictionary with query results
        """
        # Perform initial query
        where_clause = filters if filters else {}
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause
        )
        
        # If include_related is True and we got results
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
            
            # Remove duplicates
            related_ids = list(set(related_ids))
            
            # If we have related IDs, get them from the collection
            if related_ids:
                related_results = self.collection.get(
                    ids=related_ids
                )
                
                # Combine results
                for key in results:
                    if key == "ids":
                        results[key][0].extend(related_results.get(key, []))
                    elif key in ["documents", "metadatas"]:
                        results[key][0].extend(related_results.get(key, []))
                    else:
                        # For embeddings or other numerical data
                        if key in related_results:
                            results[key][0] = list(results[key][0]) + list(related_results[key])
        
        return results

    def llm_segment_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to segment text into semantically coherent blocks.
        This is a more advanced approach than the pattern-based chunking.
        
        Args:
            text: The full text to segment
            
        Returns:
            List of dictionaries with segment info including text and type
        """
        # If the text is very short, return it as a single segment
        if len(text) < 500:
            return [{"text": text, "type": self.identify_chunk_type(text)}]
            
        try:
            # Try to get Azure OpenAI model first
            try:
                # Get the Azure OpenAI chat model
                chat_model = get_azure_chat_model(model_id="gpt-4o")
                use_azure = True
                print("Using Azure OpenAI for semantic segmentation")
            except Exception as e:
                print(f"Azure OpenAI initialization failed: {str(e)}")
                # Fall back to regular OpenAI if Azure fails
                use_azure = False
                # Check if OpenAI API key is available
                if not self.openai_api_key:
                    raise ValueError("No OpenAI API key provided and Azure OpenAI initialization failed")
                
            # Prepare the system prompt for text segmentation
            system_prompt = """You are an expert system for analyzing GitHub issues.
Your task is to segment the given text into semantically coherent chunks, where each chunk:
1. Contains complete thoughts or concepts that belong together
2. Respects natural semantic boundaries like sections, code blocks, or topic changes
3. Is meaningful on its own but preserves contextual information
4. Has logical coherence within itself

Consider these boundary points when creating segments:
- Topic or subject changes
- Transitions between problem descriptions and solutions
- Boundaries between code blocks and explanatory text
- Changes in who is speaking (different commenters)
- Section headers or formatting changes

For each segment, also identify its type from these categories:
- description: General issue descriptions
- error: Error messages or descriptions of problems
- code: Code snippets or stack traces
- solution: Proposed fixes or solutions
- environment: Information about system environment/versions
- reproduction_steps: Steps to reproduce an issue
- expected_behavior: Description of expected behavior
- current_behavior: Description of current problematic behavior
- comment: User comments or feedback

Return your response as a JSON object with a 'segments' array, where each element is an object with:
- "text": The segment text
- "type": The segment type from the categories above"""

            # Prepare the complete prompt including the content
            user_prompt = f"Segment the following GitHub issue text into semantically coherent chunks:\n\n{text}"
            
            if use_azure:
                # Use Azure OpenAI via LangChain
                # Create a chat prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{issue_text}")
                ])
                
                # Set up the parser for JSON output
                parser = JsonOutputParser()
                
                # Create the chain
                chain = prompt | chat_model | parser
                
                # Execute the chain
                result = chain.invoke({"issue_text": user_prompt})
                
            else:
                # Fall back to direct OpenAI client if Azure isn't available
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # You can adjust this based on what's available
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2  # Keep it deterministic
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
            
            # Return the segments if properly formatted
            if "segments" in result and isinstance(result["segments"], list):
                # Format validation and cleanup
                valid_segments = []
                for segment in result["segments"]:
                    if "text" in segment and "type" in segment and segment["text"].strip():
                        valid_segments.append({
                            "text": segment["text"].strip(),
                            "type": segment["type"].lower()
                        })
                
                if valid_segments:
                    print(f"LLM successfully segmented text into {len(valid_segments)} chunks")
                    return valid_segments
            
            # Fallback to pattern-based chunking if result is malformed
            print("Warning: LLM returned malformed response for text segmentation. Falling back to pattern-based chunking.")
            return self.pattern_based_segment_text(text)
            
        except Exception as e:
            print(f"Error using LLM for text segmentation: {str(e)}")
            print("Falling back to pattern-based chunking.")
            return self.pattern_based_segment_text(text)
            
    def pattern_based_segment_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment text into semantic chunks using pattern-based approach.
        Preserves image references and their surrounding context.
        
        Args:
            text: The text to segment
            
        Returns:
            List of segment dictionaries with text and type
        """
        # Find semantic boundaries
        boundaries = self.identify_semantic_boundaries(text)
        
        # If no boundaries found, check if text is too long
        if not boundaries:
            if len(self._simple_tokenize(text)) > self.max_tokens_per_chunk:
                # Split by paragraphs
                paragraphs = text.split("\n\n")
                segments = []
                current_segment = ""
                for paragraph in paragraphs:
                    # Check if adding this paragraph would exceed token limit
                    if len(self._simple_tokenize(current_segment + paragraph)) > self.max_tokens_per_chunk:
                        if current_segment:  # Only add if we have content
                            segments.append({
                                "text": current_segment,
                                "type": self.identify_chunk_type(current_segment)
                            })
                            current_segment = paragraph + "\n\n"
                        else:
                            # Paragraph itself is too long, split it further
                            segments.append({
                                "text": paragraph,
                                "type": self.identify_chunk_type(paragraph)
                            })
                    else:
                        current_segment += paragraph + "\n\n"
                
                # Add the last segment if any
                if current_segment:
                    segments.append({
                        "text": current_segment,
                        "type": self.identify_chunk_type(current_segment)
                    })
                
                return segments
            else:
                # Text is small enough to be a single chunk
                return [{
                    "text": text,
                    "type": self.identify_chunk_type(text)
                }]
        
        # Create segments based on semantic boundaries
        segments = []
        start_pos = 0
        
        for boundary in boundaries:
            # Check if we should split at this boundary
            if boundary - start_pos > 50:  # Only create chunks that have meaningful content
                segment_text = text[start_pos:boundary].strip()
                
                # Only add non-empty segments
                if segment_text:
                    # Check for image references in this segment
                    has_image = bool(extract_image_urls(segment_text))
                    
                    # Determine chunk type
                    chunk_type = self.identify_chunk_type(segment_text)
                    
                    # Mark if this is an image chunk
                    if has_image:
                        chunk_type = "image_content" if chunk_type == "description" else f"{chunk_type}_with_image"
                    
                    segments.append({
                        "text": segment_text,
                        "type": chunk_type
                    })
            
            start_pos = boundary
        
        # Add the final segment if needed
        if start_pos < len(text):
            final_segment = text[start_pos:].strip()
            if final_segment:
                # Check for image references in this segment
                has_image = bool(extract_image_urls(final_segment))
                
                # Determine chunk type
                chunk_type = self.identify_chunk_type(final_segment)
                
                # Mark if this is an image chunk
                if has_image:
                    chunk_type = "image_content" if chunk_type == "description" else f"{chunk_type}_with_image"
                
                segments.append({
                    "text": final_segment,
                    "type": chunk_type
                })
        
        # Merge very small segments with their neighbors
        if len(segments) > 1:
            i = 0
            while i < len(segments) - 1:
                current_len = len(self._simple_tokenize(segments[i]["text"]))
                
                # Skip processing segments with images - keep them separate
                if "image" in segments[i]["type"] or "image" in segments[i+1]["type"]:
                    i += 1
                    continue
                
                # If current segment is very small, merge with next
                if current_len < 50:
                    segments[i+1]["text"] = segments[i]["text"] + "\n" + segments[i+1]["text"]
                    segments.pop(i)
                else:
                    i += 1
        
        return segments
    
    def get_db_stats(self):
        """
        Get statistics about the current database state.
        
        Returns:
            Dict containing database statistics
        """
        stats = {
            "database_path": self.db_path,
            "collection_name": self.collection_name,
            "embedding_type": "OpenAI" if self.use_openai_embeddings else "SentenceTransformer",
            "total_chunks": 0,
            "unique_issues": 0,
            "chunk_types": {},
            "issue_states": {}
        }
        
        try:
            # Get total number of chunks
            total_chunks = self.collection.count()
            stats["total_chunks"] = total_chunks
            
            # If no chunks, return early
            if total_chunks == 0:
                return stats
                
            # Get metadata for all chunks
            # We need to batch this for large collections
            batch_size = 1000
            all_metadata = []
            
            for offset in range(0, total_chunks, batch_size):
                # Get a batch of IDs
                batch_limit = min(batch_size, total_chunks - offset)
                batch_results = self.collection.get(limit=batch_limit, offset=offset)
                all_metadata.extend(batch_results["metadatas"])
                
            # Process metadata
            issue_numbers = set()
            chunk_types = {}
            issue_states = {}
            
            for meta in all_metadata:
                # Count unique issues
                if "issue_number" in meta:
                    issue_numbers.add(meta["issue_number"])
                    
                # Count chunk types
                if "chunk_type" in meta:
                    chunk_type = meta["chunk_type"]
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                # Count issue states
                if "state" in meta:
                    state = meta["state"]
                    issue_states[state] = issue_states.get(state, 0) + 1
            
            stats["unique_issues"] = len(issue_numbers)
            stats["chunk_types"] = chunk_types
            stats["issue_states"] = issue_states
            
            return stats
        except Exception as e:
            print(f"Error getting database statistics: {str(e)}")
            return stats

    def print_progress_report(self):
        """
        Print detailed report about the current state of the database.
        """
        stats = self.get_db_stats()
        
        print("\n" + "="*50)
        print(f"DATABASE STATUS REPORT")
        print("="*50)
        print(f"Database Path: {stats['database_path']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Embedding Model: {stats['embedding_type']}")
        print("-"*50)
        print(f"Total Semantic Chunks: {stats['total_chunks']}")
        print(f"Unique Issues: {stats['unique_issues']}")
        print("-"*50)
        
        if stats["chunk_types"]:
            print("Chunk Types Distribution:")
            # Sort by count, descending
            for chunk_type, count in sorted(stats["chunk_types"].items(), key=lambda x: x[1], reverse=True):
                percentage = count / stats["total_chunks"] * 100
                print(f"  - {chunk_type}: {count} chunks ({percentage:.1f}%)")
        
        if stats["issue_states"]:
            print("\nIssue States:")
            for state, count in sorted(stats["issue_states"].items(), key=lambda x: x[1], reverse=True):
                percentage = count / stats["unique_issues"] * 100 if stats["unique_issues"] > 0 else 0
                print(f"  - {state}: {count} issues ({percentage:.1f}%)")
        
        print("="*50)
def main():
    """
    Main function to create the vector database.
    """
    parser = argparse.ArgumentParser(description="Create a vector database from GitHub issues")
    parser.add_argument("--input", type=str, default="Issue-migration/issues_summaries.json", 
                      help="Path to the issues JSON file")
    parser.add_argument("--db-path", type=str, default="./chroma_db",
                      help="Path to store the ChromaDB database")
    parser.add_argument("--collection", type=str, default="github_issues",
                      help="Name of the collection in ChromaDB")
    parser.add_argument("--use-openai", action="store_true",
                      help="Use OpenAI embeddings (requires API key)")
    parser.add_argument("--openai-api-key", type=str,
                      help="OpenAI API key (optional if set as environment variable)")
    parser.add_argument("--use-llm-chunking", action="store_true",
                      help="Use LLM-based chunking instead of pattern-based chunking for text segmentation")
    parser.add_argument("--query", type=str,
                      help="Optional query to test the database after creation")
    parser.add_argument("--stats-only", action="store_true",
                      help="Only show database statistics without processing issues")
    # Add Azure OpenAI integration parameters
    parser.add_argument("--use-azure-openai", action="store_true",
                      help="Use Azure OpenAI embeddings with Entra ID authentication")
    parser.add_argument("--azure-endpoint", type=str,
                      help="Azure OpenAI endpoint URL (optional if set as environment variable AZURE_OPENAI_ENDPOINT)")
    parser.add_argument("--azure-deployment", type=str,
                      help="Azure OpenAI deployment name for embeddings (optional if set as environment variable AZURE_OPENAI_DEPLOYMENT)")
    parser.add_argument("--azure-vision-deployment", type=str,
                      help="Azure OpenAI deployment name for vision/chat models (optional if set as environment variable AZURE_OPENAI_VISION_DEPLOYMENT)")
    
    args = parser.parse_args()
    
    # Initialize the database
    db = IssueVectorDatabase(
        issues_json_path=args.input,
        db_path=args.db_path,
        collection_name=args.collection,
        openai_api_key=args.openai_api_key,
        use_openai_embeddings=args.use_openai and not args.use_azure_openai,  # Prioritize Azure if both are specified
        use_azure_openai=args.use_azure_openai,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_vision_deployment=args.azure_vision_deployment,
        use_llm_chunking=args.use_llm_chunking
    )
    
    # If stats-only is provided, just print the database statistics and exit
    if args.stats_only:
        db.print_progress_report()
        return
    
    # Process all issues
    db.process_issues()
    
    # Print the database statistics after processing
    print("\nDatabase processing completed. Current statistics:")
    db.print_progress_report()
    
    # Test query if provided
    if args.query:
        print(f"\nTesting query: '{args.query}'")
        results = db.query(args.query, n_results=3)
        
        print("\nResults:")
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n--- Result {i+1} ---")
            print(f"Issue: #{metadata['issue_number']} - {metadata['issue_title']}")
            print(f"Chunk type: {metadata['chunk_type']}")
            print(f"Content: {doc[:200]}...")

if __name__ == "__main__":
    main()