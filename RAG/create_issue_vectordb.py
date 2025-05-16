import os
import json
import uuid
import argparse
from typing import Dict, List, Any, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError
from tqdm import tqdm
import numpy as np

from openai import OpenAI

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
                 chunk_overlap: int = 100,
                 max_tokens_per_chunk: int = 1000):
        """
        Initialize the Issue Vector Database.
        
        Args:
            issues_json_path: Path to the issues JSON file
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection in ChromaDB
            openai_api_key: OpenAI API key (optional if using local embeddings)
            use_openai_embeddings: Whether to use OpenAI embeddings or SentenceTransformers
            chunk_overlap: Number of characters to overlap between chunks
            max_tokens_per_chunk: Maximum tokens per chunk
        """
        self.issues_json_path = issues_json_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.use_openai_embeddings = use_openai_embeddings
        self.chunk_overlap = chunk_overlap
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Choose embedding function based on configuration
        if use_openai_embeddings:
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
        Chunk text semantically considering boundaries between different parts.
        
        Args:
            text: Text to chunk
            issue_number: Issue number
            issue_data: Complete issue data
            
        Returns:
            List of semantic chunks with metadata
        """
        # Find semantic boundaries
        boundaries = self.identify_semantic_boundaries(text)
        
        # If no boundaries or text is short enough, return as single chunk
        if not boundaries or len(text) < self.max_tokens_per_chunk:
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
        
        # Add start and end positions
        positions = [0] + boundaries + [len(text)]
        positions = sorted(list(set(positions)))  # Remove duplicates and sort
        
        chunks = []
        for i in range(len(positions) - 1):
            start = max(0, positions[i])
            end = min(len(text), positions[i + 1])
            
            # Skip if chunk is too small
            if end - start < 50:
                continue
            
            # Extract the chunk text
            chunk_text = text[start:end].strip()
            
            # Skip empty chunks
            if not chunk_text:
                continue
            
            # Generate a unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            
            # Identify the chunk type
            chunk_type = self.identify_chunk_type(chunk_text)
            
            # Create chunk with metadata
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "issue_number": issue_number,
                    "issue_title": issue_data.get("title", ""),
                    "chunk_type": chunk_type,
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
        # Load issues from file
        issues = self.load_issues()
        
        # Track statistics
        total_chunks = 0
        all_chunks = []
        
        # Process each issue
        for issue in tqdm(issues, desc="Processing issues"):
            # Skip empty issues
            if not issue:
                continue
                
            # Get issue number
            issue_number = issue.get("number", 0)
            
            # Preprocess the issue
            full_text = self.preprocess_issue(issue)
            if not full_text:
                continue
                
            # Break the issue into semantic chunks
            chunks = self.semantic_chunk_text(full_text, issue_number, issue)
            all_chunks.extend(chunks)
            
            total_chunks += len(chunks)
        
        # Find and establish relationships between chunks
        print(f"Finding relationships between {len(all_chunks)} chunks...")
        all_chunks = self.find_related_chunks(all_chunks)
        
        # Add all chunks to ChromaDB
        print(f"Adding {len(all_chunks)} chunks to ChromaDB...")
        
        # Prepare data for batch insertion
        ids = []
        documents = []
        metadatas = []
        
        for chunk in all_chunks:
            chunk_id = chunk["metadata"].pop("chunk_id")  # Extract and remove ID from metadata
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
        
        # Add to collection in batches of 100
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="Adding to database"):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
        
        print(f"Successfully added {total_chunks} semantic chunks to ChromaDB")
        print(f"Vector database is now available at {self.db_path}")
    
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
    parser.add_argument("--query", type=str,
                      help="Optional query to test the database after creation")
    
    args = parser.parse_args()
    
    # Initialize the database
    db = IssueVectorDatabase(
        issues_json_path=args.input,
        db_path=args.db_path,
        collection_name=args.collection,
        openai_api_key=args.openai_api_key,
        use_openai_embeddings=args.use_openai
    )
    
    # Process all issues
    db.process_issues()
    
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