# analyzer_core/tools/rag_tool.py
"""
Enhanced RAG (Retrieval Augmented Generation) tool for general text data querying
Supports multiple database registration and flexible knowledge retrieval
"""

import os
import json
import chromadb
from typing import Dict, Any, Optional, List, Union
from chromadb.config import Settings
from .base_tool import BaseTool
from ..utils.json_utils import extract_json_from_llm_response
from ..utils.token_optimizer import create_token_optimizer, validate_query


class RAGTool(BaseTool):
    """Enhanced RAG tool for general text data querying with multi-database support"""
    
    def __init__(self):
        super().__init__("rag_query")
        
        # Registry for multiple databases
        self.database_registry = {}
        self.default_database = None
        
        # Configuration
        self.config_file = "rag_databases.json"
        self.load_database_registry()
        
    def load_database_registry(self):
        """Load database registry from configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.database_registry = config.get('databases', {})
                    self.default_database = config.get('default_database', None)
                    self.logger.info(f"Loaded {len(self.database_registry)} databases from registry")
            else:
                # Register default databases if config doesn't exist
                self._register_default_databases()
        except Exception as e:
            self.logger.error(f"Failed to load database registry: {e}")
            self._register_default_databases()
    
    def save_database_registry(self):
        """Save database registry to configuration file"""
        try:
            config = {
                'databases': self.database_registry,
                'default_database': self.default_database
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.logger.info("Database registry saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save database registry: {e}")
    
    def _register_default_databases(self):
        """Register default databases"""
        # Register the main chroma database
        main_db_path = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(main_db_path):
            self.register_database(
                name="main",
                path=main_db_path,
                description="Main issue database",
                collection_name="issue_collection",
                set_as_default=True
            )
    
    def register_database(self, name: str, path: str, description: str = "", 
                         collection_name: str = "default_collection", 
                         set_as_default: bool = False) -> bool:
        """
        Register a new database
        
        Args:
            name: Database name (unique identifier)
            path: Path to the database directory
            description: Human-readable description
            collection_name: Collection name within the database
            set_as_default: Whether to set this as the default database
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate database path
            if not os.path.exists(path):
                self.logger.error(f"Database path does not exist: {path}")
                return False
            
            # Test database connection
            if not self._test_database_connection(path, collection_name):
                self.logger.error(f"Cannot connect to database at: {path}")
                return False
            
            # Register database
            self.database_registry[name] = {
                'path': path,
                'description': description,
                'collection_name': collection_name,
                'registered_at': str(os.path.getctime(path))
            }
            
            if set_as_default or self.default_database is None:
                self.default_database = name
            
            self.save_database_registry()
            self.logger.info(f"Database '{name}' registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register database '{name}': {e}")
            return False
    
    def unregister_database(self, name: str) -> bool:
        """Unregister a database"""
        if name not in self.database_registry:
            self.logger.error(f"Database '{name}' not found in registry")
            return False
        
        del self.database_registry[name]
        
        # Update default if necessary
        if self.default_database == name:
            self.default_database = next(iter(self.database_registry.keys()), None)
        
        self.save_database_registry()
        self.logger.info(f"Database '{name}' unregistered successfully")
        return True
    
    def list_databases(self) -> Dict[str, Dict[str, Any]]:
        """List all registered databases"""
        return self.database_registry.copy()
    
    def set_default_database(self, name: str) -> bool:
        """Set default database"""
        if name not in self.database_registry:
            self.logger.error(f"Database '{name}' not found in registry")
            return False
        
        self.default_database = name
        self.save_database_registry()
        self.logger.info(f"Default database set to '{name}'")
        return True
    
    def _test_database_connection(self, path: str, collection_name: str) -> bool:
        """Test database connection"""
        try:
            client = chromadb.PersistentClient(
                path=path,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            collection = client.get_or_create_collection(name=collection_name)
            # Try to get count (basic operation)
            collection.count()            
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def query_database(self, query_text: str, database_name: str = None,
                      n_results: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query a specific database with token length protection
        
        Args:
            query_text: Text to search for
            database_name: Database to query (uses default if None)
            n_results: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Dict containing query results
        """
        try:
            # Determine which database to use
            db_name = database_name or self.default_database
            if not db_name or db_name not in self.database_registry:
                return {
                    "error": f"Database '{db_name}' not found in registry",
                    "available_databases": list(self.database_registry.keys())
                }
            
            db_info = self.database_registry[db_name]
            
            # === QUERY TOKEN PROTECTION ===
            # Get embedding model info from database config
            embedding_model = db_info.get('embedding_model', 'text-embedding-3-small')
            
            # Validate query token length
            is_valid, token_count, validation_message = validate_query(query_text, embedding_model)
            self.logger.info(f"Query validation for '{db_name}': {validation_message}")
            
            # Track original query for reporting
            original_query = query_text
            was_truncated = False
            
            if not is_valid:
                # Try to truncate the query
                token_optimizer = create_token_optimizer(embedding_model)
                truncated_query, was_truncated, truncate_message = token_optimizer.truncate_query_if_needed(query_text)
                
                if was_truncated:
                    self.logger.warning(f"Query truncated for database '{db_name}': {truncate_message}")
                    query_text = truncated_query
                else:
                    return {
                        "error": f"Query too long and cannot be truncated effectively",
                        "database": db_name,
                        "token_count": token_count,
                        "max_tokens": token_optimizer.effective_max_tokens,
                        "suggestion": "Please shorten your query"
                    }
            # === END QUERY PROTECTION ===
            
            # Connect to database
            client = chromadb.PersistentClient(
                path=db_info['path'],
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            collection = client.get_or_create_collection(name=db_info['collection_name'])
            
            # Perform query
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'] if include_metadata else ['documents']
            )
            
            # Format results with token info
            formatted_results = self._format_query_results(results, db_name, query_text)
            
            # Add token protection info to results
            formatted_results['token_info'] = {
                'original_query_tokens': token_count if not was_truncated else create_token_optimizer(embedding_model).count_tokens(original_query),
                'processed_query_tokens': token_count if not was_truncated else create_token_optimizer(embedding_model).count_tokens(query_text),
                'was_truncated': was_truncated,
                'embedding_model': embedding_model
            }
            
            if was_truncated:
                formatted_results['warnings'] = [f"Query was truncated due to token length limits"]
            
            self.logger.info(f"Query executed successfully on database '{db_name}', found {len(results['documents'][0])} results")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Query failed on database '{db_name}': {e}")
            return {
                "error": str(e),
                "database": db_name,
                "query": query_text
            }
    
    def _format_query_results(self, results: Dict, database_name: str, query_text: str) -> Dict[str, Any]:
        """Format query results for consistent output"""
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
        distances = results.get('distances', [[]])[0] if 'distances' in results else []
        
        formatted_results = []
        for i, doc in enumerate(documents):
            result_item = {
                'content': doc,
                'similarity_score': 1 - distances[i] if distances else None,
                'metadata': metadatas[i] if metadatas else {}
            }
            formatted_results.append(result_item)
        
        return {
            'query': query_text,
            'database': database_name,
            'results_count': len(documents),
            'results': formatted_results,
            'is_successful': len(documents) > 0
        }
    
    def search_across_databases(self, query_text: str, n_results: int = 5, 
                               exclude_databases: List[str] = None) -> Dict[str, Any]:
        """
        Search across all registered databases
        
        Args:
            query_text: Text to search for
            n_results: Number of results per database
            exclude_databases: List of databases to exclude from search
            
        Returns:
            Dict containing results from all databases
        """
        exclude_databases = exclude_databases or []
        all_results = {}
        
        for db_name in self.database_registry:
            if db_name in exclude_databases:
                continue
                
            result = self.query_database(query_text, db_name, n_results, include_metadata=True)
            all_results[db_name] = result
        
        # Combine and rank results
        combined_results = self._combine_search_results(all_results, query_text)
        
        return combined_results
    
    def _combine_search_results(self, all_results: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """Combine results from multiple databases"""
        combined = []
        total_results = 0
        
        for db_name, db_results in all_results.items():
            if db_results.get('is_successful', False):
                for result in db_results.get('results', []):
                    result['source_database'] = db_name
                    combined.append(result)
                    total_results += 1
        
        # Sort by similarity score (descending)
        combined.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return {
            'query': query_text,
            'total_results': total_results,
            'databases_searched': list(all_results.keys()),
            'combined_results': combined,
            'database_breakdown': all_results
        }
    
    def validate_input(self, query_data: Dict[str, Any]) -> bool:
        """Validate RAG query input data"""
        return isinstance(query_data, dict) and bool(query_data.get('query_text', '').strip())
    
    async def execute(self, query_data: Dict[str, Any]) -> Dict[str, Any]:        
        """
        Execute RAG query with flexible parameters
        
        Args:
            query_data: Dictionary containing:
                - query_text: Text to search for (required)
                - database_name: Specific database to query (optional)
                - n_results: Number of results (default: 5)
                - search_all: Whether to search all databases (default: False)
                - exclude_databases: List of databases to exclude (optional)
                - generate_suggestions: Whether to generate LLM-based usage suggestions (default: True)
                
        Returns:
            Dict containing query results and optional usage suggestions
        """
        try:
            if not self.validate_input(query_data):
                return {"error": "Invalid input data", "confidence": 0.0}
            
            query_text = query_data.get('query_text', '')
            if not query_text:
                return {"error": "query_text is required", "confidence": 0.0}
            
            # Check if any databases are registered
            if not self.database_registry:
                return {
                    "error": "No databases registered. Please register at least one database.",
                    "confidence": 0.0
                }
            
            # Determine query mode
            search_all = query_data.get('search_all', False)
            database_name = query_data.get('database_name')
            n_results = query_data.get('n_results', 5)
            exclude_databases = query_data.get('exclude_databases', [])
            if search_all:
                # Search across all databases
                results = self.search_across_databases(query_text, n_results, exclude_databases)
                confidence = self._calculate_confidence(results.get('total_results', 0))
                search_mode = "multi"
            else:
                # Search specific database
                results = self.query_database(query_text, database_name, n_results, include_metadata=True)
                confidence = self._calculate_confidence(results.get('results_count', 0))
                search_mode = "single"
            
            # Generate intelligent usage suggestions
            generate_suggestions = query_data.get('generate_suggestions', True)
            if generate_suggestions:
                use_suggestion = await self._generate_use_suggestion(query_text, results, search_mode)
                results['use_suggestion'] = use_suggestion
            
            results['confidence'] = confidence
            return results
            
        except Exception as e:
            self.logger.error(f"RAG query execution failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    def _calculate_confidence(self, results_count: int) -> float:
        """Calculate confidence score based on results count"""
        if results_count == 0:
            return 0.1
        elif results_count < 3:
            return 0.4
        elif results_count < 6:
            return 0.7
        else:
            return 0.9
    
    def get_database_info(self, database_name: str = None) -> Dict[str, Any]:
        """Get information about a specific database or all databases"""
        if database_name:
            if database_name not in self.database_registry:
                return {"error": f"Database '{database_name}' not found"}
            
            db_info = self.database_registry[database_name].copy()
            
            # Add runtime statistics
            try:
                client = chromadb.PersistentClient(
                    path=db_info['path'],
                    settings=Settings(allow_reset=True, anonymized_telemetry=False)
                )
                collection = client.get_or_create_collection(name=db_info['collection_name'])
                db_info['document_count'] = collection.count()
                db_info['status'] = 'active'
            except Exception as e:
                db_info['status'] = 'error'
                db_info['error'] = str(e)
            
            return db_info
        else:
            # Return info for all databases
            all_info = {}
            for name in self.database_registry:
                all_info[name] = self.get_database_info(name)
            
            return {
                'default_database': self.default_database,
                'total_databases': len(self.database_registry),
                'databases': all_info
            }
    
    async def _generate_use_suggestion(self, query_text: str, search_results: Dict[str, Any], 
                                       search_mode: str = "single") -> Dict[str, Any]:
        """
        Generate intelligent usage suggestions based on RAG search results
        
        Args:
            query_text: Original query text
            search_results: Results from RAG search
            search_mode: "single" or "multi" database search mode
            
        Returns:
            Dict containing intelligent suggestions
        """
        try:
            # Check if we have meaningful results
            if search_mode == "single":
                results_count = search_results.get('results_count', 0)
                results = search_results.get('results', [])
            else:
                results_count = search_results.get('total_results', 0)
                results = search_results.get('combined_results', [])
            
            if results_count == 0:
                return {
                    "summary": "No relevant information found in the knowledge base",
                    "relevance": "none",
                    "actionable_insights": ["Consider expanding the search terms or checking additional databases"],
                    "recommended_approach": "Try alternative keywords or search across different databases",
                    "user_friendly_summary": "No matching information found. You might want to try different search terms."
                }
            
            # Dynamic import of LLM to avoid circular imports
            from LLM.llm_provider import get_llm
            llm = get_llm(provider="azure")
            
            # Build analysis prompt
            prompt = self._build_rag_analysis_prompt(query_text, search_results, search_mode)
            
            # Call LLM to generate suggestions
            from langchain_core.messages import HumanMessage
            
            response = await llm.agenerate([[HumanMessage(content=prompt)]])
            response_text = self._extract_response_text(response)
            
            # Parse JSON response
            suggestion_data = self._parse_json_response(response_text)
            
            # Validate and clean results
            return self._validate_and_clean_rag_suggestion(suggestion_data, results_count, search_mode)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate RAG use suggestion: {e}")
            # Fallback processing
            return self._generate_rag_fallback_suggestion(query_text, search_results, search_mode)
    
    def _build_rag_analysis_prompt(self, query_text: str, search_results: Dict[str, Any], 
                                   search_mode: str) -> str:
        """Build LLM analysis prompt for RAG search results"""
        
        # Prepare results summary
        if search_mode == "single":
            results_count = search_results.get('results_count', 0)
            database_name = search_results.get('database', 'unknown')
            results = search_results.get('results', [])
            
            results_summary = f"Database: {database_name}\nResults: {results_count} documents found\n\n"
            for i, result in enumerate(results[:5], 1):  # Show top 5 results
                content = result.get('content', '')[:500]  # Limit content length
                similarity = result.get('similarity_score', 0)
                results_summary += f"Document {i} (Similarity: {similarity:.3f}):\n{content}...\n\n"
        else:
            results_count = search_results.get('total_results', 0)
            databases = search_results.get('databases_searched', [])
            results = search_results.get('combined_results', [])
            
            results_summary = f"Multi-database search across: {', '.join(databases)}\n"
            results_summary += f"Total results: {results_count} documents\n\n"
            
            for i, result in enumerate(results[:5], 1):  # Show top 5 results
                content = result.get('content', '')[:500]
                similarity = result.get('similarity_score', 0)
                source_db = result.get('source_database', 'unknown')
                results_summary += f"Document {i} from {source_db} (Similarity: {similarity:.3f}):\n{content}...\n\n"
        
        return f"""
You are an expert knowledge management assistant analyzing RAG (Retrieval Augmented Generation) search results.

**SEARCH QUERY**: "{query_text}"

**SEARCH RESULTS** ({results_count} documents found):
{results_summary[:3000]}{'...' if len(results_summary) > 3000 else ''}

**YOUR ANALYSIS MISSION**:
Analyze the retrieved documents and provide actionable insights for the user's query. Focus on:

✅ **Information Quality**: How well do the results answer the query?
✅ **Key Insights**: What are the most important findings from the documents?
✅ **Actionable Guidance**: What specific steps or recommendations can be extracted?
✅ **Knowledge Gaps**: What information might be missing or needs clarification?
✅ **Follow-up Actions**: What should the user do next based on these results?

**RESPONSE FORMAT (JSON REQUIRED)**:
{{
    "summary": "Brief overview of what information was found and its relevance to the query",
    "relevance": "high|medium|low",
    "actionable_insights": [
        "💡 Specific insight 1 with concrete information from the documents",
        "💡 Specific insight 2 with practical guidance",
        "💡 Specific insight 3 with implementation details"
    ],
    "recommended_approach": "Step-by-step recommendation based on the retrieved information",
    "user_friendly_summary": "Clear, jargon-free explanation of what was found and what it means for the user",
    "knowledge_confidence": "high|medium|low - how confident you are that the results fully address the query",
    "suggested_follow_up": [
        "Suggested next step 1",
        "Suggested next step 2"
    ]
}}

**ANALYSIS GUIDELINES**:
- Be specific and reference actual content from the documents
- Focus on practical, actionable information
- Highlight any patterns or common themes across documents
- Note if results are from multiple databases and how they complement each other
- Identify any contradictions or conflicting information
- Suggest ways to get more complete information if needed
- Keep technical details accessible to non-experts in the user_friendly_summary
"""
    
    def _extract_response_text(self, response) -> str:
        """Extract text from LangChain response"""
        if hasattr(response.generations[0][0], 'text'):
            return response.generations[0][0].text
        elif hasattr(response.generations[0][0], 'message') and hasattr(response.generations[0][0].message, 'content'):
            return response.generations[0][0].message.content
        else:
            return str(response.generations[0][0])
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM using existing utility"""
        result = extract_json_from_llm_response(response_text, self.logger)
        if result is None:
            raise ValueError("No valid JSON found in response")
        return result
    
    def _validate_and_clean_rag_suggestion(self, suggestion_data: Dict[str, Any], 
                                           results_count: int, search_mode: str) -> Dict[str, Any]:
        """Validate and clean RAG suggestion data"""
        return {
            "summary": suggestion_data.get("summary", f"Found {results_count} relevant documents"),
            "relevance": suggestion_data.get("relevance", "medium").lower(),
            "actionable_insights": suggestion_data.get("actionable_insights", [])[:5],  # Limit to 5 items
            "recommended_approach": suggestion_data.get("recommended_approach", "Review the retrieved documents for relevant information"),
            "user_friendly_summary": suggestion_data.get("user_friendly_summary", "Retrieved relevant documents from the knowledge base"),
            "knowledge_confidence": suggestion_data.get("knowledge_confidence", "medium").lower(),
            "suggested_follow_up": suggestion_data.get("suggested_follow_up", [])[:3],  # Limit to 3 items
            "search_mode": search_mode,
            "results_analyzed": results_count
        }
    
    def _generate_rag_fallback_suggestion(self, query_text: str, search_results: Dict[str, Any], 
                                          search_mode: str) -> Dict[str, Any]:
        """Generate fallback suggestions when LLM fails"""
        if search_mode == "single":
            results_count = search_results.get('results_count', 0)
            results = search_results.get('results', [])
        else:
            results_count = search_results.get('total_results', 0)
            results = search_results.get('combined_results', [])
        
        # Simple rule-based analysis
        insights = []
        confidence = "low"
        
        if results_count > 0:
            insights.append(f"Found {results_count} potentially relevant documents")
            
            # Analyze similarity scores if available
            similarities = [r.get('similarity_score', 0) for r in results if r.get('similarity_score')]
