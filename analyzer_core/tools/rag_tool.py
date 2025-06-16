# analyzer_core/tools/rag_tool.py
"""
RAG search tool for finding relevant information
"""

from typing import Dict, Any, Optional
from .base_tool import BaseTool
from RAG.rag_helper import query_vectordb_for_regression


class RAGSearchTool(BaseTool):
    """RAG search tool for finding relevant information"""
    
    def __init__(self):
        super().__init__("rag_search")    async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute RAG search for relevant information"""
        try:
            if not self.validate_input(issue_data):
                return {"error": "Invalid input data", "confidence": 0.0}

            issue_content = self.get_issue_content(issue_data, comment_data)
            
            # Add debug information about database path
            try:
                from RAG.rag_helper import DEFAULT_DB_PATH
                import os
                print(f"[RAG_TOOL.PY] Attempting to access RAG database at: {DEFAULT_DB_PATH}")
                print(f"[RAG_TOOL.PY] Database path exists: {os.path.exists(DEFAULT_DB_PATH)}")
                if os.path.exists(DEFAULT_DB_PATH):
                    try:
                        files = os.listdir(DEFAULT_DB_PATH)
                        print(f"[RAG_TOOL.PY] Database directory contents: {files}")
                        sqlite_file = os.path.join(DEFAULT_DB_PATH, "chroma.sqlite3")
                        if os.path.exists(sqlite_file):
                            size = os.path.getsize(sqlite_file)
                            print(f"[RAG_TOOL.PY] chroma.sqlite3 size: {size} bytes")
                    except Exception as list_error:
                        print(f"[RAG_TOOL.PY] Error listing database contents: {list_error}")
            except Exception as path_error:
                print(f"[RAG_TOOL.PY] Error accessing DEFAULT_DB_PATH: {path_error}")
            
            context = query_vectordb_for_regression(
                issue_title=issue_data.get('title', ''),
                issue_body=issue_data.get('body', ''),
                n_results=5
            )
            
            return {
                "context": context,
                "results_count": 1 if context and context != "No similar issues found in the database." else 0,
                "confidence": 0.8 if context and context != "No similar issues found in the database." else 0.3
            }
        except Exception as e:
            self.logger.error(f"[RAG_TOOL.PY] RAG search failed: {str(e)}")
            import traceback
            self.logger.error(f"[RAG_TOOL.PY] Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "confidence": 0.0}
