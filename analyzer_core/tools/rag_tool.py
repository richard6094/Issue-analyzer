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
        super().__init__("rag_search")   
           async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute RAG search for relevant information"""
        try:
            if not self.validate_input(issue_data):
                return {"error": "Invalid input data", "confidence": 0.0}

            issue_content = self.get_issue_content(issue_data, comment_data)
            
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
            self.logger.error(f"RAG search failed: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "confidence": 0.0}
