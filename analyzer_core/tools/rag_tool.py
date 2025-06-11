# analyzer_core/tools/rag_tool.py
"""
RAG search tool for finding relevant information
"""

from typing import Dict, Any, Optional
from .base_tool import BaseTool
from RAG.rag_helper import default_rag_helper


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
            
            results, context = default_rag_helper.query_for_regression_analysis(
                issue_title=issue_data.get('title', ''),
                issue_body=issue_data.get('body', ''),
                n_results=5
            )
            
            return {
                "search_results": results,
                "context": context,
                "results_count": len(results) if results else 0,
                "confidence": 0.8 if results else 0.3
            }
        except Exception as e:
            self.logger.error(f"RAG search failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
