# analyzer_core/tools/similarity_tool.py
"""
Similar issues search tool for finding related issues
"""

from typing import Dict, Any, Optional
from .base_tool import BaseTool
from RAG.rag_helper import default_rag_helper


class SimilarIssuesSearchTool(BaseTool):
    """Similar issues search tool for finding related issues"""
    
    def __init__(self):
        super().__init__("similar_issues")
    
    async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute similar issues search"""
        try:
            if not self.validate_input(issue_data):
                return {"error": "Invalid input data", "confidence": 0.0}
            
            # Use RAG helper to find similar issues
            results, context = default_rag_helper.query_for_regression_analysis(
                issue_title=issue_data.get('title', ''),
                issue_body=issue_data.get('body', ''),
                n_results=3
            )
            
            return {
                "similar_issues": results,
                "context": context,
                "count": len(results) if results else 0,
                "confidence": 0.7 if results else 0.3
            }
        except Exception as e:
            self.logger.error(f"Similar issues search failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
