# analyzer_core/tools/regression_tool.py
"""
Regression analysis tool for determining if an issue is a regression
"""

import os
from typing import Dict, Any, Optional
from .base_tool import BaseTool


class RegressionAnalysisTool(BaseTool):
    """Regression analysis tool for determining if an issue is a regression"""
    
    def __init__(self):
        super().__init__("regression_analysis")
    
    async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute regression analysis"""
        try:
            # Import here to avoid circular imports
            from scripts.analyze_issue import analyze_issue_for_regression, get_azure_chat_model
            
            issue_content = self.get_issue_content(issue_data, comment_data)
            chat_model = get_azure_chat_model("gpt-4o")
            
            # Get issue number from environment or use a default
            issue_number = os.environ.get("ISSUE_NUMBER", "0")
            
            result = analyze_issue_for_regression(
                issue_content=issue_content,
                issue_number=int(issue_number),
                chat_model=chat_model,
                issue_title=issue_data.get('title'),
                issue_body=issue_data.get('body')
            )
            
            return result
        except Exception as e:
            self.logger.error(f"Regression analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
