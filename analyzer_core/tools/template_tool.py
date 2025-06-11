# analyzer_core/tools/template_tool.py
"""
Template generation tool for creating issue templates and responses
"""

from typing import Dict, Any, Optional
from .base_tool import BaseTool


class TemplateGenerationTool(BaseTool):
    """Template generation tool for creating issue templates and responses"""
    
    def __init__(self):
        super().__init__("template_generation")
    
    async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute template generation"""
        try:
            # Basic template suggestions based on issue content
            title = issue_data.get('title', '').lower()
            body = issue_data.get('body', '').lower()
            combined_content = f"{title} {body}"
            
            if comment_data:
                combined_content += f" {comment_data.get('body', '').lower()}"
            
            if any(word in combined_content for word in ['bug', 'error', 'crash', 'fail']):
                template_type = "bug_report"
            elif any(word in combined_content for word in ['feature', 'request', 'add', 'enhance']):
                template_type = "feature_request"
            elif any(word in combined_content for word in ['question', 'how', 'help', 'why']):
                template_type = "question"
            else:
                template_type = "general"
            
            templates = {
                "bug_report": {
                    "sections": ["Steps to Reproduce", "Expected Behavior", "Actual Behavior", "Environment Details"],
                    "description": "Bug report template for systematic issue reporting"
                },
                "feature_request": {
                    "sections": ["Problem Description", "Proposed Solution", "Alternatives Considered", "Use Cases"],
                    "description": "Feature request template for enhancement proposals"
                },
                "question": {
                    "sections": ["Context", "Question", "What I've Tried", "Expected Outcome"],
                    "description": "Question template for support requests"
                },
                "general": {
                    "sections": ["Description", "Context", "Expected Outcome"],
                    "description": "General template for various issue types"
                }
            }
            
            return {
                "recommended_template": template_type,
                "template_data": templates[template_type],
                "confidence": 0.6
            }
        except Exception as e:
            self.logger.error(f"Template generation failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
