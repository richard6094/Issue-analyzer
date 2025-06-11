# analyzer_core/tools/base_tool.py
"""
Base class for all analysis tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all analysis tools"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the tool analysis"""
        pass
    
    def validate_input(self, issue_data: Dict[str, Any]) -> bool:
        """Validate input data before execution"""
        return bool(issue_data.get('title') or issue_data.get('body'))
    
    def get_issue_content(self, issue_data: Dict[str, Any], 
                         comment_data: Optional[Dict[str, Any]] = None) -> str:
        """Get combined issue and comment content"""
        content = f"{issue_data.get('title', '')} {issue_data.get('body', '')}"
        
        if comment_data:
            content += f"\n\nComment: {comment_data.get('body', '')}"
            
        return content.strip()
