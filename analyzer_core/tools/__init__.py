# analyzer_core/tools/__init__.py
"""
Tool implementations for issue analysis
"""

from .base_tool import BaseTool
from .rag_tool import RAGTool
from .image_tool import ImageAnalysisTool
from .regression_tool import RegressionAnalysisTool
from .similarity_tool import SimilarIssuesSearchTool
from .template_tool import TemplateGenerationTool
from ..models.tool_models import AvailableTools

# Tool registry
def get_tool_registry():
    """Get the complete tool registry"""
    return {
        AvailableTools.RAG_SEARCH: RAGTool(),
        AvailableTools.IMAGE_ANALYSIS: ImageAnalysisTool(),
        AvailableTools.REGRESSION_ANALYSIS: RegressionAnalysisTool(),
        AvailableTools.SIMILAR_ISSUES: SimilarIssuesSearchTool(),
        AvailableTools.TEMPLATE_GENERATION: TemplateGenerationTool(),
        # These tools are not yet implemented
        # AvailableTools.CODE_SEARCH: CodeSearchTool(),
        # AvailableTools.DOCUMENTATION_LOOKUP: DocumentationLookupTool(),
    }

__all__ = [
    'BaseTool',
    'RAGTool',
    'ImageAnalysisTool', 
    'RegressionAnalysisTool',
    'SimilarIssuesSearchTool',
    'TemplateGenerationTool',
    'get_tool_registry'
]
