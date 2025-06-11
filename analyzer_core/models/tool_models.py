# analyzer_core/models/tool_models.py
"""
Data models for tools and decisions
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


class AvailableTools(Enum):
    """Available tools that the LLM can choose to use"""
    RAG_SEARCH = "rag_search"
    IMAGE_ANALYSIS = "image_analysis"
    REGRESSION_ANALYSIS = "regression_analysis"
    CODE_SEARCH = "code_search"
    SIMILAR_ISSUES = "similar_issues"
    DOCUMENTATION_LOOKUP = "documentation_lookup"
    TEMPLATE_GENERATION = "template_generation"


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool: AvailableTools
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DecisionStep:
    """A single decision step in the analysis process"""
    reasoning: str
    selected_tools: List[AvailableTools]
    expected_outcome: str
    priority: int = 1
    user_info_assessment: Optional[Dict[str, Any]] = None
