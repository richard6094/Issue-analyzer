# analyzer_core/models/analysis_models.py
"""
Data models for analysis context and results
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class AnalysisContext:
    """Context for the entire analysis process"""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    issue_data: Dict[str, Any] = field(default_factory=dict)
    comment_data: Optional[Dict[str, Any]] = None
    trigger_decision: Optional[Dict[str, Any]] = None
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    final_analysis: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class FinalAnalysis:
    """Final analysis result from LLM"""
    issue_type: str
    severity: str
    confidence: float
    summary: str
    detailed_analysis: str
    root_cause: Optional[str] = None
    recommended_labels: List[str] = field(default_factory=list)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    user_comment: str = ""
