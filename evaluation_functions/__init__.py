"""Reusable evaluation library for DiCo-NLI.

The package is intentionally small and dependency-free so it can be reused in
local development, CI, and CodaBench scoring containers.
"""

from .errors import ScoringError, ValidationError
from .labels import ALL_LABELS, REVERSIBLE_LABELS, reverse_label
from .metrics import EvaluationReport, compute_evaluation
from .scorer import score_files
from .aggregation import ScoreProfile, TrackAggregateReport, macro_average_reports
from .templates import build_prediction_template_rows, write_prediction_template

__all__ = [
    "ALL_LABELS",
    "REVERSIBLE_LABELS",
    "EvaluationReport",
    "ScoringError",
    "ScoreProfile",
    "TrackAggregateReport",
    "ValidationError",
    "build_prediction_template_rows",
    "compute_evaluation",
    "macro_average_reports",
    "reverse_label",
    "score_files",
    "write_prediction_template",
]
