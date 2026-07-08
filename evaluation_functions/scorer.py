"""High-level scoring orchestration for local use and CodaBench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_prediction_file, read_reference_file
from .metrics import EvaluationReport, compute_evaluation
from .validation import validate_prediction_records, validate_reference_records

DEFAULT_SCORES_JSON = "scores.json"
DEFAULT_SCORES_TXT = "scores.txt"


def score_files(
    reference_path: str | Path,
    prediction_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> EvaluationReport:
    """Score one prediction file against one fixed reference CSV/TSV file.

    Args:
        reference_path: Path to the hidden fixed reference CSV/TSV file.
        prediction_path: Path to the participant prediction CSV/TSV file.
        output_dir: Optional directory where ``scores.json`` and ``scores.txt``
            are written. CodaBench can point this to its output directory.
        write_outputs: If true and ``output_dir`` is provided, write score files.

    Raises:
        ScoringError: if either file is malformed or incomplete.
    """

    gold_by_id = validate_reference_records(read_reference_file(reference_path))
    prediction_records = read_prediction_file(prediction_path)
    predicted_labels = validate_prediction_records(prediction_records, gold_by_id)
    report = compute_evaluation(gold_by_id, predicted_labels)

    if output_dir is not None and write_outputs:
        write_score_outputs(report, output_dir)
    return report


def score_codabench_directories(
    reference_dir: str | Path,
    submission_dir: str | Path,
    output_dir: str | Path,
    *,
    gold_filename: str = "gold.csv",
    prediction_filename: str = "predictions.csv",
) -> EvaluationReport:
    """Score using the directory layout commonly used by CodaBench."""

    reference_path = Path(reference_dir) / gold_filename
    prediction_path = Path(submission_dir) / prediction_filename
    return score_files(reference_path, prediction_path, output_dir=output_dir)


def write_score_outputs(report: EvaluationReport, output_dir: str | Path) -> None:
    """Write JSON and CodaBench-friendly text score outputs."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / DEFAULT_SCORES_JSON
    text_path = out_dir / DEFAULT_SCORES_TXT

    payload = report.to_dict()
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(_format_scores_txt(payload), encoding="utf-8")


def _format_scores_txt(payload: dict[str, Any]) -> str:
    """Return a simple key-value score format accepted by many platforms."""

    return (
        f"weighted_f1: {payload['weighted_f1']:.12f}\n"
        f"soft_cons: {payload['soft_cons']:.12f}\n"
        f"hard_cons: {payload['hard_cons']:.12f}\n"
    )
