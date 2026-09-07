"""Command-line entry point for the DiCo-NLI scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ScoringError
from .scorer import score_codabench_directories, score_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score DiCo-NLI predictions.")
    parser.add_argument("--gold", type=Path, help="Fixed reference CSV or TSV file.")
    parser.add_argument("--predictions", type=Path, help="Prediction CSV or TSV file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for scores.json and scores.txt.")
    parser.add_argument("--reference-dir", type=Path, help="CodaBench reference directory.")
    parser.add_argument("--submission-dir", type=Path, help="CodaBench submission directory.")
    parser.add_argument("--gold-filename", default="gold.csv", help="Gold filename inside reference-dir.")
    parser.add_argument(
        "--prediction-filename",
        default="predictions.csv",
        help="Prediction filename inside submission-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.reference_dir or args.submission_dir:
            if not (args.reference_dir and args.submission_dir and args.output_dir):
                parser.error("--reference-dir, --submission-dir, and --output-dir must be used together.")
            report = score_codabench_directories(
                args.reference_dir,
                args.submission_dir,
                args.output_dir,
                gold_filename=args.gold_filename,
                prediction_filename=args.prediction_filename,
            )
        else:
            if not (args.gold and args.predictions):
                parser.error("--gold and --predictions are required outside CodaBench directory mode.")
            report = score_files(args.gold, args.predictions, output_dir=args.output_dir)
    except ScoringError as exc:
        print(f"SCORING_ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
