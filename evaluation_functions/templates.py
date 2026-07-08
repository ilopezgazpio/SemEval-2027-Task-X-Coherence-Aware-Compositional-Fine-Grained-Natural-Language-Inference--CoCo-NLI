"""Prediction-template helpers for fixed DiCo-NLI reference rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .schemas import GoldInstance

PREDICTION_TEMPLATE_FIELDS: tuple[str, ...] = ("instance_id", "label")


def build_prediction_template_rows(records: Iterable[GoldInstance]) -> list[dict[str, str]]:
    """Return minimal prediction rows for a fixed reference set."""

    return [
        {"instance_id": record.instance_id, "label": ""}
        for record in sorted(records, key=lambda item: item.instance_id)
    ]


def write_prediction_template(records: Iterable[GoldInstance], output_path: str | Path) -> None:
    """Write a minimal prediction template with an empty label column."""

    rows = build_prediction_template_rows(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_TEMPLATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
