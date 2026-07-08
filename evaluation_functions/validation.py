"""Validation for fixed DiCo-NLI reference and prediction CSV files."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .errors import ValidationError
from .labels import is_reversible_label, reverse_label
from .schemas import GoldInstance, PredictionRecord, ReversiblePair


def validate_reference_records(records: list[GoldInstance]) -> dict[str, GoldInstance]:
    """Validate fixed hidden reference rows and return them by instance id."""

    if not records:
        raise ValidationError("Reference file contains no records.")

    gold_by_id: dict[str, GoldInstance] = {}

    for record in records:
        _validate_identifier(record.instance_id, field_name="reference instance_id")
        _validate_identifier(record.pair_id, field_name="reference pair_id")

        if record.reverse_pair_id is not None:
            _validate_identifier(record.reverse_pair_id, field_name="reference reverse_pair_id")
        if record.instance_id in gold_by_id:
            raise ValidationError(f"Duplicate reference instance_id: {record.instance_id!r}.")

        gold_by_id[record.instance_id] = record

    for record in gold_by_id.values():
        
        if record.reverse_pair_id is None:
            continue
        
        if record.reverse_pair_id == record.instance_id:
            raise ValidationError(f"{record.instance_id!r} cannot be its own reverse_pair_id.")
        if not is_reversible_label(record.label):
            raise ValidationError(
                f"{record.instance_id!r} has reverse_pair_id but non-reversible label "
                f"{record.label!r}."
            )
        
        reverse_record = gold_by_id.get(record.reverse_pair_id)
        
        if reverse_record is None:
            raise ValidationError(
                f"{record.instance_id!r} references unknown reverse_pair_id "
                f"{record.reverse_pair_id!r}."
            )
        
        if reverse_record.reverse_pair_id != record.instance_id:
            raise ValidationError(
                f"{record.instance_id!r} and {reverse_record.instance_id!r} are not "
                "reciprocal reverse pairs."
            )
        if reverse_record.pair_id != record.pair_id:
            raise ValidationError(
                f"{record.instance_id!r} and {reverse_record.instance_id!r} have "
                "different pair_id values."
            )
        if reverse_record.label != reverse_label(record.label):
            raise ValidationError(
                f"{record.instance_id!r} and {reverse_record.instance_id!r} have "
                "incompatible reverse labels."
            )

    return gold_by_id


def validate_prediction_records(predictions: list[PredictionRecord], gold_by_id: dict[str, GoldInstance]) -> dict[str, str]:
    """Validate a complete prediction set and return labels by instance id."""

    if not predictions:
        raise ValidationError("Prediction file contains no records.")
    
    predicted_labels: dict[str, str] = {}
    
    for record in predictions:
        _validate_identifier(record.instance_id, field_name="prediction instance_id")
        
        if record.instance_id in predicted_labels:
            raise ValidationError(f"Duplicate prediction for instance_id {record.instance_id!r}.")
        predicted_labels[record.instance_id] = record.label

    gold_ids = set(gold_by_id)
    prediction_ids = set(predicted_labels)
    missing = sorted(gold_ids - prediction_ids)
    extra = sorted(prediction_ids - gold_ids)
    
    if missing or extra:

        fragments: list[str] = []

        if missing:
            fragments.append(_summarize_ids("missing predictions", missing))
            
        if extra:
            fragments.append(_summarize_ids("unknown predictions", extra))
            
        raise ValidationError("; ".join(fragments) + ".")

    return predicted_labels


def build_reversible_pairs(gold_by_id: dict[str, GoldInstance]) -> list[ReversiblePair]:
    """Build unique reciprocal pairs used by SoftCons and HardCons."""

    seen: set[frozenset[str]] = set()
    pairs: list[ReversiblePair] = []
    
    for record in gold_by_id.values():
        if record.reverse_pair_id is None:
            continue
        
        reverse_record = gold_by_id[record.reverse_pair_id]
        key = frozenset((record.instance_id, reverse_record.instance_id))
        
        if key in seen:
            continue
        
        pairs.append(
            ReversiblePair(
                first_id=record.instance_id,
                second_id=reverse_record.instance_id,
                first_label=record.label,
                second_label=reverse_record.label,
            )
        )
        seen.add(key)

    if not pairs:
        raise ValidationError("No reversible pairs found for consistency metrics.")
    
    return pairs


def label_distribution(records: Iterable[GoldInstance]) -> dict[str, int]:
    """Return a gold-label distribution useful for validation reports."""

    return dict(Counter(record.label for record in records))


def _validate_identifier(value: str, *, field_name: str) -> None:
    if not value:
        raise ValidationError(f"{field_name} must not be empty.")
    
    if any(character.isspace() for character in value):
        raise ValidationError(f"{field_name} {value!r} must not contain whitespace.")


def _summarize_ids(label: str, ids: list[str], *, limit: int = 10) -> str:
    shown = ", ".join(repr(instance_id) for instance_id in ids[:limit])
    remainder = len(ids) - limit
    if remainder > 0:
        shown = f"{shown}, ... (+{remainder} more)"
    return f"{label}: {shown}"
