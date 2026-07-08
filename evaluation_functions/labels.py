"""Official DiCo-NLI label inventory and reversal algebra."""

from __future__ import annotations

from enum import Enum

from .errors import ValidationError


class Label(str, Enum):
    """Official task labels.

    The string values are the exact tokens expected in submitted prediction files.
    """

    EQUIVALENCE = "EQUIVALENCE"
    FORWARD_ENTAILMENT = "FORWARD_ENTAILMENT"
    BACKWARD_ENTAILMENT = "BACKWARD_ENTAILMENT"
    NEGATIVE_OTHER = "NEGATIVE_OTHER"


ALL_LABELS: tuple[str, ...] = tuple(label.value for label in Label)

REVERSIBLE_LABELS: tuple[str, ...] = (
    Label.EQUIVALENCE.value,
    Label.FORWARD_ENTAILMENT.value,
    Label.BACKWARD_ENTAILMENT.value,
)

_REVERSE_LABEL_MAP: dict[str, str] = {
    Label.EQUIVALENCE.value: Label.EQUIVALENCE.value,
    Label.FORWARD_ENTAILMENT.value: Label.BACKWARD_ENTAILMENT.value,
    Label.BACKWARD_ENTAILMENT.value: Label.FORWARD_ENTAILMENT.value,
}


def normalize_label(raw_label: object) -> str:
    """Return a stripped label string and reject non-string or empty labels."""

    if not isinstance(raw_label, str):
        raise ValidationError(f"Label must be a string, got {type(raw_label).__name__}.")
    
    label = raw_label.strip()

    if not label:
        raise ValidationError("Label must not be empty.")
    
    return label


def validate_label(raw_label: object, *, field_name: str = "label") -> str:
    """Validate that a raw label is one of the official task labels."""

    label = normalize_label(raw_label)
    
    if label not in ALL_LABELS:
        valid = ", ".join(ALL_LABELS)
        raise ValidationError(f"Invalid {field_name} {label!r}; expected one of: {valid}.")
    
    return label


def is_reversible_label(label: str) -> bool:
    """Return whether a label participates in the deterministic reversal map."""

    return label in _REVERSE_LABEL_MAP


def reverse_label(label: str) -> str:
    """Return the deterministic reverse of a reversible label.

    Raises:
        ValidationError: if ``label`` is not reversible.
    """

    if label not in _REVERSE_LABEL_MAP:
        raise ValidationError(f"Label {label!r} has no deterministic reverse.")
    
    return _REVERSE_LABEL_MAP[label]
