"""Typed records exchanged by the DiCo-NLI evaluation functions."""

from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_PREDICTION_FIELDS: frozenset[str] = frozenset({"instance_id", "label"})
REQUIRED_REFERENCE_FIELDS: frozenset[str] = frozenset(
    {"instance_id", "pair_id", "reverse_pair_id", "label"}
)


@dataclass(frozen=True, slots=True)
class TextPair:
    """Translated text pair for one language inside a grouped gold item."""

    text1: str
    text2: str


@dataclass(frozen=True, slots=True)
class GoldPair:
    """One grouped DiCo-NLI source pair from the official JSON gold format."""

    pair_id: str
    gold_label: str
    languages: dict[str, TextPair]
    split: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GoldInstance:
    """One fixed reference instance scored by the official evaluator."""

    instance_id: str
    pair_id: str
    label: str
    reverse_pair_id: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One participant prediction."""

    instance_id: str
    label: str


@dataclass(frozen=True, slots=True)
class ReversiblePair:
    """A pair of reciprocal evaluation instances used by consistency metrics."""

    first_id: str
    second_id: str
    first_label: str
    second_label: str
