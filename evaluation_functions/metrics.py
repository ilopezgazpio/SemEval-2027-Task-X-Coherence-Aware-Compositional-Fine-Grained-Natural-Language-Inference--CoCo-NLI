"""Pure metric computation for DiCo-NLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .labels import ALL_LABELS, REVERSIBLE_LABELS, reverse_label
from .schemas import GoldInstance, ReversiblePair
from .validation import build_reversible_pairs


@dataclass(frozen=True, slots=True)
class PerLabelMetrics:
    """Precision, recall, F1, and count statistics for one label."""

    label: str
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Item-level classification metrics over all labels."""

    weighted_f1: float
    macro_f1: float
    accuracy: float
    total_instances: int
    per_label: dict[str, PerLabelMetrics]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "weighted_f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "total_instances": self.total_instances,
            "per_label": {
                label: metrics.to_dict() for label, metrics in self.per_label.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """Pair-level consistency metrics over reversible pairs."""

    soft_cons: float
    hard_cons: float
    reversible_pairs: int
    soft_consistent_pairs: int
    hard_correct_pairs: int
    pair_errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
            "reversible_pairs": self.reversible_pairs,
            "soft_consistent_pairs": self.soft_consistent_pairs,
            "hard_correct_pairs": self.hard_correct_pairs,
            "pair_errors": self.pair_errors,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete scorer output for one track/run."""

    weighted_f1: float
    soft_cons: float
    hard_cons: float
    classification: ClassificationReport
    consistency: ConsistencyReport

    def to_dict(self) -> dict[str, object]:
        return {
            "weighted_f1": self.weighted_f1,
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
            "classification": self.classification.to_dict(),
            "consistency": self.consistency.to_dict(),
        }


def compute_evaluation(
    gold_by_id: Mapping[str, GoldInstance],
    predicted_labels: Mapping[str, str],
    *,
    max_pair_errors: int = 50,
) -> EvaluationReport:
    """Compute all official and diagnostic metrics."""

    classification = compute_classification_report(gold_by_id, predicted_labels)
    pairs = build_reversible_pairs(dict(gold_by_id))
    consistency = compute_consistency_report(
        pairs,
        predicted_labels,
        max_pair_errors=max_pair_errors,
    )
    return EvaluationReport(
        weighted_f1=classification.weighted_f1,
        soft_cons=consistency.soft_cons,
        hard_cons=consistency.hard_cons,
        classification=classification,
        consistency=consistency,
    )


def compute_classification_report(
    gold_by_id: Mapping[str, GoldInstance],
    predicted_labels: Mapping[str, str],
    *,
    labels: tuple[str, ...] = ALL_LABELS,
) -> ClassificationReport:
    """Compute weighted F1, macro-F1, accuracy, and a confusion matrix."""

    total = len(gold_by_id)

    if total == 0:
        raise ValueError("Cannot compute classification metrics for zero instances.")

    confusion = _empty_confusion_matrix(labels)
    correct = 0
    
    for instance_id, gold_record in gold_by_id.items():
        gold_label = gold_record.label
        predicted_label = predicted_labels[instance_id]
        confusion[gold_label][predicted_label] += 1
        if gold_label == predicted_label:
            correct += 1

    per_label: dict[str, PerLabelMetrics] = {}
    weighted_f1 = 0.0
    macro_f1_values: list[float] = []
    
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        predicted = sum(confusion[other][label] for other in labels)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        metric = PerLabelMetrics(
            label=label,
            support=support,
            predicted=predicted,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        per_label[label] = metric
        weighted_f1 += (support / total) * f1
        if support > 0:
            macro_f1_values.append(f1)

    return ClassificationReport(
        weighted_f1=weighted_f1,
        macro_f1=sum(macro_f1_values) / len(macro_f1_values),
        accuracy=correct / total,
        total_instances=total,
        per_label=per_label,
        confusion_matrix=confusion,
    )


def compute_consistency_report(
    pairs: list[ReversiblePair],
    predicted_labels: Mapping[str, str],
    *,
    max_pair_errors: int = 50,
) -> ConsistencyReport:
    """Compute SoftCons and HardCons over reversible source pairs."""

    if not pairs:
        raise ValueError("Cannot compute consistency metrics without reversible pairs.")

    soft_count = 0
    hard_count = 0
    pair_errors: list[dict[str, str]] = []

    for pair in pairs:
        first_prediction = predicted_labels[pair.first_id]
        second_prediction = predicted_labels[pair.second_id]

        soft_ok = (
            first_prediction in REVERSIBLE_LABELS
            and second_prediction == reverse_label(first_prediction)
        )
        hard_ok = (
            first_prediction == pair.first_label
            and second_prediction == pair.second_label
        )

        if soft_ok:
            soft_count += 1
        if hard_ok:
            hard_count += 1
        elif len(pair_errors) < max_pair_errors:
            pair_errors.append(
                {
                    "first_id": pair.first_id,
                    "second_id": pair.second_id,
                    "first_gold": pair.first_label,
                    "second_gold": pair.second_label,
                    "first_prediction": first_prediction,
                    "second_prediction": second_prediction,
                    "soft_consistent": str(soft_ok).lower(),
                    "hard_correct": str(hard_ok).lower(),
                }
            )

    total = len(pairs)
    return ConsistencyReport(
        soft_cons=soft_count / total,
        hard_cons=hard_count / total,
        reversible_pairs=total,
        soft_consistent_pairs=soft_count,
        hard_correct_pairs=hard_count,
        pair_errors=pair_errors,
    )


def _empty_confusion_matrix(labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {gold: {predicted: 0 for predicted in labels} for gold in labels}


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
