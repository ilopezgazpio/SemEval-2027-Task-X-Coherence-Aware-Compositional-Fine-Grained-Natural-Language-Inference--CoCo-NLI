from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, Union, List


def evaluate(y_true: List[int], y_pred: List[int], num_labels: int) -> Dict[str, Union[float, np.ndarray]]:
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")

    matrix: np.ndarray = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=np.arange(num_labels))

    # Add more metrics if needed

    return {
        "accuracy": float(accuracy),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "matrix": matrix,
    }