"""Scoring, with the threshold treated as something to choose rather than assume."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


@dataclass(frozen=True)
class Evaluation:
    threshold: float
    accuracy: float
    auc: float
    report: str
    matrix: np.ndarray


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1).

    The default 0.5 is only the right cut when the classes are balanced and the
    two error types cost the same. Neither holds here: there are roughly 2.7
    times as many pneumonia cases as normal ones, and a missed pneumonia is not
    interchangeable with a false alarm.
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    return float(thresholds[np.argmax(true_positive_rate - false_positive_rate)])


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float | None = None,
) -> Evaluation:
    """Score predictions at a given threshold, or at the ROC-optimal one."""
    probabilities = np.asarray(probabilities).ravel()
    if threshold is None:
        threshold = best_threshold(y_true, probabilities)
    predicted = (probabilities >= threshold).astype(int)

    return Evaluation(
        threshold=threshold,
        accuracy=float((predicted == y_true).mean()),
        auc=float(roc_auc_score(y_true, probabilities)),
        report=classification_report(y_true, predicted, target_names=class_names, digits=3),
        matrix=confusion_matrix(y_true, predicted),
    )
