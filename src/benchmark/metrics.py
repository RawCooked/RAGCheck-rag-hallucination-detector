"""Scores a hallucination detector's predictions against the labeled
benchmark examples from src/benchmark/datasets.py, and reports metrics
that let the NLI-based checker and the embedding-similarity baseline be
compared head-to-head.

Treated as binary classification (hallucinated vs. supported) at the
whole-answer level: an answer counts as "predicted hallucinated" if
run_benchmark.py finds at least one unsupported claim in it. That's the
granularity HaluEval's labels are at (one label per answer, not per
claim), so it's what these metrics score against -- even though the
checkers themselves operate per-claim internally.
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Computes precision/recall/F1/accuracy for hallucination predictions.

    Args:
        y_true: ground-truth hallucination labels (True = hallucinated).
        y_pred: predicted hallucination labels from a checker.

    Returns:
        A dict with keys "accuracy", "precision", "recall", "f1".
        Precision/recall/F1 treat "hallucinated" (True) as the positive
        class, since that's the case we actually care about catching.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
