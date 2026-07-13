"""STUB -- not implemented yet.

Purpose: score a hallucination detector's predictions against the labeled
benchmark examples from src/benchmark/datasets.py, and report metrics that
let the NLI-based checker and the embedding-similarity baseline be
compared head-to-head.

Planned approach (not yet decided/implemented):
  - Treat this as binary classification (hallucinated vs. supported) per
    claim, and report precision / recall / F1 (scikit-learn's
    precision_recall_fscore_support covers this).
  - Also report accuracy at the whole-answer level (an answer counts as
    hallucinated if it contains >=1 hallucinated claim), since that's
    closer to what a product-facing "flag this answer" feature needs.
  - Confusion matrix broken down by task type (QA vs. summarization vs.
    data-to-text) if using RAGTruth, since hallucination patterns differ
    a lot across those.
"""


def compute_classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Computes precision/recall/F1/accuracy for hallucination predictions.

    Args:
        y_true: ground-truth hallucination labels (True = hallucinated).
        y_pred: predicted hallucination labels from a checker.

    Returns:
        A dict of metric name -> value.
    """
    raise NotImplementedError("Metrics computation is not implemented yet (Step 1 is scaffolding only).")
