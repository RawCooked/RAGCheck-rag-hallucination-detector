from src.benchmark.metrics import compute_classification_metrics


def test_perfect_predictions_score_1():
    y_true = [True, True, False, False]
    y_pred = [True, True, False, False]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_missed_hallucination_lowers_recall_not_precision():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]  # missed one real hallucination
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 1.0  # every flag it made was correct
    assert metrics["recall"] == 0.5  # only caught half the real hallucinations


def test_false_alarm_lowers_precision_not_recall():
    y_true = [True, False, False, False]
    y_pred = [True, True, False, False]  # one false alarm
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["recall"] == 1.0  # caught the one real hallucination
    assert metrics["precision"] == 0.5  # half its flags were wrong


def test_no_positives_predicted_does_not_crash():
    y_true = [True, False]
    y_pred = [False, False]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0  # zero_division=0, not a ZeroDivisionError
    assert metrics["recall"] == 0.0
