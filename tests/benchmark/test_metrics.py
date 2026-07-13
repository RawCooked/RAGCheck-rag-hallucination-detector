import pytest

from src.benchmark.metrics import compute_classification_metrics


def test_compute_classification_metrics_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        compute_classification_metrics(y_true=[True], y_pred=[False])
