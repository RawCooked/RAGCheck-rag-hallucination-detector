import pytest

from src.benchmark.datasets import load_ragtruth, load_halueval


def test_load_ragtruth_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        load_ragtruth()


def test_load_halueval_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        load_halueval()
