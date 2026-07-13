import pytest

from src.benchmark.run_benchmark import main


def test_run_benchmark_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        main()
