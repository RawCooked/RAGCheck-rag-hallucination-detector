import pytest

from src.benchmark.datasets import load_halueval, load_ragtruth


def test_load_ragtruth_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        load_ragtruth()


@pytest.mark.integration
def test_load_halueval_returns_matched_supported_and_hallucinated_pairs():
    examples = load_halueval(limit=5)

    # Each of the 5 source rows yields 2 examples (right + hallucinated).
    assert len(examples) == 10

    # Rows alternate: even index = supported, odd index = hallucinated,
    # and each pair shares the same question/context.
    for i in range(0, len(examples), 2):
        supported, hallucinated = examples[i], examples[i + 1]
        assert supported.is_hallucinated is False
        assert hallucinated.is_hallucinated is True
        assert supported.question == hallucinated.question
        assert supported.context == hallucinated.context
        assert supported.answer != hallucinated.answer
