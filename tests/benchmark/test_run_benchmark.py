import pytest

from src.benchmark.datasets import BenchmarkExample
from src.benchmark.run_benchmark import predict_nli, predict_embedding, run

SUPPORTED_EXAMPLE = BenchmarkExample(
    question="Was Lincoln the sixteenth President?",
    context=["Abraham Lincoln was the sixteenth President of the United States."],
    answer="Lincoln was the sixteenth President.",
    is_hallucinated=False,
)

HALLUCINATED_EXAMPLE = BenchmarkExample(
    question="Was Lincoln the sixteenth President?",
    context=["Abraham Lincoln was the sixteenth President of the United States."],
    answer="Lincoln was the seventeenth President.",
    is_hallucinated=True,
)


@pytest.mark.integration
def test_predict_nli_agrees_with_ground_truth_on_clear_cases():
    assert predict_nli(SUPPORTED_EXAMPLE) is False
    assert predict_nli(HALLUCINATED_EXAMPLE) is True


@pytest.mark.integration
def test_predict_embedding_misses_the_swapped_number_hallucination():
    """The baseline's known weakness, exercised through the same
    predict_* interface run_benchmark.run() uses -- topically-identical
    wrong answers pass the similarity bar.
    """
    assert predict_embedding(SUPPORTED_EXAMPLE) is False
    assert predict_embedding(HALLUCINATED_EXAMPLE) is False  # wrongly says "not hallucinated"


@pytest.mark.integration
def test_run_end_to_end_prints_comparison_table(capsys):
    run(limit=3)
    captured = capsys.readouterr()
    assert "NLI checker" in captured.out
    assert "embedding baseline" in captured.out
    assert "accuracy" in captured.out
