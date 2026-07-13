import pytest

from src.rag.pipeline import ensure_index_built, answer_question


@pytest.mark.integration
def test_end_to_end_pipeline_returns_an_answer():
    """Downloads the corpus + models and builds the index if needed.

    Marked integration (skipped by default -- see pytest.ini) since it's
    slow and needs network access on first run. Run explicitly with:
        pytest -m integration
    """
    ensure_index_built()
    result = answer_question("Was Abraham Lincoln the sixteenth President of the United States?")

    assert result.answer
    assert len(result.retrieved_chunks) > 0
    assert all("text" in chunk for chunk in result.retrieved_chunks)
