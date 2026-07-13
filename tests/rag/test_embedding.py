import pytest

from src.rag.embedding import embed_texts


@pytest.mark.integration
def test_embed_texts_returns_one_vector_per_input():
    vectors = embed_texts(["hello world", "another sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0
