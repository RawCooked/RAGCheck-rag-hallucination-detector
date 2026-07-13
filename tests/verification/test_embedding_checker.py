import pytest

from src.verification.embedding_checker import check_claim_similarity


def test_check_claim_similarity_not_implemented_yet():
    """Documents current state: the embedding-similarity baseline is
    scaffolded, not built. Replace this with real assertions once
    check_claim_similarity() is implemented.
    """
    with pytest.raises(NotImplementedError):
        check_claim_similarity("Some claim.", evidence_chunks=[])
