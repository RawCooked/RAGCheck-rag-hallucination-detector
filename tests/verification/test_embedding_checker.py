import pytest

from src.verification.embedding_checker import check_claim_similarity

EVIDENCE = [
    {
        "chunk_id": "c0",
        "text": "Abraham Lincoln was the sixteenth President of the United States, "
        "serving from March 4, 1861 until his assassination.",
    },
    {
        "chunk_id": "c1",
        "text": "Montevideo, Uruguay's capital, is home to 1.7 million people.",
    },
]


def test_empty_evidence_is_never_supported():
    result = check_claim_similarity("Lincoln was president.", evidence_chunks=[])
    assert result.is_supported is False
    assert result.max_similarity == 0.0
    assert result.most_similar_chunk_id is None


@pytest.mark.integration
def test_topically_related_claim_scores_higher_than_unrelated_claim():
    related = check_claim_similarity("Lincoln was the sixteenth President.", EVIDENCE)
    unrelated = check_claim_similarity("Bananas are a good source of potassium.", EVIDENCE)
    assert related.max_similarity > unrelated.max_similarity
    assert related.most_similar_chunk_id == "c0"


@pytest.mark.integration
def test_baseline_weakness_wrong_fact_still_scores_high_similarity():
    """Demonstrates the documented limitation this baseline exists to
    illustrate: a claim with a swapped number is topically identical to
    the evidence, so embedding similarity alone can't tell it's wrong.
    """
    wrong_claim = check_claim_similarity("Lincoln was the seventeenth President.", EVIDENCE)
    threshold = 0.5
    # Similarity stays high (same entities/topic) even though the fact is
    # false -- this is exactly what nli_checker.py is meant to catch and
    # this baseline can't.
    assert wrong_claim.max_similarity >= threshold
