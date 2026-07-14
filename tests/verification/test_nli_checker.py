import pytest

from src.verification.nli_checker import check_claim_entailment, EntailmentLabel

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


def test_no_evidence_is_neutral():
    result = check_claim_entailment("Lincoln was president.", evidence_chunks=[])
    assert result.label == EntailmentLabel.NEUTRAL
    assert result.supporting_chunk_id is None


@pytest.mark.integration
def test_correct_claim_is_entailed():
    result = check_claim_entailment("Lincoln was the sixteenth President.", EVIDENCE)
    assert result.label == EntailmentLabel.ENTAILMENT
    assert result.supporting_chunk_id == "c0"
    assert result.confidence > 0.9


@pytest.mark.integration
def test_swapped_number_is_caught_as_contradiction_where_baseline_would_pass_it():
    """This is the project's core thesis, verified: a claim that swaps one
    number ("seventeenth" for "sixteenth") stays topically identical to
    the evidence -- the embedding baseline (see test_embedding_checker.py's
    test_baseline_weakness_wrong_fact_still_scores_high_similarity) scores
    it as similar enough to pass. NLI reads the actual semantics and
    correctly flags the contradiction.
    """
    result = check_claim_entailment("Lincoln was the seventeenth President.", EVIDENCE)
    assert result.label == EntailmentLabel.CONTRADICTION
    assert result.supporting_chunk_id == "c0"
    assert result.confidence > 0.9
