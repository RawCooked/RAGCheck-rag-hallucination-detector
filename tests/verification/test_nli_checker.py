import pytest

from src.verification.nli_checker import check_claim_entailment


def test_check_claim_entailment_not_implemented_yet():
    """Documents current state: NLI verification is scaffolded, not built.
    Replace this with real assertions once check_claim_entailment() is implemented.
    """
    with pytest.raises(NotImplementedError):
        check_claim_entailment("Some claim.", evidence_chunks=[])
