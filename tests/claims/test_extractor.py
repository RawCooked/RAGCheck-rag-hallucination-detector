import pytest

from src.claims.extractor import extract_claims


def test_extract_claims_not_implemented_yet():
    """Documents current state: claim extraction is scaffolded, not built.
    Replace this with real assertions once extract_claims() is implemented.
    """
    with pytest.raises(NotImplementedError):
        extract_claims("Some generated answer.")
