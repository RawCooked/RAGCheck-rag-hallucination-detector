from src.claims.extractor import extract_claims


def test_single_sentence_answer_returns_one_claim():
    claims = extract_claims("Montevideo is the capital of Uruguay.")
    assert len(claims) == 1
    assert claims[0].text == "Montevideo is the capital of Uruguay."
    assert claims[0].source_sentence == claims[0].text


def test_multi_sentence_answer_splits_into_multiple_claims():
    answer = "Lincoln was the sixteenth President. He was assassinated in 1865."
    claims = extract_claims(answer)
    assert [c.text for c in claims] == [
        "Lincoln was the sixteenth President.",
        "He was assassinated in 1865.",
    ]


def test_empty_answer_returns_no_claims():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_single_word_answer_returns_one_claim():
    # flan-t5-base often returns terse answers like "yes" or "Montevideo"
    # with no terminal punctuation -- these must still produce one claim.
    claims = extract_claims("Montevideo")
    assert len(claims) == 1
    assert claims[0].text == "Montevideo"


def test_abbreviation_mis_split_is_a_known_limitation():
    # Documents current behavior rather than asserting it's correct: a
    # plain sentence-ending-punctuation regex has no notion of
    # abbreviations, so "Gen." reads as a sentence boundary. Flagged in
    # the extractor's docstring as an accepted limitation of the
    # sentence-splitting baseline, to be fixed by the planned LLM-based
    # decomposition upgrade.
    claims = extract_claims("Gen. Grant led the Union army.")
    assert len(claims) == 2
