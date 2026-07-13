from src.rag.chunking import chunk_text


def test_short_text_produces_single_chunk():
    chunks = chunk_text("doc1", "one two three four five")
    assert len(chunks) == 1
    assert chunks[0].text == "one two three four five"
    assert chunks[0].chunk_id == "doc1-0"
    assert chunks[0].doc_id == "doc1"


def test_empty_text_produces_no_chunks():
    assert chunk_text("doc1", "") == []


def test_long_text_overlaps_between_consecutive_chunks(monkeypatch):
    # Force small windows so the overlap behavior is easy to assert on.
    monkeypatch.setattr(
        "src.rag.chunking.get_config",
        lambda: {"chunking": {"chunk_size_words": 4, "chunk_overlap_words": 2}},
    )
    text = " ".join(str(i) for i in range(10))  # "0 1 2 3 4 5 6 7 8 9"
    chunks = chunk_text("doc1", text)

    assert chunks[0].text == "0 1 2 3"
    assert chunks[1].text == "2 3 4 5"  # last 2 words of chunk 0 repeat
    assert all(c.doc_id == "doc1" for c in chunks)
