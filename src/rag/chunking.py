"""Splits raw document text into overlapping word-count chunks.

rag-mini-wikipedia passages are already paragraph-sized, so this mostly
passes them through untouched -- but keeping a real chunker in the
pipeline (rather than skipping this step) means swapping in longer, raw
documents later doesn't require touching any other module.
"""

from dataclasses import dataclass

from src.config import get_config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def chunk_text(doc_id: str, text: str) -> list[Chunk]:
    """Splits `text` into word-count windows of chunk_size_words, each
    overlapping the previous one by chunk_overlap_words.

    A pure word-count split is intentionally simple (no sentence-boundary
    awareness) -- good enough for a demo corpus of short passages, and
    easy to reason about in an interview.
    """
    cfg = get_config()["chunking"]
    size = cfg["chunk_size_words"]
    overlap = cfg["chunk_overlap_words"]

    words = text.split()
    if not words:
        return []

    step = max(size - overlap, 1)
    chunks = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + size]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{index}",
                doc_id=doc_id,
                text=" ".join(window),
            )
        )
        index += 1
        start += step

    return chunks


def chunk_documents(documents) -> list[Chunk]:
    """Applies chunk_text() across a list of CorpusDocument objects."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc.doc_id, doc.text))
    return all_chunks
