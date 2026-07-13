"""Orchestrates the demo RAG pipeline: build the index once, then answer
questions against it.

This is the only module scripts/run_pipeline.py talks to -- it doesn't
know about Chroma, sentence-transformers, or flan-t5 directly, just this
module's two functions.
"""

from dataclasses import dataclass

from src.rag.corpus import load_corpus
from src.rag.chunking import chunk_documents
from src.rag.vectorstore import is_populated, index_chunks
from src.rag.retriever import retrieve
from src.rag.generator import generate_answer


@dataclass
class RAGResult:
    question: str
    retrieved_chunks: list[dict]
    answer: str


def ensure_index_built() -> None:
    """Builds the Chroma index from the demo corpus if it isn't already
    populated. Safe to call on every run -- it's a no-op after the first.
    """
    if is_populated():
        return

    print("Index is empty. Loading corpus and building it now (one-time)...")
    documents = load_corpus()
    chunks = chunk_documents(documents)
    print(f"  {len(documents)} documents -> {len(chunks)} chunks")
    index_chunks(chunks)
    print("  Index built.")


def answer_question(question: str, top_k: int | None = None) -> RAGResult:
    """Runs the full retrieve -> generate flow for a single question."""
    chunks = retrieve(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return RAGResult(question=question, retrieved_chunks=chunks, answer=answer)
