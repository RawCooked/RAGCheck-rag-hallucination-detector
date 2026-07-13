"""Loads the demo corpus used to seed the RAG pipeline.

We use `rag-datasets/rag-mini-wikipedia` (HuggingFace Hub): a small set of
~3.2k Wikipedia passages plus a matching QA split. It's a standard, widely
used "toy but real" corpus for demoing RAG, and it stays out of the way of
the actual benchmark data -- RAGTruth / HaluEval will be loaded separately
in src/benchmark/, since they already carry their own (question, context,
answer, hallucination-label) tuples and don't need retrieval at all.
"""

from dataclasses import dataclass

from datasets import load_dataset

from src.config import get_config


@dataclass
class CorpusDocument:
    doc_id: str
    text: str


def load_corpus() -> list[CorpusDocument]:
    """Downloads (or loads from HF cache) the passage corpus.

    Returns one CorpusDocument per passage. Passages in rag-mini-wikipedia
    are already short paragraphs, so downstream chunking mostly passes them
    through -- but we still route everything through chunk_text() so the
    pipeline works unmodified on longer, arbitrary documents later.
    """
    cfg = get_config()["corpus"]
    dataset = load_dataset(cfg["dataset_name"], cfg["text_config"], split="passages")

    return [
        CorpusDocument(doc_id=str(row["id"]), text=row["passage"])
        for row in dataset
    ]


def load_sample_questions(n: int = 5) -> list[dict]:
    """Pulls a few (question, ground_truth_answer) pairs for smoke-testing.

    Only used by the demo script to have something to ask -- not part of
    the real benchmark, which will use RAGTruth/HaluEval questions instead.
    """
    cfg = get_config()["corpus"]
    dataset = load_dataset(cfg["dataset_name"], cfg["qa_config"], split="test")
    subset = dataset.select(range(min(n, len(dataset))))

    return [
        {"question": row["question"], "ground_truth_answer": row["answer"]}
        for row in subset
    ]
