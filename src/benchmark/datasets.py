"""STUB -- not implemented yet.

Purpose: load a public hallucination-detection benchmark so the pipeline
in claims/ + verification/ can be scored against real labeled examples,
instead of just eyeballing demo output.

Note this is a *different* dataset than src/rag/corpus.py's
rag-mini-wikipedia -- these benchmark datasets already ship their own
(question, retrieved/reference context, generated answer,
hallucination-label) tuples, produced by real LLMs, so no retrieval step
is needed to use them.

Planned approach (not yet decided/implemented):
  - RAGTruth (https://github.com/ParticleMedia/RAGTruth): ~18k
    word-level-annotated LLM responses across QA / summarization /
    data-to-text tasks -- best fit if we want span-level hallucination
    labels to compare against claim-level verification output.
  - HaluEval (https://github.com/RUCAIBox/HaluEval): simpler
    binary-labeled (hallucinated vs. not) QA/dialogue/summarization
    samples -- a faster first pass if span-level detail isn't needed yet.
  - Decision on which one (or both) to use is deferred until
    claims/verification are implemented and we know what granularity of
    label the metrics in benchmark/metrics.py actually need.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkExample:
    question: str
    context: list[str]
    answer: str
    is_hallucinated: bool


def load_ragtruth() -> list[BenchmarkExample]:
    """Loads and normalizes the RAGTruth dataset into BenchmarkExample rows."""
    raise NotImplementedError("RAGTruth loading is not implemented yet (Step 1 is scaffolding only).")


def load_halueval() -> list[BenchmarkExample]:
    """Loads and normalizes the HaluEval dataset into BenchmarkExample rows."""
    raise NotImplementedError("HaluEval loading is not implemented yet (Step 1 is scaffolding only).")
