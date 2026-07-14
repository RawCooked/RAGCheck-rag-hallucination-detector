"""Loads a labeled hallucination-detection benchmark so the pipeline in
claims/ + verification/ can be scored against real examples, instead of
just eyeballing demo output.

Note this is a *different* dataset than src/rag/corpus.py's
rag-mini-wikipedia -- HaluEval and RAGTruth already ship their own
(question, context, answer, hallucination-label) tuples produced by real
LLMs, so no retrieval step is needed to use them.

load_halueval() is implemented (see below for why HaluEval was picked
over RAGTruth for this first pass). load_ragtruth() is still a stub --
RAGTruth ships as raw JSON on GitHub rather than a clean HF Hub mirror
with a documented schema, and adding it is a reasonable next increment
once the benchmark runner using HaluEval is proven out.
"""

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class BenchmarkExample:
    question: str
    context: list[str]
    answer: str
    is_hallucinated: bool


def load_halueval(limit: int | None = None) -> list[BenchmarkExample]:
    """Loads HaluEval's QA split and normalizes it into BenchmarkExample rows.

    Dataset: pminervini/HaluEval, "qa" config -- a clean HF Hub mirror of
    the original HaluEval (Li et al., 2023) QA task, with a documented
    schema: each row has {knowledge, question, right_answer,
    hallucinated_answer}. Chosen over RAGTruth for this first pass because
    it's directly loadable via `datasets.load_dataset()` with no manual
    parsing, and its right/hallucinated pairing gives an exactly balanced
    binary benchmark for free -- useful for a clean precision/recall
    comparison between the NLI checker and the embedding baseline.

    Each source row produces two BenchmarkExamples (one grounded in
    right_answer, one in hallucinated_answer), both sharing the same
    question and knowledge context -- so the checkers are compared on
    matched pairs, not just aggregate stats.

    Args:
        limit: if given, only load the first `limit` source rows (i.e.
            up to 2 * limit BenchmarkExamples). Useful for a fast local
            smoke test before running the full 10k-row set.

    Returns:
        A list of BenchmarkExample rows, alternating supported/hallucinated.
    """
    split = "data" if limit is None else f"data[:{limit}]"
    dataset = load_dataset("pminervini/HaluEval", "qa", split=split)

    examples = []
    for row in dataset:
        context = [row["knowledge"]]
        examples.append(
            BenchmarkExample(
                question=row["question"], context=context, answer=row["right_answer"], is_hallucinated=False
            )
        )
        examples.append(
            BenchmarkExample(
                question=row["question"],
                context=context,
                answer=row["hallucinated_answer"],
                is_hallucinated=True,
            )
        )
    return examples


def load_ragtruth() -> list[BenchmarkExample]:
    """Loads and normalizes the RAGTruth dataset into BenchmarkExample rows."""
    raise NotImplementedError(
        "RAGTruth loading is not implemented yet -- use load_halueval() for now. "
        "RAGTruth ships as raw JSON (source_info.jsonl + response.jsonl) on "
        "GitHub (github.com/ParticleMedia/RAGTruth) rather than a documented "
        "HF Hub dataset, so this needs a custom parser for its span-level "
        "hallucination labels."
    )
