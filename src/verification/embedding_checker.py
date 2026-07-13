"""STUB -- not implemented yet.

Purpose: the "simple baseline" the NLI-based checker (nli_checker.py) is
benchmarked against. Instead of true entailment reasoning, this scores a
claim as "supported" purely by embedding-similarity to the evidence -- the
approach many naive RAG-QA validators actually ship with. Part of this
project's point is to show *why* that's insufficient (e.g. a claim can be
semantically close to a chunk about the same topic while still asserting
something the chunk doesn't say, or even contradicts).

Planned approach (not yet decided/implemented):
  - Reuse src/rag/embedding.py's embed_texts() to embed the claim and each
    evidence chunk with the same sentence-transformers model already used
    for retrieval.
  - Score = max cosine similarity between the claim embedding and any
    evidence chunk embedding.
  - Threshold that score (to be tuned against labeled examples) to decide
    "supported" vs. "hallucinated".

Will be compared against verification/nli_checker.py in src/benchmark/,
on the same RAGTruth / HaluEval examples, to quantify how much NLI-based
entailment improves over similarity alone.
"""

from dataclasses import dataclass


@dataclass
class EmbeddingVerificationResult:
    claim_text: str
    max_similarity: float
    is_supported: bool
    most_similar_chunk_id: str | None


def check_claim_similarity(claim_text: str, evidence_chunks: list[dict]) -> EmbeddingVerificationResult:
    """Scores `claim_text` by embedding-similarity to `evidence_chunks`.

    Args:
        claim_text: a single atomic claim (see src/claims/extractor.py).
        evidence_chunks: retrieved chunks, as returned by
            src/rag/retriever.py's retrieve() (each with a "text" key).

    Returns:
        An EmbeddingVerificationResult with the best similarity score found
        and a supported/hallucinated decision against a threshold.
    """
    raise NotImplementedError("Embedding-similarity baseline is not implemented yet (Step 1 is scaffolding only).")
