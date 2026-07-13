"""STUB -- not implemented yet. This is the core of the project.

Purpose: given one claim and the retrieved evidence chunks it should be
grounded in, decide whether the evidence *entails* the claim, *contradicts*
it, or is *neutral* (unrelated/insufficient) -- using a Natural Language
Inference (NLI) model rather than surface-level word/embedding overlap.
NLI catches cases embedding similarity misses, e.g. a claim that mentions
all the right entities but gets a number or relationship backwards.

Planned approach (not yet decided/implemented):
  - Load a pretrained NLI model (candidates: "roberta-large-mnli",
    or a DeBERTa-v3 model fine-tuned on MNLI/FEVER -- these are the
    standard choices for entailment classification).
  - For each claim, run NLI with (premise=evidence_chunk, hypothesis=claim)
    for every retrieved chunk, and treat the claim as "supported" if at
    least one chunk entails it, "hallucinated" if none do (or the top
    scores are contradiction/neutral).
  - Needs a documented threshold/aggregation rule across multiple chunks
    (e.g. max entailment score across chunks vs. requiring one full
    entailment) -- to be decided once real examples are in front of us.

Will be compared against verification/embedding_checker.py (the simpler
baseline) in src/benchmark/, on RAGTruth / HaluEval examples.
"""

from dataclasses import dataclass
from enum import Enum


class EntailmentLabel(Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass
class VerificationResult:
    claim_text: str
    label: EntailmentLabel
    confidence: float
    supporting_chunk_id: str | None


def check_claim_entailment(claim_text: str, evidence_chunks: list[dict]) -> VerificationResult:
    """Checks whether `claim_text` is entailed by any of `evidence_chunks`.

    Args:
        claim_text: a single atomic claim (see src/claims/extractor.py).
        evidence_chunks: retrieved chunks, as returned by
            src/rag/retriever.py's retrieve() (each with a "text" key).

    Returns:
        A VerificationResult describing the strongest entailment relation
        found across all evidence chunks.
    """
    raise NotImplementedError("NLI-based verification is not implemented yet (Step 1 is scaffolding only).")
