"""STUB -- not implemented yet.

Purpose: break a generated answer into individual "atomic claims" -- short,
self-contained factual statements -- so each one can be checked against the
retrieved documents independently. A hallucination often hides inside one
clause of an otherwise-supported sentence, so claim-level checking is more
precise than checking the whole answer as one block.

Planned approach (not yet decided/implemented):
  - Simplest baseline: split the answer into sentences and treat each
    sentence as one claim.
  - Better: use an LLM prompt ("list the atomic factual claims in this
    text") to decompose compound sentences into single-fact statements,
    similar to the decomposition step used in the RAGTruth / FActScore
    literature.

Will be exercised by src/benchmark/run_benchmark.py once verification/ is
also implemented, so extracted claims can be fed to the NLI/embedding
checkers.
"""

from dataclasses import dataclass


@dataclass
class Claim:
    text: str
    source_sentence: str


def extract_claims(answer: str) -> list[Claim]:
    """Splits `answer` into atomic claims.

    Args:
        answer: the generated answer text to decompose.

    Returns:
        A list of Claim objects, one per atomic factual statement.
    """
    raise NotImplementedError("Claim extraction is not implemented yet (Step 1 is scaffolding only).")
