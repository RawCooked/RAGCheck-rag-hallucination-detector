"""End-to-end evaluation: scores the NLI checker and the embedding-
similarity baseline against labeled HaluEval examples, and prints a
side-by-side comparison.

Run from the project root:
    python -m src.benchmark.run_benchmark
    python -m src.benchmark.run_benchmark --limit 200

Pipeline per example:
    1. Load labeled (question, context, answer, is_hallucinated) rows
       (src/benchmark/datasets.py).
    2. Extract claims from the answer (src/claims/extractor.py).
    3. Verify each claim against the context with both checkers
       (src/verification/nli_checker.py, embedding_checker.py).
    4. An answer is predicted "hallucinated" if ANY of its claims comes
       back unsupported -- matching HaluEval's one-label-per-answer
       granularity (see metrics.py's docstring).
    5. Score both checkers' predictions against ground truth
       (src/benchmark/metrics.py) and print a comparison table.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.benchmark.datasets import load_halueval, BenchmarkExample
from src.benchmark.metrics import compute_classification_metrics
from src.claims.extractor import extract_claims
from src.config import get_config
from src.verification.embedding_checker import check_claim_similarity
from src.verification.nli_checker import check_claim_entailment, EntailmentLabel


def _as_evidence_chunks(context: list[str]) -> list[dict]:
    """Adapts BenchmarkExample.context (list[str]) into the
    {"text", "chunk_id"} dict shape both checkers expect, which normally
    comes from src/rag/retriever.py's retrieve(). HaluEval ships one
    knowledge string per example, so this just wraps it with a synthetic
    chunk_id.
    """
    return [{"chunk_id": f"ctx-{i}", "text": text} for i, text in enumerate(context)]


def predict_nli(example: BenchmarkExample) -> bool:
    """True = predicted hallucinated."""
    evidence = _as_evidence_chunks(example.context)
    claims = extract_claims(example.answer)
    return any(check_claim_entailment(c.text, evidence).label != EntailmentLabel.ENTAILMENT for c in claims)


def predict_embedding(example: BenchmarkExample) -> bool:
    """True = predicted hallucinated."""
    evidence = _as_evidence_chunks(example.context)
    claims = extract_claims(example.answer)
    return any(not check_claim_similarity(c.text, evidence).is_supported for c in claims)


def run(limit: int) -> None:
    print(f"Loading {limit} HaluEval source rows ({limit * 2} examples: supported + hallucinated pairs)...")
    examples = load_halueval(limit=limit)

    y_true: list[bool] = []
    y_pred_nli: list[bool] = []
    y_pred_embedding: list[bool] = []

    start = time.time()
    for i, example in enumerate(examples, start=1):
        y_true.append(example.is_hallucinated)
        y_pred_nli.append(predict_nli(example))
        y_pred_embedding.append(predict_embedding(example))
        if i % 20 == 0 or i == len(examples):
            print(f"  scored {i}/{len(examples)}")
    elapsed = time.time() - start

    nli_metrics = compute_classification_metrics(y_true, y_pred_nli)
    embedding_metrics = compute_classification_metrics(y_true, y_pred_embedding)

    print(f"\nScored {len(examples)} examples in {elapsed:.1f}s")
    print("\n" + "=" * 60)
    print(f"{'metric':<12}{'NLI checker':>16}{'embedding baseline':>22}")
    print("-" * 60)
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key:<12}{nli_metrics[key]:>16.3f}{embedding_metrics[key]:>22.3f}")
    print("=" * 60)
    print(
        "\nprecision/recall/f1 treat 'hallucinated' as the positive class:\n"
        "recall = share of true hallucinations the checker caught;\n"
        "precision = share of the checker's hallucination flags that were correct.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=get_config()["benchmark"]["sample_limit"],
        help="Number of HaluEval source rows to score (each yields 2 examples).",
    )
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
