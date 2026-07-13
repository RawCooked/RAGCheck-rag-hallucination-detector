"""STUB -- not implemented yet.

Purpose: the end-to-end evaluation script that will tie everything
together once claims/, verification/, and benchmark/datasets.py +
metrics.py are implemented:

    1. Load labeled examples (src/benchmark/datasets.py).
    2. For each example, extract claims from its answer
       (src/claims/extractor.py).
    3. Verify each claim against its context with both checkers
       (src/verification/nli_checker.py and embedding_checker.py).
    4. Score both checkers' predictions against ground truth
       (src/benchmark/metrics.py) and print a side-by-side comparison.

Left unimplemented deliberately -- there is nothing to run end-to-end
until the pieces above exist.
"""


def main() -> None:
    raise NotImplementedError("Benchmark runner is not implemented yet (Step 1 is scaffolding only).")


if __name__ == "__main__":
    main()
