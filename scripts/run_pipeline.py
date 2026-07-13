"""End-to-end demo: question in -> retrieved chunks + generated answer out.

Run from the project root:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py "Was Abraham Lincoln the sixteenth President?"

The first run downloads the corpus, the embedding model, and the local
generation model, then builds the Chroma index -- this takes a minute or
two. Subsequent runs reuse the persisted index (data/chroma_db/).

Note: this script produces the (question, retrieved_docs, generated_answer)
triple that the hallucination detector (claims/ + verification/) will
eventually check -- it does not do any hallucination checking itself yet.
"""

import sys

# Windows terminals often default to a legacy codepage (e.g. cp1252) that
# can't print some characters from Wikipedia text (accents, en-dashes,
# etc.) -- force UTF-8 on stdout so those don't crash the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running as `python scripts/run_pipeline.py` without installing
# the package -- adds the project root to sys.path.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import ensure_index_built, answer_question

DEFAULT_QUESTION = "Was Abraham Lincoln the sixteenth President of the United States?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    ensure_index_built()
    result = answer_question(question)

    print("\n" + "=" * 70)
    print(f"QUESTION:\n  {result.question}")

    print(f"\nRETRIEVED CHUNKS ({len(result.retrieved_chunks)}):")
    for i, chunk in enumerate(result.retrieved_chunks, start=1):
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"  [{i}] (doc_id={chunk['doc_id']}, distance={chunk['distance']:.4f})")
        print(f"      {preview}...")

    print(f"\nGENERATED ANSWER:\n  {result.answer}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
