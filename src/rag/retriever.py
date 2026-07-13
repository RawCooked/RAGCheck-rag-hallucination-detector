"""Question -> top-k relevant chunks.

Separated from vectorstore.py so callers (pipeline.py, and later
verification/ modules that need "what evidence backs this claim") depend
on a small, stable interface instead of Chroma's query API directly.
"""

from src.config import get_config
from src.rag.vectorstore import query_top_k


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Returns the top_k chunks most relevant to `question`.

    top_k falls back to config.yaml's retrieval.top_k when not given
    explicitly, so most call sites don't need to know the default.
    """
    if top_k is None:
        top_k = get_config()["retrieval"]["top_k"]
    return query_top_k(question, top_k)
