"""Thin wrapper around a sentence-transformers model.

Kept as its own module (rather than inlined in vectorstore.py) because
verification/embedding_checker.py will need to embed claims with the same
model later -- both call sites should share one loaded model instance.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import get_config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Loads the model once per process; subsequent calls reuse it."""
    cfg = get_config()["embedding"]
    return SentenceTransformer(cfg["model_name"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Returns one embedding vector per input string, in order."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()
