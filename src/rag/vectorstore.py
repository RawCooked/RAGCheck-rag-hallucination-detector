"""Persistent ChromaDB collection used to store and query chunk embeddings.

Chroma handles both the vector index and metadata storage, so this module
just wraps its client with the two operations the pipeline needs: build
(add chunks once) and query (top-k nearest chunks for a question).
"""

import chromadb

from src.config import get_config, PROJECT_ROOT
from src.rag.embedding import embed_texts
from src.rag.chunking import Chunk


def get_collection():
    cfg = get_config()["vectorstore"]
    persist_path = PROJECT_ROOT / cfg["persist_dir"]
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    return client.get_or_create_collection(name=cfg["collection_name"])


def is_populated() -> bool:
    return get_collection().count() > 0


def index_chunks(chunks: list[Chunk], batch_size: int = 256) -> None:
    """Embeds and stores chunks. Safe to call once; re-running against an
    already-populated collection would create duplicate IDs, so callers
    should check is_populated() first (see rag/pipeline.py).
    """
    collection = get_collection()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts)

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"doc_id": c.doc_id} for c in batch],
        )


def query_top_k(query_text: str, top_k: int) -> list[dict]:
    """Returns the top_k chunks most similar to query_text, each as
    {"chunk_id", "text", "doc_id", "distance"}.
    """
    collection = get_collection()
    query_embedding = embed_texts([query_text])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "distance": results["distances"][0][i],
            }
        )
    return hits
