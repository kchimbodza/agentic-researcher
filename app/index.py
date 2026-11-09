"""Vector index: chunk every page, embed with nomic-embed-text (Ollama), store in Chroma.

Chunking and embedding follow ../datasheet-answer-finder/ingest.py and search.py
(800-character line chunks, file name prefixed to each chunk, cosine distance).
The index rebuilds itself when the documents change.
"""

import hashlib
import json
import threading

import chromadb
import ollama

from . import docs

EMBED_MODEL = "nomic-embed-text"
CHUNK_CHARS = 800
_lock = threading.Lock()


def _db_dir():
    tag = hashlib.sha1(str(docs.DATA.resolve()).encode()).hexdigest()[:8]
    return docs.ROOT / "logs" / f"chroma-{tag}"


def _fingerprint() -> str:
    files = sorted((d["id"], docs.path_of(d["id"]).stat().st_mtime) for d in docs.list_documents())
    return hashlib.sha1(json.dumps(files).encode()).hexdigest()


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    # nomic-embed-text expects "search_document: " for stored text and "search_query: " for questions.
    return ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])["embeddings"]


def split_page(text: str) -> list[str]:
    """Group words into chunks of about CHUNK_CHARS characters (pages are already whitespace-normalised)."""
    chunks, current = [], ""
    for word in text.split(" "):
        if current and len(current) + len(word) > CHUNK_CHARS:
            chunks.append(current.strip())
            current = ""
        current += word + " "
    return [c for c in chunks + [current.strip()] if c]


def collection():
    """The Chroma collection, (re)built if the library changed since the last build."""
    with _lock:
        client = chromadb.PersistentClient(path=str(_db_dir()))
        fp = _fingerprint()
        col = client.get_or_create_collection("library", metadata={"hnsw:space": "cosine"})
        if (col.metadata or {}).get("fingerprint") == fp and col.count() > 0:
            return col
        client.delete_collection("library")
        col = client.create_collection("library", metadata={"hnsw:space": "cosine", "fingerprint": fp})
        ids, texts, metas = [], [], []
        for d in docs.list_documents():
            for p, text in enumerate(docs.pages_of(docs.path_of(d["id"])), start=1):
                for i, chunk in enumerate(split_page(text)):
                    ids.append(f"{d['id']}#p{p}c{i}")
                    texts.append(chunk)
                    metas.append({"doc": d["id"], "page": p})
        for s in range(0, len(ids), 64):
            names = [docs.name_of(m["doc"]) for m in metas[s:s + 64]]
            col.add(ids=ids[s:s + 64], documents=texts[s:s + 64], metadatas=metas[s:s + 64],
                    embeddings=embed([f"{n}\n{t}" for n, t in zip(names, texts[s:s + 64])], "search_document: "))
        return col


def vector_search(query: str, k: int = 5) -> list[dict]:
    col = collection()
    r = col.query(query_embeddings=embed([query], "search_query: "), n_results=min(k, col.count()))
    return [{"doc": m["doc"], "doc_name": docs.name_of(m["doc"]), "page": m["page"], "text": t,
             "distance": round(dist, 4)}
            for t, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
