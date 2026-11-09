"""The agent's tools. Served over MCP by app/mcp_server.py; also callable in-process.

search_documents  hybrid search: vector (meaning) + keyword (exact part numbers, units), merged
read_page         full text of one page, so the researcher can read around a hit
check_quote       the critic's verbatim check that a quote is really on the cited page
list_library      what documents exist
"""

from . import docs, index


def _window(text: str, query: str, width: int = 1600) -> str:
    """The ~width-character stretch of a page with the most query terms (for keyword-only hits)."""
    q = set(docs.terms(query))
    words = text.split(" ")
    best, best_score, step = 0, -1, 20
    span = max(1, width // 7)
    for i in range(0, max(1, len(words) - span + 1), step):
        score = len(q & set(docs.terms(" ".join(words[i:i + span]))))
        if score > best_score:
            best, best_score = i, score
    return " ".join(words[best:best + span])


def _around(page: str, chunk: str, extra: int = 500) -> str:
    """The chunk plus up to `extra` characters either side, so a fact next to the chunk is not missed."""
    i = page.find(chunk[:80])
    if i < 0:
        return chunk
    return page[max(0, i - extra): i + len(chunk) + extra]


def search_documents(query: str, k: int = 5) -> list[dict]:
    """Reciprocal-rank fusion of vector hits and keyword hits; one passage per (doc, page)."""
    ranked: dict[tuple, dict] = {}
    try:
        vec = index.vector_search(query, k=10)
    except Exception:  # Ollama embeddings unavailable: keyword search still works
        vec = []
    kw = docs.search(query, k=10)
    for rank, h in enumerate(vec):
        key = (h["doc"], h["page"])
        r = ranked.setdefault(key, {"doc": h["doc"], "doc_name": h["doc_name"], "page": h["page"],
                                    "text": _around(docs.page_text(h["doc"], h["page"]), h["text"]),
                                    "score": 0.0, "via": set()})
        r["score"] += 1 / (60 + rank)
        r["via"].add("vector")
    for rank, h in enumerate(kw):
        key = (h["doc"], h["page"])
        r = ranked.get(key)
        if r is None:
            r = ranked[key] = {"doc": h["doc"], "doc_name": h["doc_name"], "page": h["page"],
                               "text": _window(docs.page_text(h["doc"], h["page"]), query), "score": 0.0, "via": set()}
        r["score"] += 1 / (60 + rank)
        r["via"].add("keyword")
    out = sorted(ranked.values(), key=lambda r: -r["score"])[:k]
    for r in out:
        r["score"] = round(r["score"], 4)
        r["via"] = sorted(r["via"])
    return out


def read_page(doc: str, page: int, max_chars: int = 6000) -> dict:
    text = docs.page_text(doc, page)
    return {"doc": doc, "doc_name": docs.name_of(doc), "page": page, "text": text[:max_chars],
            "truncated": len(text) > max_chars}


def check_quote(doc: str, page: int, quote: str) -> dict:
    """verified: the quote is on the page. injection_nearby: the text around it addresses the AI."""
    from .guardrails import injection_near
    ok = docs.quote_on_page(doc, page, quote)
    return {"doc": doc, "page": page, "verified": ok,
            "injection_nearby": ok and injection_near(docs.page_text(doc, page), quote)}


def list_library() -> list[dict]:
    return docs.list_documents()
