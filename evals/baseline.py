"""Baseline: the single-step RAG from ../datasheet-answer-finder (ask.py), unchanged in method.

One vector search (top 3 chunks) -> one local-model call that must answer with one verbatim quote
-> the quote is checked against the retrieved text. No planning, no multi-step search, no critic.
Uses the same chunking/embeddings as the agent (app/index.py) so the comparison isolates the agent.
"""

import json
import time

import ollama

from app import docs, index, settings

# Prompt copied from ../datasheet-answer-finder/ask.py
PROMPT = """Answer the question using ONLY the datasheet pages below.
Reply in JSON with three keys:
  "answer": a short answer,
  "quote": one sentence copied word for word from the pages that supports the answer,
  "page_id": the id of the page the quote came from (for example "demo-mcu32.pdf p.4").
If the pages do not contain the answer, set "answer" to "not found" and "quote" to "".

{pages}

Question: {question}"""


def ask(question: str, k: int = 3) -> dict:
    t0 = time.perf_counter()
    hits = index.vector_search(question, k=k)
    pages = "\n\n".join(f"[{h['doc'].split('/')[-1]} p.{h['page']}]\n{h['text']}" for h in hits)
    model = settings.load()["ollama_model"]
    r = ollama.chat(model=model, format="json", options={"temperature": 0, "num_ctx": 8192},
                    messages=[{"role": "user", "content": PROMPT.format(pages=pages, question=question)}])
    reply = json.loads(r["message"]["content"])
    answer, quote = str(reply.get("answer", "")), str(reply.get("quote", ""))
    source = next((h for h in hits if len(docs.squash(quote)) >= 8 and docs.squash(quote) in docs.squash(h["text"])), None)
    return {"answer": answer, "quote": quote, "verified": source is not None,
            "cited": [(source["doc"], source["page"])] if source else [],
            "retrieved": [(h["doc"], h["page"]) for h in hits],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "tokens": (r.get("prompt_eval_count") or 0) + (r.get("eval_count") or 0), "cost_usd": 0.0, "model": model}
