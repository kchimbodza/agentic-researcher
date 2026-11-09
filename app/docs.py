"""Load PDFs page by page, keyword-search them, and check that quotes really exist.

These are the low-level document tools. app/tools.py combines them with vector search,
and app/mcp_server.py serves them over MCP to the agent.
"""

import math
import os
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
# AR_DATA_DIR lets the safety eval run against a copy of the library with a test document added.
DATA = Path(os.environ.get("AR_DATA_DIR", ROOT / "data"))
KINDS = {"datasheets": "Datasheet", "papers": "Paper"}

STOP = set("""a an and are as at be by can could did do does for from has have how i if in into is it its
of on or so than that the their them then there these this to was we what when where which who why will
with would you your about any both each more most much other over same some such only also between
paper papers datasheet says say tell me show compare""".split())

# PDF text uses look-alike characters (non-breaking hyphens, a "⚶" minus in the ESP32 datasheet).
_FIX = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                      "⚶": "-", " ": " ", "‘": "'", "’": "'", "“": '"', "”": '"',
                      "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi"})

_cache: dict[str, tuple[float, list[str]]] = {}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_FIX)).strip()


def squash(text: str) -> str:
    """Comparison form: lower case, no whitespace (PDFs break lines mid-word; models re-join them)."""
    return "".join(text.translate(_FIX).lower().split()).strip(".\"'")


def list_documents() -> list[dict]:
    docs = []
    for folder, kind in KINDS.items():
        for pdf in sorted((DATA / folder).glob("*.pdf")):
            docs.append({
                "id": f"{folder}/{pdf.name}",
                "name": pdf.stem.replace("_", " "),
                "kind": kind,
                "pages": len(pages_of(pdf)),
                "size_kb": pdf.stat().st_size // 1024,
            })
    return docs


def id_numbers() -> set[float]:
    """Numbers inside document file names (years, arXiv ids): they name a source, they are not figures."""
    from .guardrails import numbers
    out = set()
    for folder in KINDS:
        for pdf in (DATA / folder).glob("*.pdf"):
            out |= numbers(pdf.stem.replace("_", " "))
    return out


def doc_text(doc_id: str) -> str:
    """All pages of one document (cached through pages_of)."""
    return " ".join(pages_of(path_of(doc_id)))


def path_of(doc_id: str) -> Path:
    """Map 'papers/x.pdf' to a file, refusing anything outside the two library folders."""
    folder, _, name = doc_id.partition("/")
    if folder not in KINDS or not name or name != Path(name).name or name.startswith("."):
        raise FileNotFoundError(doc_id)
    p = DATA / folder / name
    if not p.is_file():
        raise FileNotFoundError(doc_id)
    return p


def name_of(doc_id: str) -> str:
    return Path(doc_id).stem.replace("_", " ")


def pages_of(pdf: Path) -> list[str]:
    key, mtime = str(pdf), pdf.stat().st_mtime
    if key not in _cache or _cache[key][0] != mtime:
        reader = PdfReader(pdf)
        _cache[key] = (mtime, [clean(p.extract_text() or "") for p in reader.pages])
    return _cache[key][1]


def page_text(doc_id: str, page: int) -> str:
    pages = pages_of(path_of(doc_id))
    if not 1 <= page <= len(pages):
        raise IndexError(f"{doc_id} has {len(pages)} pages")
    return pages[page - 1]


def terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|[a-z0-9]", text.translate(_FIX).lower())
    return [w for w in words if w not in STOP and len(w) > 1]


def search(query: str, k: int = 4) -> list[dict]:
    """Rank every page of every document by keyword overlap (TF-IDF style)."""
    q = set(terms(query))
    if not q:
        return []
    pages = []
    for d in list_documents():
        for i, text in enumerate(pages_of(path_of(d["id"]))):
            pages.append((d, i + 1, text, terms(text)))
    n = len(pages) or 1
    df = {t: sum(1 for p in pages if t in p[3]) for t in q}
    hits = []
    for d, page, text, toks in pages:
        if not toks:
            continue
        score = sum(math.log(1 + toks.count(t)) * math.log(1 + n / (1 + df[t])) for t in q if t in toks)
        if score > 0:
            hits.append({"doc": d["id"], "doc_name": d["name"], "kind": d["kind"], "page": page,
                         "score": round(score, 2), "quote": best_quote(text, q)})
    hits.sort(key=lambda h: -h["score"])
    return hits[:k]


def best_quote(text: str, q: set[str], width: int = 260) -> str:
    """The sentence-sized window on the page that contains the most query terms."""
    sentences = re.split(r"(?<=[.;:!?])\s+", text)
    best, best_score = "", -1
    for i in range(len(sentences)):
        window = " ".join(sentences[i:i + 2])[:width]
        score = len(q & set(terms(window)))
        if score > best_score:
            best, best_score = window, score
    return best.strip()


def quote_on_page(doc_id: str, page: int, quote: str) -> bool:
    """The critic's check: does this exact quote appear on the cited page (ignoring spacing)?"""
    try:
        text = page_text(doc_id, page)
    except (FileNotFoundError, IndexError, ValueError):
        return False
    q = squash(quote)
    return len(q) >= 8 and q in squash(text)
