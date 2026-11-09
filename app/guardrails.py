"""Deterministic guardrails used by the critic and the writer.

1. Prompt injection: document text that tries to instruct the agent is treated as hostile data.
2. Number grounding: every number in a claim or answer sentence must come from its quotes or the
   question, or be simple arithmetic on them (40 + 60 = 100 W is fine; an invented 999 mA is not).
3. Model codes: a claim naming "S16-PoE" is not supported by a quote about "S8-PoE".
"""

import itertools
import re

from .docs import squash

INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)"
    r"|disregard (the |your )?(previous|above|system|instructions)"
    r"|(reveal|print|show|repeat) (your|the) (system prompt|instructions|hidden)"
    r"|you are now|new instructions:|system prompt|as an ai (assistant|model),? you must"
    r"|(ai|llm|language model|assistant)s?[,:]? (must|should) (answer|say|respond|reply|state)"
    r"|do not (cite|mention|tell)",
    re.I,
)
# Numbers may be glued to words in PDF text ("at12.5Hz"); digits inside model codes (S24-PoE) are removed first.
NUM = re.compile(r"(?<![\d.])[-+]?\d[\d,]*(?:\.\d+)?")
CODE = re.compile(r"\b(?:[A-Z]{1,6}\d+[A-Z0-9]*(?:-[A-Za-z0-9]+)*|ESP32(?:-[A-Z0-9]+)?)\b")
# Board names whose digits are not measurements ("Raspberry Pi 4B", "Pi 4 Model B").
BOARD = re.compile(r"\bPi\s*\d+\s*(?:Model\s*)?[A-Z]?\b")
# Labels and citation years are references, not measurements ("Table 3", "Fig. 2", "Rezaei et al., 2023").
LABEL = re.compile(r"\b(?:Table|Tab\.|Fig(?:ure)?\.?|Section|Sec\.|Eq\.?|Ref\.?|ref\.?)\s*\[?[IVX\d]+\]?"
                   r"|et al\.?,?\s*\(?(?:19|20)\d\d[a-z]?\)?")


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text or ""))


def numbers(text: str) -> set[float]:
    out = set()
    text = LABEL.sub(" ", BOARD.sub(" ", CODE.sub(" ", re.sub(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]", " ", text or ""))))
    for m in NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 6))
        except ValueError:
            pass
    return out


def _derivable(n: float, pool: set[float]) -> bool:
    if n in pool or abs(n) < 2:  # 0/1 and tiny counts are not facts worth checking
        return True
    vals = list(pool)[:40]
    for a in vals:  # unit conversion of a single figure (0.8 mA = 800 µA)
        if any(abs(a * k - n) <= max(0.01, abs(n) * 0.01) for k in (1000, 0.001)):
            return True
    for a, b in itertools.permutations(vals, 2):  # arithmetic on two different figures
        for v in (a + b, a - b, a * b, a / b if b else None):
            # unit scale: figures in mA vs µA, W vs mW, ms vs s differ by 1000
            for scale in (1, 1000, 0.001):
                if v is not None and abs(v * scale - n) <= max(0.01, abs(n) * 0.01):
                    return True
    return False


def ungrounded_numbers(text: str, sources: list[str], ignore: set[float] = frozenset()) -> list[float]:
    """Numbers in text that cannot be traced to the sources (quotes + question).

    ignore: numbers that only identify a document (the year in "adebesin2026", the arXiv id
    2603.17069). A claim naming its source is not inventing a figure; they are not used for arithmetic.
    """
    pool = set().union(*(numbers(s) for s in sources)) if sources else set()
    return sorted(n for n in numbers(text) if n not in ignore and not _derivable(n, pool))


def codes(text: str) -> set[str]:
    return {c.upper() for c in CODE.findall(text or "")}


def _family(code: str) -> str:
    return re.match(r"[A-Z]*", code).group()


def unlabelled_page_codes(page_text: str, doc_text: str) -> set[str]:
    """Product names from elsewhere in the document that this page may stand for.

    Some spec pages show the product name only in an image (the Alta switch pages list "RJ45" but never
    "S8-PoE"). A code from the same document is allowed on such a page only if the page names no other
    code of the same family: a page that says S8-PoE still cannot support a claim about S16-PoE.
    """
    on_page = {_family(c) for c in codes(page_text)}
    return {c for c in codes(doc_text) if _family(c) not in on_page}


def unsupported_codes(claim: str, quote: str, context: str = "") -> set[str]:
    """Model/part codes the claim names that neither the quote nor the context (question, doc name) mentions."""
    have = codes(quote) | codes(context)
    sq = squash(quote + " " + context).upper()
    return {c for c in codes(claim) if c not in have and c.replace("-", "") not in sq.replace("-", "")}


# The same patterns with spaces removed, for text compared in squashed form (see docs.squash).
INJECTION_SQUASHED = re.compile(
    r"ignore(all|any|the)?(previous|prior|above|earlier)(instructions|rules|prompts?)"
    r"|disregard(the|your)?(previous|above|system|instructions)"
    r"|(reveal|print|show|repeat)(your|the)(systemprompt|instructions|hidden)"
    r"|youarenow|newinstructions:|systemprompt"
    r"|(ai|llm|languagemodel|assistant)s?[,:]?(must|should)(answer|say|respond|reply|state)"
    r"|notetoai")


def injection_near(page_text: str, quote: str, radius: int = 300) -> bool:
    """Is there an instruction aimed at the AI in the same stretch of the page as the quote?"""
    sp, sq = squash(page_text), squash(quote)
    i = sp.find(sq)
    if i < 0:
        return False
    return bool(INJECTION_SQUASHED.search(sp[max(0, i - radius): i + len(sq) + radius]))


def check_claim(claim: str, quote: str, context: str = "", names_context: str | None = None,
                ignore: set[float] = frozenset()) -> list[str]:
    """Reasons a (claim, quote) pair is not supported; empty list = passes the rule checks.

    context: text whose numbers may be used (the question). names_context: where product names
    must appear (the cited page + document name); defaults to context. ignore: document-id numbers.
    """
    reasons = []
    if looks_like_injection(quote):
        reasons.append("quote contains instructions aimed at the AI (possible prompt injection)")
    bad = ungrounded_numbers(claim, [quote, context], ignore)
    if bad:
        reasons.append("numbers not in the quote: " + ", ".join(f"{n:g}" for n in bad))
    wrong = unsupported_codes(claim, quote, context if names_context is None else names_context)
    if wrong:
        reasons.append("product/part names not on the cited page: " + ", ".join(sorted(wrong)))
    return reasons
