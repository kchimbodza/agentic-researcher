"""Build the claim-checker training data.

Two sources:
  1. MultiNLI (nyu-mll/multi_nli on the Hugging Face Hub, parquet) turned into a
     binary task: entailment -> SUPPORTED (1), neutral/contradiction -> NOT_SUPPORTED (0).
     The pair is fed as (claim=hypothesis, quote=premise).
  2. Synthetic domain pairs built from the project's own PDFs (datasheets + papers).
     Sentences that contain numbers become quotes. Positives restate the sentence;
     negatives change a number, swap a unit, or swap a product / metric name.
     Domain splits are made by (document, page) so test pages are never seen in training.

Output: finetune/data/{train,validation,nli_validation,domain_test}.jsonl
Each line: {"claim", "quote", "label", "source", "kind", "doc", "page"}
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

SEED = 13
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "data"
PDF_DIRS = [REPO / "data" / "datasheets", REPO / "data" / "papers"]

NLI_TRAIN_PER_CLASS = 10_000  # 10k SUPPORTED + 10k NOT_SUPPORTED
NLI_VAL_PER_CLASS = 1_500

# ---------------------------------------------------------------- MNLI


def build_nli(rng: random.Random) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/multi_nli")

    def binarise(split, per_class):
        rows = [r for r in ds[split] if r["label"] in (0, 1, 2)]
        pos = [r for r in rows if r["label"] == 0]
        neg = [r for r in rows if r["label"] != 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        out = []
        for r in pos[:per_class] + neg[:per_class]:
            out.append(
                {
                    "claim": r["hypothesis"].strip(),
                    "quote": r["premise"].strip(),
                    "label": 1 if r["label"] == 0 else 0,
                    "source": "mnli",
                    "kind": ["entailment", "neutral", "contradiction"][r["label"]],
                    "doc": r["genre"],
                    "page": None,
                }
            )
        rng.shuffle(out)
        return out

    return binarise("train", NLI_TRAIN_PER_CLASS), binarise("validation_matched", NLI_VAL_PER_CLASS)


# ---------------------------------------------------------------- PDF sentences

HYPH = "[-‐‑‒–]"
BOILERPLATE = re.compile(
    r"(Espressif Systems|Submit Documentation Feedback|Series Datasheet v|Copyright|"
    r"www\.|https?://|doi\.org|arXiv|©|Edge Intell)",
    re.I,
)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def page_sentences(text: str) -> list[str]:
    """Join wrapped lines, split into sentences, keep ones that look like factual numeric statements."""
    text = clean(text)
    # de-hyphenate line wraps ("subject-\nindependent" keeps the hyphen; "connec-\ntions" is rare, ignore)
    text = re.sub(r"\s*\n\s*", " ", text)
    parts = re.split(r"(?<=[.;!?])\s+(?=[A-Z(•])|\s*•\s*", text)
    keep = []
    for s in parts:
        s = s.strip(" •")
        if not (40 <= len(s) <= 360):
            continue
        if not re.search(r"\d", s):
            continue
        if BOILERPLATE.search(s):
            continue
        letters = sum(c.isalpha() for c in s)
        if letters / len(s) < 0.5 or len(s.split()) < 7:
            continue
        keep.append(s)
    return keep


VALUE_UNITS = (
    r"Watts|watts|W|kW|mW|mA|µA|μA|uA|nA|A|ms|µs|μs|ns|s|MHz|GHz|kHz|Hz|Mbps|Gbps|Kbps|Mbit|KB|MB|GB|"
    r"V|mV|dBm|dBi|dB|°C|° C|nm|mm|cm|kg|mJ|J|%|min|h|hours"
)
# "<label words> <value> <unit>" at the start of a line, e.g. "POE Budget 40W" or "RTC timer + RTC memory 10 µA"
SPEC_ROW = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z /+&()\-‑]{2,60}?)\s+(?P<qual>up to |< ?|> ?|~ ?)?"
    r"(?P<num>\d+(?:\.\d+)?)\s?(?P<unit>" + VALUE_UNITS + r")(?![A-Za-z])"
)
# facts inside sentences, two reliable shapes only:
#   A: "... a PoE budget of up to 60 Watts"            -> metric = words before "of"
#   B: "... reaches 96.1% window-level accuracy, ..."   -> metric = words after the value (paper scores)
SENT_OF = re.compile(
    r"(?P<pre>(?:[A-Za-z][\w/+\-‑]*\s){1,6})of (?P<qual>(?:up to|less than|more than|at least|about|approximately|around) )?"
    r"(?P<num>\d+(?:\.\d+)?)\s?(?P<unit>" + VALUE_UNITS + r")(?![A-Za-z\d])"
)
SENT_SCORE = re.compile(
    r"(?<![\d.])(?P<num>\d+(?:\.\d+)?)\s?(?P<unit>%)\s(?P<post>(?:[A-Za-z][\w\-]*\s?){1,3})"
)
SCORE_WORDS = {"accuracy", "precision", "recall", "f1", "score", "auc", "specificity", "sensitivity", "error", "loss"}
STOP = {
    "a", "an", "the", "of", "is", "are", "was", "were", "at", "to", "from", "with", "by", "for", "in", "on", "and",
    "or", "as", "than", "that", "which", "its", "it", "has", "have", "be", "can", "reaches", "raises", "reduces",
    "lowers", "increases", "offers", "provides", "supports", "up", "only", "about", "approximately", "per", "this",
    "into", "under", "over", "between", "each", "every", "around", "within", "we", "our", "also", "while",
}
PRODUCTS = [
    f"S8{HYPH}PoE", f"S16{HYPH}PoE", f"S24{HYPH}PoE", "Route10", "AP6 Pro",
    f"ESP32{HYPH}D0WD{HYPH}V3", f"ESP32{HYPH}D0WDR2{HYPH}V3", f"ESP32{HYPH}U4WDH", f"ESP32{HYPH}S0WD",
    r"ESP32(?![\-‑A-Za-z0-9])",
]
DOC_SUBJECT = {
    "Alta_Labs_Switches_Datasheet.pdf": "the Alta Labs switch",
    "Route10_Datasheet.pdf": "the Route10",
    "esp32_datasheet_en.pdf": "the ESP32",
    "adebesin2026_wearable_fall_detection_lora.pdf": "the wearable fall-detection system",
    "bathroom_fall_mmwave_arxiv2603.17069.pdf": "the proposed fall-detection method",
}


def find_product(text: str) -> str | None:
    for pat in PRODUCTS:
        m = re.search(r"(?<![A-Za-z0-9])" + pat, text)
        if m:
            return m.group(0).replace("‑", "-")
    return None


def tail_metric(words: str, max_words: int = 3) -> str | None:
    """Trailing run of content words before "of": "with a PoE budget" -> "PoE budget"."""
    toks = re.findall(r"[A-Za-z][\w/+\-‑]*", words)
    keep = []
    for t in reversed(toks):
        if t.lower() in STOP or (t.endswith(("ing", "ed", "s")) and t.lower() in {"ranges", "using", "used"}):
            break
        if t.endswith(("ing", "ed")) and t.lower() not in {"setting", "spacing", "rating", "speed"}:
            break
        keep.insert(0, t)
        if len(keep) == max_words:
            break
    return clean_metric(" ".join(keep)) if keep else None


def clean_metric(words: str, max_words: int = 5) -> str | None:
    toks = re.findall(r"[A-Za-z][\w/+\-‑]*", words)
    while toks and toks[0].lower() in STOP:
        toks.pop(0)
    while toks and toks[-1].lower() in STOP:
        toks.pop()
    toks = toks[-max_words:]
    while toks and toks[0].lower() in STOP:
        toks.pop(0)
    if not toks or any(t.lower() in STOP - {"of", "and", "per"} for t in toks):
        return None
    if toks[-1].endswith(("ing", "ed")) and toks[-1].lower() not in {"setting", "spacing", "rating"}:
        return None
    # "Idle Power Consumption" -> "idle power consumption", keep acronyms ("PoE", "RTC")
    toks = [t.lower() if t[:1].isupper() and t[1:].islower() and len(t) > 1 else t for t in toks]
    metric = " ".join(toks).replace("‑", "-")
    return metric if 3 <= len(metric) <= 45 else None


def spec_rows(text: str) -> list[dict]:
    """Spec-table lines: one fact each (label, value, unit). Quote = the line (sometimes with the next line)."""
    lines = [clean(l) for l in clean(text).splitlines() if l.strip()]
    out = []
    for i, line in enumerate(lines):
        m = SPEC_ROW.match(line)
        if not m or BOILERPLATE.search(line) or len(line) > 160:
            continue
        metric = clean_metric(m.group("label"))
        if not metric:
            continue
        out.append({"line": line, "next": lines[i + 1] if i + 1 < len(lines) else "", "metric": metric,
                    "qual": (m.group("qual") or "").strip(), "num": m.group("num"), "unit": m.group("unit")})
    return out


def sentence_facts(s: str) -> list[dict]:
    facts = []
    for m in SENT_OF.finditer(s):
        metric = tail_metric(m.group("pre"))
        if metric:
            facts.append({"metric": metric, "qual": (m.group("qual") or "").strip(), "num": m.group("num"), "unit": m.group("unit")})
    for m in SENT_SCORE.finditer(s):
        words = re.split(r"[,;.()]", m.group("post"))[0].split()
        keep = []
        for w in words:
            if w.lower() in STOP:
                break
            keep.append(w)
        if keep and keep[-1].lower() in SCORE_WORDS:
            facts.append({"metric": " ".join(keep), "qual": "", "num": m.group("num"), "unit": "%"})
    return facts[:3]


def read_pdfs() -> dict[tuple[str, int], dict]:
    pages: dict[tuple[str, int], dict] = {}
    for d in PDF_DIRS:
        for pdf in sorted(d.glob("*.pdf")):
            reader = PdfReader(str(pdf))
            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    sents = page_sentences(text)
                    rows = spec_rows(text)
                except Exception:  # noqa: BLE001 - a broken page should not stop the build
                    continue
                if sents or rows:
                    pages[(pdf.name, i)] = {"sents": sents, "rows": rows, "product": find_product(text)}
    return pages


# ---------------------------------------------------------------- perturbations

NUM_RE = re.compile(r"(?<![A-Za-z_\d.‑-])(\d+(?:[.,]\d+)?)(?![\d]|\.\d|[a-z]{2,}|[A-Z]\w)")

UNIT_SWAPS = {
    "mA": ["µA", "A"], "µA": ["mA", "nA"], "μA": ["mA", "nA"], "uA": ["mA"], "nA": ["µA"],
    "W": ["kW", "mW"], "mW": ["W", "µW"], "kW": ["W"], "Watts": ["Kilowatts", "Milliwatts"],
    "ms": ["s", "µs"], "µs": ["ms", "ns"], "μs": ["ms", "ns"], "ns": ["µs", "ms"], "s": ["ms", "min"],
    "MHz": ["GHz", "kHz"], "GHz": ["MHz"], "kHz": ["MHz", "Hz"], "Hz": ["kHz"],
    "Mbps": ["Gbps", "Kbps"], "Gbps": ["Mbps"], "Kbps": ["Mbps"],
    "KB": ["MB"], "MB": ["KB", "GB"], "GB": ["MB", "TB"], "Mbit": ["Gbit"],
    "V": ["mV", "kV"], "mV": ["V"], "dBm": ["dB", "mW"],
    "°C": ["°F"], "nm": ["µm"], "mm": ["cm", "m"], "cm": ["mm", "m"],
    "mJ": ["J", "µJ"], "J": ["mJ"], "%": [" percentage points", "‰"],
    "min": ["s", "h"], "h": ["min"], "dB": ["dBm"],
    # spelled-out forms used by the reworded claims
    "watts": ["kilowatts", "milliwatts"], "milliamps": ["microamps", "amps"], "microamps": ["milliamps", "nanoamps"],
    "milliseconds": ["seconds", "microseconds"], "seconds": ["milliseconds", "minutes"],
    "millijoules": ["joules", "microjoules"], "volts": ["millivolts", "kilovolts"], "Gb/s": ["Mb/s"], "Mb/s": ["Gb/s", "Kb/s"],
    "percent": ["percentage points"], "kg": ["g", "lbs"], "nanometres": ["micrometres"], "kilobytes": ["megabytes"],
    "hours": ["minutes", "days"], "dBi": ["dBm"], "° C": ["° F"],
}
UNIT_RE = re.compile(
    r"(?<=\d)(\s?)(" + "|".join(sorted((re.escape(u) for u in UNIT_SWAPS), key=len, reverse=True)) + r")(?![A-Za-z])"
)

ENTITY_GROUPS = [
    [f"S8{HYPH}PoE", f"S16{HYPH}PoE", f"S24{HYPH}PoE"],
    ["Route10", "AP6 Pro", "S8-PoE"],
    [f"ESP32{HYPH}D0WD{HYPH}V3", f"ESP32{HYPH}D0WDR2{HYPH}V3", f"ESP32{HYPH}U4WDH", f"ESP32{HYPH}S0WD", f"ESP32{HYPH}C3", f"ESP32{HYPH}S3",
     r"ESP32(?![\-‑A-Za-z0-9])"],
    ["Raspberry Pi 4B", "Raspberry Pi 5", "Jetson Nano"],
    ["LoRa", "Wi-Fi", "Bluetooth"],
    ["precision", "recall", "accuracy", "F1 score", "specificity"],
    ["SFP\\+", "SFP(?!\\+)"],
    ["mmWave radar", "floor vibration"],
    ["MLP", "ANN", "CNN", "LSTM"],
    ["minimum", "maximum"],
    ["Min", "Max"],
    ["input", "output"],
    ["transmit", "receive"],
]


def _plain(pattern: str) -> str:
    return re.sub(r"\(\?<?[!=][^)]*\)", "", pattern.replace("\\+", "+").replace(HYPH, "-"))


def change_number(s: str, rng: random.Random) -> str | None:
    nums = list(NUM_RE.finditer(s))
    # skip years and section-like numbers
    def ok(m):
        before = s[max(0, m.start() - 9): m.start()]
        if re.fullmatch(r"(19|20)\d\d", m.group(1)) or m.start() < 3:
            return False
        if re.search(r"(Section|Table|Figure|Fig\.|Chapter|Note|of|v|\d\.)\s?$", before):
            return False
        return not re.match(r"\.\s?\d", s[m.end():])  # section numbers like "4.8. 1"
    nums = [m for m in nums if ok(m)]
    with_unit = [m for m in nums if UNIT_RE.match(s, m.end() - 1)]
    nums = with_unit or nums
    if not nums:
        return None
    m = rng.choice(nums)
    raw = m.group(1)
    dec = "," if "," in raw and "." not in raw else "."
    val = float(raw.replace(",", "."))
    places = len(raw.split(dec)[1]) if dec in raw else 0
    for _ in range(10):
        op = rng.choice(["x2", "half", "plus", "minus", "x10", "digit"])
        if op == "x2":
            new = val * 2
        elif op == "half":
            new = val / 2
        elif op == "plus":
            new = val + rng.choice([1, 2, 5, 10]) * (10 ** -places if places else 1)
        elif op == "minus":
            new = val - rng.choice([1, 2, 5]) * (10 ** -places if places else 1)
        elif op == "x10":
            new = val * 10
        else:
            new = val + (rng.randint(1, 9) * (10 ** -places if places else 1))
        if new < 0:
            continue
        txt = f"{new:.{places}f}" if places else (str(int(round(new))) if abs(new - round(new)) < 1e-9 else f"{new:.1f}")
        if dec == ",":
            txt = txt.replace(".", ",")
        if txt != raw:
            return s[: m.start(1)] + txt + s[m.end(1):]
    return None


def swap_unit(s: str, rng: random.Random) -> str | None:
    ms = list(UNIT_RE.finditer(s))
    if not ms:
        return None
    m = rng.choice(ms)
    new = rng.choice(UNIT_SWAPS[m.group(2)])
    sp = "" if new.startswith(" ") else m.group(1)
    return s[: m.start(1)] + sp + new + s[m.end(2):]


def swap_entity(s: str, rng: random.Random) -> str | None:
    options = []
    for group in ENTITY_GROUPS:
        for pat in group:
            for m in re.finditer(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", s):
                others = [_plain(p) for p in group if p != pat]
                options.append((m, others))
    if not options:
        return None
    m, others = rng.choice(options)
    return s[: m.start()] + rng.choice(others) + s[m.end():]


POS_TEMPLATES = [
    "{s}",
    "{s}",
    "According to the document, {l}",
    "The document states that {l}",
    "The source says {l}",
    "Per the datasheet, {l}",
]


def lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s and not (len(s) > 1 and s[1].isupper()) and not re.match(r"[A-Z]+\d", s) else s


def positive_claim(s: str, rng: random.Random) -> tuple[str, str]:
    r = rng.random()
    # light abbreviation paraphrase keeps the same facts
    abbrev = [(r"\bWatts\b", "W"), (r"\bmilliseconds\b", "ms"), (r"\bpercent\b", "%"), (r"\bup to\b", "a maximum of")]
    if r < 0.2:
        for pat, rep in abbrev:
            if re.search(pat, s):
                return re.sub(pat, rep, s, count=1), "pos_abbrev"
    t = rng.choice(POS_TEMPLATES)
    return t.format(s=s, l=lower_first(s)), "pos_template" if t != "{s}" else "pos_verbatim"


# ---------------------------------------------------------------- reworded claims

UNIT_SPELLINGS = {
    "W": ["W", "Watts", "watts"], "Watts": ["W", "Watts", "watts"], "watts": ["W", "watts"],
    "mA": ["mA", "milliamps"], "µA": ["µA", "uA", "microamps"], "μA": ["µA", "uA", "microamps"], "uA": ["µA", "uA"],
    "ms": ["ms", "milliseconds"], "s": ["s", "seconds"], "mJ": ["mJ", "millijoules"], "V": ["V", "volts"],
    "Gbps": ["Gbps", "Gb/s"], "Mbps": ["Mbps", "Mb/s"], "%": ["%", " percent"], "nm": ["nm", "nanometres"],
    "KB": ["KB", "kilobytes"], "hours": ["hours", "h"], "h": ["h", "hours"],
}

REWORD_TEMPLATES = [
    "{S} has a {V} {M}.",
    "The {M} of {s} is {V}.",
    "{S}'s {M} is {V}.",
    "{S} has {M} of {V}.",
    "{S} supports {M} of {V}.",
    "{M_cap} for {s}: {V}.",
    "For {s}, the {M} is {V}.",
    "The document gives {V} as the {M} of {s}.",
]


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def subject_phrase(product: str | None, doc: str) -> str:
    return f"the {product}" if product else DOC_SUBJECT.get(doc, "the device")


def reword(fact: dict, subject: str, rng: random.Random, template: str | None = None) -> str:
    unit = rng.choice(UNIT_SPELLINGS.get(fact["unit"], [fact["unit"]]))
    sep = "" if unit.startswith(" ") or unit == "%" or (unit in ("W", "V") and rng.random() < 0.4) else " "
    qual = {"<": "less than", ">": "more than", "~": "about"}.get(fact["qual"], fact["qual"])
    value = (qual + " " if qual else "") + fact["num"] + (unit if unit.startswith(" ") else sep + unit)
    t = template or rng.choice(REWORD_TEMPLATES)
    if fact["qual"] and t == "{S} has a {V} {M}.":
        t = "The {M} of {s} is {V}."  # "has a less than 1 mA ..." reads badly
    return t.format(S=cap(subject), s=subject, M=fact["metric"], M_cap=cap(fact["metric"]), V=value)


def swap_metric(claim: str, fact: dict, others: list[dict], rng: random.Random) -> str | None:
    cands = [o for o in others if o["metric"].lower() != fact["metric"].lower() and o["num"] != fact["num"]]
    if not cands or fact["metric"] not in claim:
        return None
    return claim.replace(fact["metric"], rng.choice(cands)["metric"], 1)


def make_quote(sents: list[str], i: int, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return sents[i]
    lo = max(0, i - rng.randint(0, 1))
    hi = min(len(sents), i + 1 + rng.randint(0, 1))
    return " ".join(sents[lo:hi])


def page_split(pages: dict, rng: random.Random) -> dict[tuple[str, int], str]:
    """Split by page, per document. Pages with sentences use the run-1 procedure (same rng order), so the
    domain test pages are identical to run 1. Pages that only have spec-table rows (new in run 2) get a
    split from a per-page seed."""
    split_of: dict[tuple[str, int], str] = {}
    keys = sorted(k for k, v in pages.items() if v["sents"])
    for doc in sorted({k[0] for k in keys}):
        dk = [k for k in keys if k[0] == doc]
        rng.shuffle(dk)
        n = len(dk)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.1)) if n >= 4 else 0
        for j, k in enumerate(dk):
            split_of[k] = "test" if j < n_test else ("validation" if j < n_test + n_val else "train")
    for k in sorted(pages):
        if k not in split_of:
            r = random.Random(f"{SEED}:{k[0]}:{k[1]}").random()
            split_of[k] = "test" if r < 0.2 else ("validation" if r < 0.3 else "train")
    return split_of


def reworded_pairs(fact, subject, quotes, page_facts, rng) -> list[dict]:
    """Reworded positives + negatives made from the reworded claim (number / unit / name / metric swap)."""
    templates = rng.sample(REWORD_TEMPLATES, k=len(REWORD_TEMPLATES))
    pos = reword(fact, subject, rng, templates[0])
    quote = rng.choice(quotes)
    negs = []
    for name, fn in [("neg_rw_number", change_number), ("neg_rw_unit", swap_unit)]:
        c = fn(pos, rng)
        if c and c != pos:
            negs.append((name, c))
    product = subject[4:] if subject.startswith("the ") else subject
    if product.lower() in quote.lower():  # a name swap is only checkable if the quote names the product
        c = swap_entity(pos, rng)
        if c and c != pos:
            negs.append(("neg_rw_entity", c))
    c = swap_metric(pos, fact, page_facts, rng)
    if c:
        negs.append(("neg_rw_metric", c))
    rows = [{"claim": pos, "quote": quote, "label": 1, "kind": "pos_reworded"}]
    for t in templates[1 : len(negs)]:  # one positive per negative keeps the classes balanced
        rows.append({"claim": reword(fact, subject, rng, t), "quote": rng.choice(quotes), "label": 1, "kind": "pos_reworded"})
    rows += [{"claim": c, "quote": quote if n == "neg_rw_entity" else rng.choice(quotes), "label": 0, "kind": n} for n, c in negs]
    return rows


def build_domain(rng: random.Random) -> dict[str, list[dict]]:
    pages = read_pdfs()
    keys = sorted(pages)
    split_of = page_split(pages, rng)

    out: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    perturb = [("neg_number", change_number), ("neg_unit", swap_unit), ("neg_entity", swap_entity)]
    for key in keys:
        sents = pages[key]["sents"]
        page_product = pages[key]["product"]
        base = {"source": "domain", "doc": key[0], "page": key[1]}
        rounds = 2 if split_of[key] == "train" else 1
        seen = set()

        def add(r):
            if (r["claim"], r["quote"]) not in seen:
                seen.add((r["claim"], r["quote"]))
                out[split_of[key]].append({**base, **r})

        # (1) run-1 style pairs: the sentence itself / with a lead-in, and edited copies
        for i, s in enumerate(sents):
            for _ in range(rounds):
                negs = []
                for name, fn in perturb:
                    c = fn(s, rng)
                    if c and c != s:
                        if rng.random() < 0.3:
                            c = "According to the document, " + lower_first(c)
                        negs.append((name, c))
                rows = [{"claim": c, "label": 0, "kind": n} for n, c in negs]
                for _ in range(max(1, len(negs))):
                    c, k = positive_claim(s, rng)
                    rows.append({"claim": c, "label": 1, "kind": k})
                for r in rows:
                    r["quote"] = make_quote(sents, i, rng)
                    add(r)

        # (2) reworded claims: "The S8-PoE has a 60 W PoE budget." from sentences and spec-table rows
        page_facts = [f for s in sents for f in sentence_facts(s)] + pages[key]["rows"]
        for i, s in enumerate(sents):
            subject = subject_phrase(find_product(s) or page_product, key[0])
            for fact in sentence_facts(s):
                for _ in range(rounds):
                    quotes = [s, make_quote(sents, i, rng)]
                    for r in reworded_pairs(fact, subject, quotes, page_facts, rng):
                        add(r)
        for row in pages[key]["rows"]:
            subject = subject_phrase(page_product, key[0])
            quotes = [row["line"], (row["line"] + " " + row["next"]).strip()]
            if page_product and page_product.lower() not in row["line"].lower():
                quotes.append(f"{page_product} ... {row['line']}")
            for _ in range(rounds):
                for r in reworded_pairs(row, subject, quotes, page_facts, rng):
                    add(r)
    for v in out.values():
        rng.shuffle(v)
    return out


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    domain = build_domain(rng)
    nli_train, nli_val = build_nli(rng)

    train = nli_train + domain["train"]
    rng.shuffle(train)
    validation = domain["validation"]
    write("train", train)
    write("validation", validation)  # domain validation (used for model selection)
    write("nli_validation", nli_val)
    write("domain_test", domain["test"])

    stats = {}
    for name, rows in [("train", train), ("validation", validation), ("nli_validation", nli_val), ("domain_test", domain["test"])]:
        stats[name] = {
            "n": len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "sources": dict(Counter(r["source"] for r in rows)),
            "kinds": dict(Counter(r["kind"] for r in rows if r["source"] == "domain")),
        }
    test_pages = sorted({(r["doc"], r["page"]) for r in domain["test"]})
    train_pages = {(r["doc"], r["page"]) for r in domain["train"]}
    assert not (set(test_pages) & train_pages), "test page leaked into train"
    stats["domain_test_pages"] = len(test_pages)
    stats["domain_train_pages"] = len(train_pages)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
