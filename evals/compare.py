"""Side-by-side table of result files, for the README.

    uv run python -m evals.compare baseline-main agent-main-rules agent-main-auto > evals/results/REPORT.md
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main(names: list[str]):
    data = [(n, json.loads((RESULTS / f"{n}.json").read_text())) for n in names]
    keys = [k for k in data[0][1]["summary"] if k not in ("System",)]
    for _, d in data[1:]:
        keys += [k for k in d["summary"] if k not in keys and k != "System"]
    print("| | " + " | ".join(n for n, _ in data) + " |")
    print("|---|" + "---|" * len(data))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(d["summary"].get(k, "–")) for _, d in data) + " |")
    print("\n## Per question\n")
    ids = list(dict.fromkeys(r["id"] for _, d in data for r in d["rows"]))
    print("| id | type | " + " | ".join(n for n, _ in data) + " |")
    print("|---|---|" + "---|" * len(data))
    for i in ids:
        cells, typ = [], ""
        for _, d in data:
            rows = [r for r in d["rows"] if r["id"] == i]
            typ = typ or (rows[0]["type"] if rows else "")
            cells.append(" ".join("✓" if r["correct"] else "✗" for r in rows) or "–")
        print(f"| {i} | {typ} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
