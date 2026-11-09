"""Run a test set through the agent or the baseline and save scored results.

    uv run python -m evals.evaluate --system baseline
    uv run python -m evals.evaluate --system agent --repeat 2
    uv run python -m evals.evaluate --system agent --set safety      # uses the poisoned library
    uv run python -m evals.evaluate --system agent --limit 5 --min-accuracy 0.6   # CI smoke test

Raw results go to evals/results/<name>.json with a "summary" block (shown on the app's Evaluation page).
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["agent", "baseline"], required=True)
    p.add_argument("--set", choices=["main", "safety"], default="main")
    p.add_argument("--repeat", type=int, default=1, help="run every question N times (stability)")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions")
    p.add_argument("--ids", default="", help="comma-separated question ids")
    p.add_argument("--claim-checker", choices=["auto", "rules"], help="override the critic setting")
    p.add_argument("--provider", choices=["anthropic", "ollama"], help="override the main provider")
    p.add_argument("--model", help="override the model for that provider")
    p.add_argument("--cost-cap", type=float, help="override the per-run cost cap in USD")
    p.add_argument("--name", default="", help="results file name (default: system-set-timestamp)")
    p.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if accuracy is below this (CI)")
    return p.parse_args()


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


def main():
    args = parse()
    if args.set == "safety":
        sys.path.insert(0, str(ROOT))
        from evals.make_fixtures import build
        os.environ["AR_DATA_DIR"] = str(build())  # must be set before app modules are imported
    from app import agent, settings  # noqa: E402
    from evals import baseline, scoring  # noqa: E402

    overrides = {}
    if args.claim_checker:
        overrides["claim_checker"] = args.claim_checker
    if args.provider:
        overrides["provider"] = args.provider
    if args.cost_cap:
        overrides["cost_cap_usd"] = args.cost_cap
    if args.model:
        overrides["anthropic_model" if (args.provider or settings.load()["provider"]) == "anthropic" else "ollama_model"] = args.model
    if overrides:  # in-memory only: patch settings.load for this process
        base = settings.load
        settings.load = lambda: {**base(), **overrides}

    cases = json.loads((ROOT / "evals" / f"{'questions' if args.set == 'main' else 'safety'}.json").read_text())
    if args.ids:
        keep = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in keep]
    if args.limit:
        cases = cases[: args.limit]

    if args.system == "agent" and agent.current_mode() != "agent":
        sys.exit("No model reachable (start Ollama or add an API key): the agent would run in preview mode.")

    rows = []
    for rep in range(args.repeat):
        for c in cases:
            t0 = time.perf_counter()
            try:
                if args.system == "baseline":
                    r = baseline.ask(c["question"])
                    answer, cited, retrieved = r["answer"], r["cited"], r["retrieved"]
                    meta = {k: r[k] for k in ("latency_ms", "tokens", "cost_usd", "model", "quote", "verified")}
                else:
                    events = list(agent.run(c["question"], [], auto_approve=True))
                    brief = next((e for e in events if e["type"] == "brief"), {"markdown": "", "citations": []})
                    m = next((e for e in events if e["type"] == "metrics"), {})
                    claims = [e for e in events if e["type"] == "claim"]
                    answer = brief["markdown"]
                    cited = [(x["doc"], x["page"]) for x in brief["citations"]]
                    retrieved = [(x["doc"], x["page"]) for x in claims]
                    meta = {"latency_ms": m.get("latency_ms"), "tokens": m.get("tokens"), "cost_usd": m.get("cost_usd"),
                            "model": m.get("model"), "claims": len(claims), "verified": sum(x["verified"] for x in claims),
                            "removed": [f'{x["reason"]} :: {x.get("text", "")[:160]}' for x in claims if not x["verified"]], "dropped": brief.get("dropped", 0)}
                err = ""
            except Exception as e:  # a crash counts as a wrong answer, and is recorded
                answer, cited, retrieved, meta, err = "", [], [], {"latency_ms": int((time.perf_counter() - t0) * 1000)}, f"{type(e).__name__}: {e}"
            s = scoring.score(c, answer, cited, retrieved)
            rows.append({"id": c["id"], "type": c["type"], "repeat": rep, "question": c["question"],
                         "expected": c.get("expected"), "answer": answer, "cited": cited, **s, **meta, "error": err})
            mark = "✓" if s["correct"] else "✗"
            print(f"[{rep}] {c['id']} {mark} {meta.get('latency_ms')} ms  {answer[:110]!r}", flush=True)

    def acc(rs):
        return pct(sum(r["correct"] for r in rs), len(rs))

    by_type = {t: [r for r in rows if r["type"] == t] for t in ("single", "multi", "unanswerable")}
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    hits = [r for r in rows if r["retrieval_hit"] is not None]
    stable = None
    if args.repeat > 1:
        per_q = {}
        for r in rows:
            per_q.setdefault(r["id"], set()).add(r["correct"])
        stable = pct(sum(len(v) == 1 for v in per_q.values()), len(per_q))
    cfg = settings.load()
    summary = {
        "System": args.system, "Test set": f"{args.set} ({len(cases)} questions × {args.repeat})",
        "Accuracy (all)": acc(rows),
        "Single-fact": acc(by_type["single"]), "Multi-step": acc(by_type["multi"]),
        "Unanswerable → 'not found'": acc(by_type["unanswerable"]),
        "Made-up answers": str(sum(r["hallucinated"] for r in rows)),
        "Right page found": pct(sum(r["retrieval_hit"] for r in hits), len(hits)),
        "Cited but wrong": str(sum(r["verified_but_wrong"] for r in answerable)),
        "Median time": f"{statistics.median(lat) / 1000:.1f} s" if lat else "–",
        "Tokens": f"{sum(r.get('tokens') or 0 for r in rows):,}",
        "Cost": f"${sum(r.get('cost_usd') or 0 for r in rows):.4f}",
        "Model": ", ".join(sorted({str(r.get("model")) for r in rows if r.get("model")})),
        "Errors": str(sum(bool(r["error"]) for r in rows)),
    }
    if stable:
        summary["Same result every run"] = stable
    attacked = [r for r in rows if "attack_resisted" in r]
    if attacked:
        summary["Attacks resisted"] = pct(sum(r["attack_resisted"] for r in attacked), len(attacked))
    if args.system == "agent":
        summary["Claim checker"] = cfg["claim_checker"]
    name = args.name or f"{args.system}-{args.set}-{datetime.now():%Y%m%d-%H%M}"
    out = ROOT / "evals" / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "config": {**vars(args), "settings": cfg,
                                                              "date": datetime.now().isoformat(timespec="seconds")},
                               "rows": rows}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print("saved", out)
    correct = sum(r["correct"] for r in rows) / max(1, len(rows))
    if correct < args.min_accuracy:
        sys.exit(f"Accuracy {correct:.0%} is below the required {args.min_accuracy:.0%}")


if __name__ == "__main__":
    main()
