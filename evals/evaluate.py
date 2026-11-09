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
    p.add_argument("--set", choices=["main", "safety", "heldout"], default="main")
    p.add_argument("--repeat", type=int, default=1, help="run every question N times (stability)")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions")
    p.add_argument("--ids", default="", help="comma-separated question ids")
    p.add_argument("--claim-checker", choices=["auto", "rules"], help="override the critic setting")
    p.add_argument("--provider", choices=["anthropic", "ollama"], help="override the main provider")
    p.add_argument("--model", help="override the model for that provider")
    p.add_argument("--cost-cap", type=float, help="override the per-run cost cap in USD")
    p.add_argument("--name", default="", help="results file name (default: system-set-timestamp)")
    p.add_argument("--allow-fallback", action="store_true",
                   help="let the router fall back to another model (off by default: a fallback would mix models in one score)")
    p.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if accuracy is below this (CI)")
    return p.parse_args()


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


MODEL_FAILED = "No model could answer"
MAX_MODEL_FAILURES = 3  # in a row: stop instead of scoring a dead API (e.g. credit ran out)


def invalid_reason(rows: list[dict], expected_model: str | None) -> str:
    """Why a run cannot be reported as a result for expected_model ('' = valid).

    Every answer must come from the requested model alone: a router fallback (to a local model when
    the API fails) or a model error would silently mix models into one score.
    """
    failed = sum(MODEL_FAILED in (r.get("error") or "") for r in rows)
    other = [r["id"] for r in rows if r.get("model") and r["model"] not in (expected_model, "none")]
    reasons = []
    if failed:
        reasons.append(f"{failed} answer(s) had no model available")
    if other:
        reasons.append(f"{len(other)} answer(s) used a model other than {expected_model} ({', '.join(other[:5])})")
    return "; ".join(reasons)


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

    expected_model = None
    if args.system == "agent":
        from app import llm  # noqa: E402
        cfg0 = settings.load()
        main_provider = llm.route()[0] if llm.route() else (None, None)
        if main_provider[0] != cfg0["provider"]:
            sys.exit(f"The main provider '{cfg0['provider']}' is not available (no API key?); refusing to "
                     f"evaluate '{main_provider[0]}' under its name.")
        expected_model = main_provider[1]
        if not args.allow_fallback:  # strict: only the requested model may answer
            full_route = llm.route
            llm.route = lambda: full_route()[:1]

    cases = json.loads((ROOT / "evals" / {"main": "questions.json", "safety": "safety.json", "heldout": "heldout.json"}[args.set]).read_text())
    if args.ids:
        keep = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in keep]
    if args.limit:
        cases = cases[: args.limit]

    if args.system == "agent" and agent.current_mode() != "agent":
        sys.exit("No model reachable (start Ollama or add an API key): the agent would run in preview mode.")

    rows, failures_in_row, stopped = [], 0, ""
    for rep in range(args.repeat):
        if stopped:
            break
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
            failures_in_row = failures_in_row + 1 if MODEL_FAILED in err else 0
            if failures_in_row >= MAX_MODEL_FAILURES:
                stopped = f"stopped after {failures_in_row} model failures in a row: {err[:300]}"
                print(stopped, flush=True)
                break

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
    invalid = "; ".join(x for x in (stopped, invalid_reason(rows, expected_model)) if x) if args.system == "agent" else ""
    if invalid:
        summary["INVALID"] = invalid
    name = args.name or f"{args.system}-{args.set}-{datetime.now():%Y%m%d-%H%M}"
    if invalid:
        name = "INVALID-" + name
    out = ROOT / "evals" / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "config": {**vars(args), "settings": cfg,
                                                              "date": datetime.now().isoformat(timespec="seconds")},
                               "rows": rows}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print("saved", out)
    if invalid:
        sys.exit(f"Run is INVALID and must not be reported: {invalid}")
    correct = sum(r["correct"] for r in rows) / max(1, len(rows))
    if correct < args.min_accuracy:
        sys.exit(f"Accuracy {correct:.0%} is below the required {args.min_accuracy:.0%}")


if __name__ == "__main__":
    main()
