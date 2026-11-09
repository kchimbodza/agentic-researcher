"""Evaluate the LoRA claim checker and a numbers-match baseline.

Run:  uv run python evaluate.py
Reads finetune/models/lora, writes finetune/results/eval_<date>.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from train import HERE, MAX_LENGTH, batches, load_jsonl

NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_match(claim: str, quote: str) -> int:
    """Baseline: SUPPORTED if every number in the claim also appears in the quote."""
    q = set(NUM.findall(quote))
    return int(all(n in q for n in NUM.findall(claim)))


def metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    # also report NOT_SUPPORTED as the positive class: that is the class the critic cares about catching
    pn, rn, fn, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=0, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_supported": round(p, 4),
        "recall_supported": round(r, 4),
        "f1_supported": round(f, 4),
        "precision_not_supported": round(pn, 4),
        "recall_not_supported": round(rn, 4),
        "f1_not_supported": round(fn, 4),
        "macro_f1": round((f + fn) / 2, 4),
    }


@torch.no_grad()
def predict(model, tok, rows, device) -> list[int]:
    preds = []
    for enc, _ in batches(rows, tok, 64, shuffle=False):
        preds += model(**{k: v.to(device) for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
    return preds


def by_kind(rows, preds) -> dict:
    out = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        out[r["kind"]][0] += int(p == r["label"])
        out[r["kind"]][1] += 1
    return {k: {"n": n, "accuracy": round(c / n, 4)} for k, (c, n) in sorted(out.items())}


def score_set(model, tok, rows, device) -> dict:
    pred = predict(model, tok, rows, device)
    base = [numbers_match(r["claim"], r["quote"]) for r in rows]
    y = [r["label"] for r in rows]
    return {"model": metrics(y, pred), "numbers_match_baseline": metrics(y, base),
            "model_by_kind": by_kind(rows, pred), "baseline_by_kind": by_kind(rows, base)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models" / "lora"))
    ap.add_argument("--tag", default="", help="label stored in the results file, e.g. run2_gpu")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoPeftModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    data = HERE / "data"
    # run-1 files are kept so every run is scored on exactly the same pairs
    nli_file = data / "nli_validation_run1.jsonl" if (data / "nli_validation_run1.jsonl").exists() else data / "nli_validation.jsonl"
    sets = {"domain_test_run1": data / "domain_test_run1.jsonl", "domain_test_run2": data / "domain_test.jsonl"}
    nli = load_jsonl(nli_file)
    y_nli = [r["label"] for r in nli]
    results = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": args.tag, "model_dir": args.model, "device": device,
        "max_length": MAX_LENGTH, "nli_file": nli_file.name,
        "nli_validation": {"model": metrics(y_nli, predict(model, tok, nli, device))},
    }
    for name, path in sets.items():
        if not path.exists():
            continue
        rows = load_jsonl(path)
        results[name] = score_set(model, tok, rows, device)
        rw = [r for r in rows if r["kind"] == "pos_reworded" or r["kind"].startswith("neg_rw")]
        if rw:
            results[name + "_reworded_only"] = score_set(model, tok, rw, device)

    out = HERE / "results" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    (HERE / "results" / "eval_latest.json").write_text(json.dumps(results, indent=2))

    print("| Set | Method | n | Accuracy | Precision (S) | Recall (S) | F1 (S) | Recall (NOT_S) | Macro F1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    table = [("MNLI validation_matched (binary)", "LoRA DistilBERT", results["nli_validation"]["model"])]
    for name in ["domain_test_run1", "domain_test_run2", "domain_test_run2_reworded_only"]:
        if name in results:
            table.append((name, "LoRA DistilBERT", results[name]["model"]))
            table.append((name, "numbers-match baseline", results[name]["numbers_match_baseline"]))
    for s_, m, r in table:
        print(f"| {s_} | {m} | {r['n']} | {r['accuracy']:.3f} | {r['precision_supported']:.3f} | {r['recall_supported']:.3f} "
              f"| {r['f1_supported']:.3f} | {r['recall_not_supported']:.3f} | {r['macro_f1']:.3f} |")
    for name in ["domain_test_run1", "domain_test_run2"]:
        if name in results:
            print(f"\n{name}: accuracy by pair kind (model / baseline)")
            for k, v in results[name]["model_by_kind"].items():
                print(f"  {k:14s} n={v['n']:4d}  {v['accuracy']:.3f} / {results[name]['baseline_by_kind'][k]['accuracy']:.3f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
