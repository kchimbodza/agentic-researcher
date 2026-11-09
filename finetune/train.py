"""LoRA fine-tune of distilbert-base-uncased as a (claim, quote) -> SUPPORTED / NOT_SUPPORTED classifier.

Run:  uv run python train.py [--cpu] [--epochs 3]
Writes the LoRA adapter to finetune/models/lora/ and a log to finetune/results/train_log.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 192
SEED = 13
LABELS = ["NOT_SUPPORTED", "SUPPORTED"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def batches(rows, tok, batch_size, shuffle, rng=None):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [rows[j] for j in idx[i : i + batch_size]]
        enc = tok(
            [r["claim"] for r in chunk],
            [r["quote"] for r in chunk],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc.pop("token_type_ids", None)
        yield enc, torch.tensor([r["label"] for r in chunk])


@torch.no_grad()
def accuracy(model, rows, tok, device, batch_size=64) -> float:
    model.eval()
    correct = 0
    for enc, y in batches(rows, tok, batch_size, shuffle=False):
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        correct += (logits.argmax(-1).cpu() == y).sum().item()
    model.train()
    return correct / max(1, len(rows))


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
        modules_to_save=["pre_classifier", "classifier"],
    )
    return get_peft_model(model, cfg)


def train(device: str, rows: list[dict], val_rows: list[dict], args) -> dict:
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = build_model().to(device)
    trainable, total = model.get_nb_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(rows) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = int(0.06 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )
    use_amp = device == "cuda"
    rng = random.Random(SEED)
    history = []
    best = (-1.0, None)
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for enc, y in batches(rows, tok, args.batch_size, shuffle=True, rng=rng):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(**enc, labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += out.loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / 100:.4f} ({time.time() - t0:.0f}s)", flush=True)
                running = 0.0
        val_acc = accuracy(model, val_rows, tok, device)
        history.append({"epoch": epoch, "domain_val_acc": val_acc, "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch} done: domain validation accuracy {val_acc:.4f}", flush=True)
        if val_acc > best[0]:
            best = (val_acc, epoch)
            model.save_pretrained(HERE / "models" / "lora")
            tok.save_pretrained(HERE / "models" / "lora")
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train_seconds": round(time.time() - t0, 1),
        "trainable_params": trainable,
        "total_params": total,
        "best_epoch": best[1],
        "best_domain_val_acc": best[0],
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--domain-upsample", type=int, default=3, help="repeat domain training pairs N times")
    ap.add_argument("--cpu-mnli", type=int, default=4000, help="MNLI pairs used when training on CPU")
    ap.add_argument("--cpu-epochs", type=int, default=1)
    ap.add_argument("--require-gpu", action="store_true", help="exit instead of falling back to CPU")
    args = ap.parse_args()

    set_seed(SEED)
    rows = load_jsonl(HERE / "data" / "train.jsonl")
    rows = rows + [r for r in rows if r["source"] == "domain"] * (args.domain_upsample - 1)
    val_rows = load_jsonl(HERE / "data" / "validation.jsonl")

    def cpu_subset(rows):
        # CPU is far slower here, so train on a smaller MNLI sample, domain pairs upsampled 2x, fewer epochs.
        rng = random.Random(SEED)
        mnli = [r for r in rows if r["source"] == "mnli"]
        rng.shuffle(mnli)
        domain = list({id(r): r for r in rows if r["source"] == "domain"}.values())
        args.epochs = min(args.epochs, args.cpu_epochs)
        return mnli[: args.cpu_mnli] + domain * 2

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    note = "CPU forced with --cpu" if args.cpu else None
    if device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"free GPU memory: {free_gb:.2f} GB", flush=True)
        if free_gb < 1.5 and args.require_gpu:
            raise SystemExit(f"only {free_gb:.2f} GB GPU memory free; --require-gpu set, not training")
        if free_gb < 1.5:  # Ollama often holds ~5 GB of the 6 GB card
            device, note = "cpu", f"GPU had only {free_gb:.2f} GB free (Ollama loaded) -> CPU"
    if device == "cpu" and args.require_gpu:
        raise SystemExit("CUDA not available; --require-gpu set, not training")
    if device == "cpu":
        rows = cpu_subset(rows)
    try:
        log = train(device, rows, val_rows, args)
    except torch.OutOfMemoryError:
        if args.require_gpu:
            raise
        print("CUDA out of memory -> falling back to CPU with a smaller MNLI sample", flush=True)
        torch.cuda.empty_cache()
        rows = cpu_subset(rows)
        device, note = "cpu", "fell back to CPU after CUDA OOM"
        log = train(device, rows, val_rows, args)
    log["note"] = note

    log.update(
        {
            "base_model": BASE_MODEL,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "targets": "q,k,v,out,lin1,lin2", "modules_to_save": "pre_classifier, classifier"},
            "n_train_rows": len(rows),
            "n_train_mnli": sum(r["source"] == "mnli" for r in rows),
            "n_train_domain_after_upsample": sum(r["source"] == "domain" for r in rows),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
