# Claim checker (Phase 6: LoRA fine-tune)

A small model that reads a **claim** and the **quote** it was written from, and answers
`SUPPORTED` or `NOT_SUPPORTED`.

## Why

The critic step in the app already checks that a quote appears word for word on the cited
PDF page. That catches made-up quotes, but not this failure, which we call **verified but wrong**:
the quote is real, but the claim written from it is not. Examples:

| Quote (real, on the page) | Claim the agent wrote | Problem |
|---|---|---|
| "The S8-PoE offers four PoE+ ports with a PoE budget of 60 Watts." | "The S16-PoE has a 60 W budget." | wrong product |
| "... designed with the TSMC low-power 40 nm technology." | "... 28 nm technology." | wrong number |
| "... consuming less than 1 mA" | "... consuming less than 1 µA" | wrong unit |

The model is meant to be a second check that runs after the verbatim-quote check.

## What is in this folder

| File | What it does |
|---|---|
| `pyproject.toml` | Its own uv project (Python 3.12, CUDA torch, transformers, peft, datasets, onnx, onnxruntime). Separate from the app's environment. |
| `make_data.py` | Builds the training data (MNLI + synthetic pairs from the project PDFs). |
| `train.py` | LoRA fine-tune of `distilbert-base-uncased` for 2-class classification. |
| `evaluate.py` | Accuracy / precision / recall / F1 for the model, plus a "numbers-match" baseline. |
| `export_onnx.py` | Merges the LoRA weights into the base model, exports to ONNX, checks ONNX against PyTorch. |
| `onnx_predict.py` | 28-line reference loader that needs only `onnxruntime` + `tokenizers` (no torch). |
| `results/` | Raw outputs of every run (see "Results files" below). |

## Data

**1. MultiNLI** ([`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli), loaded as parquet from the Hugging Face Hub).
Licence: most of the corpus is under the OANC licence (free to use, change and share); the fiction
part uses several permissive licences (e.g. CC BY-SA 3.0 for *Seven Swords*), as stated on the dataset card.
Turned into a binary task: `entailment` → SUPPORTED, `neutral` / `contradiction` → NOT_SUPPORTED.
The hypothesis is used as the claim and the premise as the quote.
A balanced 20,000 training pairs, and a balanced 3,000 pairs from `validation_matched` for evaluation.

**2. Synthetic domain pairs** from the project's own public PDFs (`../data/datasheets/*.pdf`,
`../data/papers/*.pdf`, read with pypdf, read-only). Fixed seed (13).

- *Near-copy pairs (run 1 and run 2).* A sentence with a number is the quote (sometimes with the
  sentence before/after). Positive: the sentence itself or with a lead-in ("According to the document, ...").
  Negatives: one number changed, a unit swapped (mA↔µA, W↔kW, ms↔s, ...), or a name swapped
  (S8-PoE/S16-PoE/S24-PoE, Route10, ESP32 variants, precision/recall/accuracy, min/max, ...).
- *Reworded pairs (added in run 2).* Facts are taken from spec-table rows ("POE Budget 40W",
  "Idle Power Consumption 6.5 Watts") and from two safe sentence shapes ("a PoE budget of 60 Watts",
  "94.8% precision"). Each fact is restated as a plain sentence with the product or paper name, the
  metric, the value and the unit, using 8 templates and several unit spellings (W/Watts, µA/uA/microamps,
  ms/milliseconds, mJ/millijoules, ...), e.g. "The S8-PoE has a 60 W PoE budget.",
  "The idle power consumption of the Alta Labs switch is 6.5 Watts."
  The negatives are made **from the reworded claim** (number, unit, product name, or metric swapped),
  with one reworded positive per negative, so "reworded" does not mean "false".
- Split by **document page**, per document: 20% of pages → test, 10% → validation, the rest → train.
  Run 2 uses the same 17 test pages as run 1 (checked), and no test page is in training (checked with an assert).

| Split | Run 1 | Run 2 |
|---|---|---|
| train file | 22,001 (20,000 MNLI + 2,001 domain) | 22,769 (20,000 MNLI + 2,769 domain, of which 615 reworded) |
| domain validation | 202 | 392 |
| MNLI validation | 3,000 | same 3,000 pairs as run 1 |
| domain test | 298 (`domain_test_run1.jsonl`) | 381 (`domain_test.jsonl`), same 17 pages, incl. 64 reworded pairs |

## How to run

```bash
cd finetune
uv sync                                  # creates finetune/.venv (Python 3.12, CUDA torch)
uv run python make_data.py               # -> data/*.jsonl, results/data_stats.json
uv run python train.py --require-gpu     # -> models/lora/, results/train_log.json
uv run python evaluate.py --tag run2_gpu # -> results/eval_<time>_run2_gpu.json + table
uv run python export_onnx.py             # -> ../models/claimcheck/, results/onnx_check.json
uv run python onnx_predict.py "The S16-PoE has a 60 W PoE budget." "The S8-PoE offers four PoE+ ports with a PoE budget of 60 Watts."
```

`train.py` uses the GPU when at least 1.5 GB is free. Without `--require-gpu` it falls back to the
CPU with a smaller sample (4,000 MNLI pairs + domain pairs ×2, 1 epoch) when the GPU is busy
(for example when Ollama holds most of the 6 GB card). With `--require-gpu` it stops instead.

## Training settings

Same model and LoRA settings in both runs: `distilbert-base-uncased` (66M parameters), LoRA r=16,
alpha=32, dropout 0.1 on q/k/v/out and both feed-forward layers, classifier head fully trained;
1,919,234 of 68,874,244 parameters trainable (2.8%). Max length 192 tokens (claim + quote,
longest-first truncation). AdamW, lr 5e-4, weight decay 0.01, 6% warm-up then linear decay,
batch 16, seed 13. The best epoch is picked by domain-validation accuracy.

| | Run 1 | Run 2 (current model) |
|---|---|---|
| Device | CPU (GPU had 0.83 GB free, Ollama loaded) | RTX 4050 Laptop GPU (5.95 GB free), bf16 autocast |
| Training pairs | 8,002 (4,000 MNLI + domain ×2) | 28,307 (20,000 MNLI + domain ×3) |
| Reworded domain pairs | no | yes |
| Epochs | 1 | 3 (best: 3) |
| Time | 1,617 s (~27 min) | 307 s (~5 min) |

## Results (measured)

"S" = SUPPORTED, "NOT_S" = NOT_SUPPORTED. Recall (NOT_S) is the share of wrong claims that get caught.
Both models are scored on the same files. Raw numbers: `results/eval_20261108_185104_run1_cpu_rescored.json`
(run 1) and `results/eval_20261108_185419_run2_gpu.json` (run 2).

| Set | Method | n | Accuracy | Precision (S) | Recall (S) | F1 (S) | Recall (NOT_S) | Macro F1 |
|---|---|---|---|---|---|---|---|---|
| MNLI validation_matched (binary) | run 1 | 3000 | 0.728 | 0.767 | 0.653 | 0.706 | 0.802 | 0.726 |
| MNLI validation_matched (binary) | run 2 | 3000 | 0.814 | 0.824 | 0.798 | 0.811 | 0.830 | 0.814 |
| Domain test, run-1 pairs | run 1 | 298 | 0.866 | 0.867 | 0.888 | 0.877 | 0.839 | 0.865 |
| Domain test, run-1 pairs | run 2 | 298 | 0.943 | 0.919 | 0.981 | 0.949 | 0.898 | 0.942 |
| Domain test, run-1 pairs | numbers-match baseline | 298 | 0.789 | 0.719 | 1.000 | 0.836 | 0.540 | 0.769 |
| Domain test, run-2 pairs | run 1 | 381 | 0.800 | 0.826 | 0.775 | 0.800 | 0.827 | 0.800 |
| Domain test, run-2 pairs | run 2 | 381 | 0.913 | 0.914 | 0.918 | 0.916 | 0.908 | 0.913 |
| Domain test, run-2 pairs | numbers-match baseline | 381 | 0.740 | 0.692 | 0.893 | 0.779 | 0.578 | 0.732 |
| Reworded pairs only (subset of run-2 test) | run 1 | 64 | 0.500 | 0.500 | 0.125 | 0.200 | 0.875 | 0.418 |
| Reworded pairs only (subset of run-2 test) | run 2 | 64 | 0.750 | 0.944 | 0.531 | 0.680 | 0.969 | 0.737 |
| Reworded pairs only (subset of run-2 test) | numbers-match baseline | 64 | 0.484 | 0.478 | 0.344 | 0.400 | 0.625 | 0.474 |

Run-2 domain test (381 pairs) accuracy by kind of pair:

| Kind | n | Run 1 | Run 2 | Numbers-match |
|---|---|---|---|---|
| number changed | 73 | 0.904 | 0.877 | 0.945 |
| unit swapped | 33 | 0.909 | 0.909 | 0.000 |
| name swapped | 47 | 0.617 | 0.915 | 0.383 |
| positive, verbatim | 49 | 0.816 | 1.000 | 1.000 |
| positive, with lead-in | 113 | 0.956 | 0.991 | 1.000 |
| positive, abbreviated | 2 | 0.000 | 1.000 | 1.000 |
| **reworded positive** | 32 | 0.125 | 0.531 | 0.344 |
| reworded, number changed | 10 | 0.800 | 1.000 | 1.000 |
| reworded, unit swapped | 16 | 0.938 | 0.938 | 0.438 |
| reworded, metric swapped | 4 | 0.750 | 1.000 | 0.500 |
| reworded, name swapped | 2 | 1.000 | 1.000 | 0.500 |

The baseline ("SUPPORTED if every number in the claim is in the quote") is good at changed numbers
but cannot see unit, name or metric swaps. It also counts digits inside names ("S8", "ESP32"), so it
wrongly rejects some reworded claims.

**ONNX check** (`results/onnx_check_run2_gpu.json`): on 20 domain-test pairs the ONNX model and the
merged PyTorch model gave the same prediction 20/20, max logit difference 6.0e-6. Running
`onnx_predict.py` (onnxruntime + tokenizers only) gave the same accuracy as PyTorch on both test sets
(0.943 on 298, 0.913 on 381), at about 13–15 ms per pair on the CPU.

## Spot checks (run 2, exported ONNX model, `results/spot_checks_run2_gpu.txt`)

| # | Claim | Quote | P(SUPPORTED) |
|---|---|---|---|
| a | The S8-PoE has a 60 W PoE budget. | The S8-PoE offers four PoE+ ports with a PoE budget of 60 Watts. | 0.988 |
| b | The S16-PoE has a 60 W PoE budget. | (same) | 0.001 |
| c | The ESP32 draws 10 µA in Deep-sleep with the RTC timer and memory on. | Deep-sleep The ULP coprocessor is powered up. 150 µA ULP sensor-monitored pattern 100 µA @1% duty RTC timer + RTC memory 10 µA | 0.990 |
| d | The Route10 supports a PoE budget of 40W per port. | POE Budget 40W Per-Port PoE (2) 802.3at POE+ | 0.998 |
| e | The radar-only model has 35.9 ms latency. | reduces latency from 35.9 ms to 15.8 ms | 0.022 |

Run 1 gave 0.079 for (a), so the reworded training data fixed that case. (d) is probably a
**false SUPPORTED**: in the datasheet, 40 W is the router's total PoE budget, not a per-port value,
and the model does not catch the "per port" change.

## Limitations

- **Reworded true claims are still the weak spot.** Run 2 accepts 53% of them (17 of 32), up from 12.5%.
  When it is unsure it leans towards NOT_SUPPORTED (precision for SUPPORTED on reworded pairs is
  0.944), so in the app it works best as a warning flag, not as a hard reject.
- **Scope words are missed**: "per port" vs total (spot check d), and probably "typical" vs
  "maximum", unless the change is a name or metric the model saw in training.
- **Small, synthetic domain test set** (381 pairs from 17 pages; only 64 reworded pairs). The
  negatives are made by rules, so they are easier and more regular than real agent mistakes.
- The reworded claims come from templates and a simple fact extractor; some training sentences read
  awkwardly. Most domain data comes from the ESP32 datasheet; pypdf table text is messy.
- MNLI accuracy (0.814) comes from 20,000 pairs and LoRA only; full fine-tuning on all of MNLI would score higher.
- The ONNX file is 268 MB (fp32, not quantised).

## Results files

| File | What |
|---|---|
| `train_log_run1_cpu.json`, `train_stdout.txt` | run 1 training log |
| `train_log_run2_gpu.json` (= `train_log.json`), `train_stdout_run2_gpu.txt` | run 2 training log |
| `eval_20261108_184209.json` | run 1, first evaluation (run-1 test set only) |
| `eval_20261108_185104_run1_cpu_rescored.json` | run 1 model scored on all current test files |
| `eval_20261108_185419_run2_gpu.json` | run 2 model |
| `data_stats_run1.json`, `data_stats.json` | data sizes per run |
| `onnx_check_run1.json`, `onnx_check_run2_gpu.json` | ONNX vs PyTorch checks |
| `spot_checks_run1.txt`, `spot_checks_run2_gpu.txt` | hand-written spot checks |

The run-1 LoRA adapter is kept in `models/lora_run1_cpu/`. `models/lora/` and `../models/claimcheck/` hold run 2.

## Credits

Built with help from an AI coding assistant (Claude Code). Base model: `distilbert-base-uncased`
(Apache 2.0). Training data: MultiNLI (see licence above) and the project's own public PDFs.
