"""Merge the LoRA adapter into DistilBERT, export to ONNX, and check ONNX == PyTorch.

Run:  uv run python export_onnx.py
Writes ../models/claimcheck/{model.onnx, tokenizer.json, ..., config.json}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from peft import AutoPeftModelForSequenceClassification
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from train import BASE_MODEL, HERE, LABELS, MAX_LENGTH, load_jsonl

OUT = HERE.parent / "models" / "claimcheck"


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    peft_model = AutoPeftModelForSequenceClassification.from_pretrained(HERE / "models" / "lora", attn_implementation="eager")
    merged = peft_model.merge_and_unload().eval().to("cpu")
    tok = AutoTokenizer.from_pretrained(HERE / "models" / "lora")
    tok.save_pretrained(OUT)

    dummy = tok("a claim", "a quote", return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    torch.onnx.export(
        LogitsOnly(merged).eval(),
        (dummy["input_ids"], dummy["attention_mask"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    (OUT / "config.json").write_text(
        json.dumps(
            {"labels": LABELS, "max_length": MAX_LENGTH, "base_model": BASE_MODEL, "adapter": "LoRA r=16 (merged)",
             "inputs": ["input_ids", "attention_mask"], "trained": time.strftime("%Y-%m-%d")},
            indent=2,
        )
    )

    merged.eval()  # export restores the wrapper's train() flag, which would switch dropout back on
    # ---- verify: ONNX (tokenizers + onnxruntime only) vs PyTorch merged model on 20 domain test pairs
    rows = load_jsonl(HERE / "data" / "domain_test.jsonl")[:20]
    sess = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    print("onnx inputs:", [i.name for i in sess.get_inputs()])
    rt_tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    rt_tok.enable_truncation(MAX_LENGTH)
    agree, max_diff = 0, 0.0
    for r in rows:
        e = rt_tok.encode(r["claim"], r["quote"])
        ids = np.array([e.ids], dtype=np.int64)
        mask = np.array([e.attention_mask], dtype=np.int64)
        onnx_logits = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0][0]
        with torch.no_grad():
            pt_logits = merged(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)).logits[0].numpy()
        hf_ids = tok(r["claim"], r["quote"], truncation=True, max_length=MAX_LENGTH)["input_ids"]
        assert hf_ids == e.ids, "tokenizers and transformers tokenisation differ"
        agree += int(onnx_logits.argmax() == pt_logits.argmax())
        max_diff = max(max_diff, float(np.abs(onnx_logits - pt_logits).max()))
    result = {"pairs": len(rows), "same_prediction": agree, "max_abs_logit_diff": max_diff,
              "onnx_inputs": [i.name for i in sess.get_inputs()],
              "onnx_size_mb": round((OUT / "model.onnx").stat().st_size / 1e6, 1)}
    (HERE / "results" / "onnx_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
