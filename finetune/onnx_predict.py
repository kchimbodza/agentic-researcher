"""Reference loader for the claim checker: needs only onnxruntime + tokenizers (no torch)."""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "claimcheck"
CFG = json.loads((MODEL_DIR / "config.json").read_text())
TOK = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
TOK.enable_truncation(CFG["max_length"])
SESSION = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])


def supported_probability(claim: str, quote: str) -> float:
    enc = TOK.encode(claim, quote)
    feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
             "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    logits = SESSION.run(None, feeds)[0][0]
    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    return float(probs[1])  # index 1 = SUPPORTED


if __name__ == "__main__":
    print(round(supported_probability(sys.argv[1], sys.argv[2]), 4))
