"""Optional fine-tuned claim checker (Phase 6): does this quote support this claim?

A small encoder fine-tuned with LoRA (see finetune/), exported to ONNX so the app only needs
onnxruntime + tokenizers. If models/claimcheck/ is missing, available() is False and the critic
uses only the rule checks in guardrails.py.
"""

import json
from functools import lru_cache

import numpy as np

from .docs import ROOT

MODEL_DIR = ROOT / "models" / "claimcheck"


def available() -> bool:
    return (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    cfg = json.loads((MODEL_DIR / "config.json").read_text()) if (MODEL_DIR / "config.json").exists() else {}
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    max_len = int(cfg.get("max_length", 256))
    tok.enable_truncation(max_length=max_len)
    sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    return tok, sess, cfg


def support_probability(claim: str, quote: str) -> float:
    """Probability (0-1) that the quote supports the claim."""
    tok, sess, _ = _load()
    enc = tok.encode(claim, quote)
    feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
             "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    logits = sess.run(None, feeds)[0][0]
    e = np.exp(logits - logits.max())
    return float(e[1] / e.sum())


def info() -> dict:
    if not available():
        return {"available": False}
    _, _, cfg = _load()
    return {"available": True, **{k: cfg.get(k) for k in ("base_model", "trained", "max_length")}}
