"""Model routing: one call site for every LLM request the agent makes.

complete_json(system, user, schema, budget, purpose) asks the main provider for JSON matching
`schema`; on failure (no key, network error, refusal, over budget) it falls back to the next
provider. Every call is logged to logs/llm_calls.jsonl with model, tokens, cost and latency.

Providers:
  anthropic  Claude via the official SDK (structured outputs; server-side refusal fallbacks on Opus/Sonnet)
  ollama     a local model (free, private, slower; JSON mode with a schema)
  azure      Azure OpenAI chat completions over REST (only if AZURE_OPENAI_* env vars are set)
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic
import httpx
import ollama

from . import docs, settings

LOG = docs.ROOT / "logs" / "llm_calls.jsonl"

# USD per million tokens (input, output). Claude prices from Anthropic's published rates.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK_OK = {"claude-opus-5-5", "claude-sonnet-5-5"}  # models that accept fallbacks="default"


class LLMError(RuntimeError):
    pass


class OverBudget(LLMError):
    pass


@dataclass
class Budget:
    """Per-run spend tracking and cap."""
    cap_usd: float
    cost_usd: float = 0.0
    tokens: int = 0
    calls: int = 0
    models: list = field(default_factory=list)

    def add(self, model: str, tokens_in: int, tokens_out: int, cost: float):
        self.cost_usd += cost
        self.tokens += tokens_in + tokens_out
        self.calls += 1
        if model not in self.models:
            self.models.append(model)


def _log(row: dict):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- providers
def _anthropic(model, system, user, schema, max_tokens):
    client = anthropic.Anthropic(max_retries=2, timeout=120)
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}],
                  output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    if model in FALLBACK_OK:
        # On a safety decline, Anthropic re-runs the request on its recommended fallback model.
        r = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        r = client.messages.create(**kwargs)
    if r.stop_reason == "refusal":
        raise LLMError(f"{model} declined the request")
    if r.stop_reason == "max_tokens":
        raise LLMError(f"{model} hit max_tokens")
    text = next((b.text for b in r.content if b.type == "text"), "")
    tin, tout = r.usage.input_tokens, r.usage.output_tokens
    pin, pout = PRICES.get(r.model, PRICES.get(model, (0, 0)))
    return json.loads(text), r.model, tin, tout, (tin * pin + tout * pout) / 1e6


def _ollama(model, system, user, schema, max_tokens):
    r = ollama.chat(model=model, format=schema, keep_alive="10m",
                    options={"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (json.loads(r["message"]["content"]), model,
            r.get("prompt_eval_count", 0) or 0, r.get("eval_count", 0) or 0, 0.0)


def _azure(model, system, user, schema, max_tokens):
    endpoint, key = os.environ.get("AZURE_OPENAI_ENDPOINT"), os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model)
    if not (endpoint and key and deployment):
        raise LLMError("Azure OpenAI is not configured")
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": False}}}
    r = httpx.post(url, params={"api-version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")},
                   headers={"api-key": key}, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    u = data.get("usage", {})
    rate_in = float(os.environ.get("AZURE_PRICE_IN", 0)); rate_out = float(os.environ.get("AZURE_PRICE_OUT", 0))
    tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    return (json.loads(data["choices"][0]["message"]["content"]), f"azure:{deployment}", tin, tout,
            (tin * rate_in + tout * rate_out) / 1e6)


PROVIDERS = {"anthropic": _anthropic, "ollama": _ollama, "azure": _azure}


def route() -> list[tuple[str, str]]:
    """Provider order for this request: the main provider from Settings, then the fallbacks."""
    s = settings.load()
    order = {"anthropic": [("anthropic", s["anthropic_model"]), ("ollama", s["ollama_model"])],
             "ollama": [("ollama", s["ollama_model"]), ("anthropic", s["anthropic_model"])]}[s["provider"]]
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        order.append(("azure", os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt")))
    if not settings.key_status()["set"]:
        order = [o for o in order if o[0] != "anthropic"]
    return order


def available() -> bool:
    """Is any model reachable? (Decides agent mode vs preview mode.)"""
    if settings.key_status()["set"] or os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return True
    try:
        names = {m["model"] for m in ollama.list()["models"]}
        return settings.load()["ollama_model"] in names
    except Exception:
        return False


def complete_json(system: str, user: str, schema: dict, budget: Budget, purpose: str,
                  max_tokens: int = 2000, order: list | None = None) -> dict:
    errors = []
    for provider, model in order or route():
        if provider != "ollama" and budget.cost_usd >= budget.cap_usd:
            errors.append(f"{provider}: over the ${budget.cap_usd:.2f} per-run cap")
            continue
        t0 = time.perf_counter()
        try:
            data, used, tin, tout, cost = PROVIDERS[provider](model, system, user, schema, max_tokens)
        except Exception as e:
            errors.append(f"{provider}/{model}: {type(e).__name__}: {str(e)[:200]}")
            _log({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "purpose": purpose,
                  "provider": provider, "model": model, "ok": False, "error": errors[-1]})
            continue
        ms = int((time.perf_counter() - t0) * 1000)
        budget.add(used, tin, tout, cost)
        _log({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "purpose": purpose,
              "provider": provider, "model": used, "ok": True, "tokens_in": tin, "tokens_out": tout,
              "cost_usd": round(cost, 6), "latency_ms": ms, "fallback": bool(errors)})
        return data
    raise LLMError("No model could answer: " + " | ".join(errors))
