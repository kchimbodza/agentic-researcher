"""User settings (~/.config/agentic-researcher/settings.json) and the API key (env file).

The key is kept in the private env file (chmod 600), never in the repo or settings.json,
and is never sent back to the browser: the UI only sees whether a key is set and its last 4 characters.
"""

import json
import os
import pwd
import re
import urllib.request
from pathlib import Path

CONF = Path.home() / ".config" / "agentic-researcher"
FILE = CONF / "settings.json"
ENV = CONF / "env"

CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
PROVIDERS = ["anthropic", "ollama"]
DEFAULTS = {
    "reviewer_name": "",
    "provider": "anthropic",           # main model; the other is the fallback (Phase 5)
    "anthropic_model": "claude-haiku-5-5",
    "ollama_model": "qwen2.5:7b",
    "cost_cap_usd": 0.05,              # per run
    "claim_checker": "rules",          # rules | auto (auto adds the fine-tuned model: fewer wrong citations, lower recall; see README)
    "tool_transport": "mcp",           # mcp | direct
}


def load() -> dict:
    try:
        data = json.loads(FILE.read_text())
    except (OSError, ValueError):
        data = {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save(patch: dict) -> dict:
    s = load()
    for k, v in patch.items():
        if k not in DEFAULTS:
            raise ValueError(f"Unknown setting: {k}")
        if k == "reviewer_name":
            v = str(v).strip()[:80]
        elif k == "provider" and v not in PROVIDERS:
            raise ValueError("provider must be anthropic or ollama")
        elif k == "anthropic_model" and v not in CLAUDE_MODELS:
            raise ValueError("Unknown Claude model")
        elif k == "ollama_model":
            v = str(v).strip()[:80]
        elif k == "claim_checker" and v not in ("auto", "rules"):
            raise ValueError("claim_checker must be auto or rules")
        elif k == "tool_transport" and v not in ("mcp", "direct"):
            raise ValueError("tool_transport must be mcp or direct")
        elif k == "cost_cap_usd":
            v = float(v)
            if not 0 < v <= 5:
                raise ValueError("Cost cap must be between $0.01 and $5 per run")
        s[k] = v
    CONF.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(s, indent=2))
    return s


def user_name() -> str:
    """Name used for review sign-off and the audit log."""
    name = load()["reviewer_name"] or os.environ.get("REVIEWER_NAME", "")
    if name:
        return name
    p = pwd.getpwuid(os.getuid())
    return (p.pw_gecos.split(",")[0] or p.pw_name).strip()


def key_status() -> dict:
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"set": bool(k), "hint": f"…{k[-4:]}" if len(k) > 8 else ""}


def _write_env(name: str, value: str | None) -> None:
    CONF.mkdir(parents=True, exist_ok=True)
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    lines = [l for l in lines if not l.startswith(f"{name}=")]
    if value is not None:
        lines.append(f"{name}={value}")
    ENV.write_text("\n".join(lines) + "\n")
    ENV.chmod(0o600)


def set_key(key: str) -> dict:
    key = key.strip()
    if not re.fullmatch(r"sk-ant-[A-Za-z0-9_\-]{20,}", key):
        raise ValueError("That doesn't look like an Anthropic API key (it starts with sk-ant-).")
    _write_env("ANTHROPIC_API_KEY", key)
    os.environ["ANTHROPIC_API_KEY"] = key
    return key_status()


def clear_key() -> dict:
    _write_env("ANTHROPIC_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    return key_status()


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as r:
            return sorted(m["name"] for m in json.load(r).get("models", []) if "embed" not in m["name"])
    except (OSError, ValueError):
        return []
