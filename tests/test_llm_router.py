import json

import pytest

from app import llm


def test_falls_back_to_next_provider_and_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "LOG", tmp_path / "calls.jsonl")
    calls = []

    def broken(model, *a):
        calls.append(model)
        raise RuntimeError("down")

    def works(model, *a):
        calls.append(model)
        return {"ok": True}, model, 100, 20, 0.0012

    monkeypatch.setitem(llm.PROVIDERS, "anthropic", broken)
    monkeypatch.setitem(llm.PROVIDERS, "ollama", works)
    b = llm.Budget(cap_usd=0.05)
    out = llm.complete_json("sys", "user", {}, b, "test", order=[("anthropic", "claude-opus-5-5"), ("ollama", "qwen")])
    assert out == {"ok": True} and calls == ["claude-opus-5-5", "qwen"]
    assert b.tokens == 120 and b.models == ["qwen"]
    rows = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in rows] == [False, True] and rows[1]["fallback"] is True


def test_cost_cap_skips_paid_providers(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "LOG", tmp_path / "calls.jsonl")
    monkeypatch.setitem(llm.PROVIDERS, "anthropic", lambda *a: pytest.fail("should be skipped"))
    monkeypatch.setitem(llm.PROVIDERS, "ollama", lambda m, *a: ({"x": 1}, m, 1, 1, 0.0))
    b = llm.Budget(cap_usd=0.01, cost_usd=0.02)
    assert llm.complete_json("s", "u", {}, b, "t", order=[("anthropic", "m"), ("ollama", "q")]) == {"x": 1}


def test_all_providers_failing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "LOG", tmp_path / "calls.jsonl")
    monkeypatch.setitem(llm.PROVIDERS, "ollama", lambda *a: (_ for _ in ()).throw(RuntimeError("no")))
    with pytest.raises(llm.LLMError):
        llm.complete_json("s", "u", {}, llm.Budget(1), "t", order=[("ollama", "q")])


def test_route_skips_claude_without_a_key():
    assert all(p != "anthropic" for p, _ in llm.route())
