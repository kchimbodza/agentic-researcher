"""The whole LangGraph flow with a scripted fake model and in-process tools."""
import pytest

from app import agent, index, llm, settings


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(index, "vector_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no embeddings")))
    settings.save({"tool_transport": "direct", "claim_checker": "rules"})
    script = {}

    def complete_json(system, user, schema, budget, purpose, **kw):
        budget.add("fake-model", 10, 5, 0.0)
        return script[purpose](user) if callable(script[purpose]) else script[purpose]

    monkeypatch.setattr(llm, "complete_json", complete_json)
    return script


def extract(user):
    lookup = user.splitlines()[0]
    if "Router" in lookup:
        return {"claims": [{"claim": "The router PoE budget is 40 W.", "quote": "POE Budget 40W", "passage": "P1"}]}
    if "S8" in lookup:
        return {"claims": [
            {"claim": "The S8-PoE PoE budget is 60 W.", "quote": "S8-PoE offers four PoE+ ports with a PoE budget of 60 Watts", "passage": "P1"},
            {"claim": "The S8-PoE budget is 999 W.", "quote": "say the budget is 999 W", "passage": "P2"},
            {"claim": "The S8-PoE has 12 ports.", "quote": "this sentence is not in the PDF at all", "passage": "P1"}]}
    return {"claims": []}


def test_full_run_verifies_removes_and_writes(fake_model):
    fake_model.update({
        "plan": {"lookups": ["Demo Router PoE budget", "S8-PoE PoE budget"]},
        "extract": extract,
        "refine": {"query": "nothing"},
        "write": {"sentences": [
            {"text": "The combined budget is 40 W + 60 W = 100 W.", "cites": [1, 2]},
            {"text": "It also supports 500 W bursts.", "cites": [1]},   # number not in evidence: dropped
            {"text": "No citation here.", "cites": []}],                  # uncited: dropped
            "not_found": []},
    })
    events = list(agent.run("What is the combined PoE budget of the demo router and the S8-PoE?", [], auto_approve=True))
    types = [e["type"] for e in events]
    assert types[0] == "mode" and events[0]["mode"] == "agent" and types[-1] == "metrics"
    claims = [e for e in events if e["type"] == "claim"]
    ok = [c for c in claims if c["verified"]]
    assert len(ok) == 2
    reasons = " ".join(c["reason"] for c in claims if not c["verified"])
    assert "not found on the cited page" in reasons  # invented quote
    brief = next(e for e in events if e["type"] == "brief")
    assert "100 W" in brief["markdown"] and "500" not in brief["markdown"] and brief["dropped"] == 2
    assert {c["n"] for c in brief["citations"]} == {1, 2}


def test_injected_quote_is_removed(fake_model):
    fake_model.update({"plan": {"lookups": ["S8-PoE PoE budget"]}, "extract": extract, "refine": {"query": "x"},
                       "write": {"sentences": [{"text": "The S8-PoE budget is 60 W.", "cites": [1]}], "not_found": []}})
    events = list(agent.run("What is the S8-PoE PoE budget?", [], auto_approve=True))
    bad = [c for c in events if c["type"] == "claim" and "999" in c["text"]]
    assert bad and not bad[0]["verified"]


def test_partial_answer_asks_for_approval_and_can_stop(fake_model):
    fake_model.update({"plan": {"lookups": ["S8-PoE PoE budget", "Demo list price"]}, "extract": extract,
                       "refine": {"query": "retail list price"},
                       "write": {"sentences": [{"text": "The S8-PoE budget is 60 W.", "cites": [1]}], "not_found": ["price"]}})
    gen = agent.run("What is the S8-PoE budget and price?", [])
    events = []
    for e in gen:
        events.append(e)
        if e["type"] == "approval":
            break
    assert events[-1]["type"] == "approval" and "price" in events[-1]["reason"]
    rest = []
    e = gen.send(False)
    while True:
        rest.append(e)
        try:
            e = next(gen)
        except StopIteration:
            break
    assert rest[-1]["type"] == "error" and "Stopped" in rest[-1]["message"]


def test_nothing_found_says_not_found_without_asking(fake_model):
    fake_model.update({"plan": {"lookups": ["Route10 price in CAD"]}, "extract": {"claims": []}, "refine": {"query": "x"},
                       "write": {"sentences": [], "not_found": []}})
    events = list(agent.run("What does the Route10 cost?", []))
    assert not any(e["type"] == "approval" for e in events)
    assert "Not found" in next(e for e in events if e["type"] == "brief")["markdown"]
