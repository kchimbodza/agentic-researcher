"""The eval must not report a score that silently mixed models (router fallback, dead API)."""

from evals.evaluate import invalid_reason


def row(i, model="claude-haiku-5-5", error=""):
    return {"id": i, "model": model, "error": error}


def test_clean_run_is_valid():
    assert invalid_reason([row("s01"), row("s02")], "claude-haiku-5-5") == ""


def test_fallback_to_local_model_is_invalid():
    why = invalid_reason([row("s01"), row("s02", model="claude-haiku-5-5 + qwen2.5:7b")], "claude-haiku-5-5")
    assert "s02" in why and "other than claude-haiku-5-5" in why


def test_no_model_available_is_invalid():
    why = invalid_reason([row("s01", model=None, error="LLMError: No model could answer: credit balance too low")],
                         "claude-haiku-5-5")
    assert "no model available" in why
