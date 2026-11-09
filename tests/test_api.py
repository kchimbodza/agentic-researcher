from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_status_and_documents():
    s = client.get("/api/status").json()
    assert s["documents"] == 3 and "app_version" in s
    assert {d["id"] for d in client.get("/api/documents").json()} >= {"datasheets/Demo_Router.pdf"}


def test_settings_validation_and_key_never_returned():
    assert client.put("/api/settings", json={"cost_cap_usd": 99}).status_code == 400
    assert client.put("/api/settings", json={"provider": "ollama"}).json()["provider"] == "ollama"
    assert client.post("/api/settings/key", json={"key": "nope"}).status_code == 400
    r = client.post("/api/settings/key", json={"key": "sk-ant-" + "a" * 40}).json()
    assert r == {"set": True, "hint": "…aaaa"}
    assert "sk-ant-" not in client.get("/api/settings").text
    client.delete("/api/settings/key")


def test_files_refuse_traversal():
    assert client.get("/files/datasheets/..%2F..%2Fpyproject.toml").status_code == 404
    assert client.get("/files/datasheets/Demo_Router.pdf").headers["content-type"] == "application/pdf"


def test_access_token_gate(monkeypatch):
    import app.server as srv
    monkeypatch.setattr(srv, "ACCESS_TOKEN", "s3cret")
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"authorization": "Bearer s3cret"}).status_code == 200
    r = client.get("/api/status?token=s3cret")
    assert r.status_code == 200 and "ar_token" in r.headers.get("set-cookie", "")


def test_healthz_is_open_even_with_token(monkeypatch):
    import app.server as srv
    monkeypatch.setattr(srv, "ACCESS_TOKEN", "s3cret")
    assert client.get("/healthz").json() == {"ok": True}
