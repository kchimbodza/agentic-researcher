"""Local web app: serves the UI, streams agent events as NDJSON, keeps an audit trail."""

import hmac
import html
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, claimcheck, docs, settings, store

APP_VERSION = "0.4.1"

RESULTS = docs.ROOT / "evals" / "results"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Agentic Researcher", version=APP_VERSION)
ACCESS_TOKEN = os.environ.get("AR_ACCESS_TOKEN", "")  # set when the app is reachable from a network


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Optional shared-token gate for cloud deployments: open /?token=... once, a cookie remembers it."""
    if not ACCESS_TOKEN or request.url.path == "/healthz":
        return await call_next(request)
    given = request.query_params.get("token") or request.cookies.get("ar_token") or \
        request.headers.get("authorization", "").removeprefix("Bearer ")
    if not hmac.compare_digest(given, ACCESS_TOKEN):
        return PlainTextResponse("Access token required: open the app as /?token=<your token>", status_code=401)
    response = await call_next(request)
    if request.query_params.get("token"):
        response.set_cookie("ar_token", ACCESS_TOKEN, httponly=True, samesite="strict", secure=request.url.scheme == "https")
    return response
paused: dict[str, dict] = {}  # run_id -> {"gen", "events", "t"} while waiting for approval


user_name = settings.user_name  # who signs reviews and appears in the audit log


class Ask(BaseModel):
    question: str
    thread_id: str | None = None


class Decision(BaseModel):
    approve: bool


def _stream(rid: str, gen, events: list[dict], send=None, intro: list[dict] = ()):
    def lines():
        for e in intro:
            yield json.dumps(e) + "\n"
        try:
            event = gen.send(send) if send is not None else next(gen)
            while True:
                events.append(event)
                yield json.dumps(event) + "\n"
                if event["type"] == "approval":
                    paused[rid] = {"gen": gen, "events": events, "t": time.time()}
                    store.save_run(rid, "waiting", events)
                    store.audit(rid, "agent", "paused", event["reason"])
                    yield json.dumps({"type": "paused", "run_id": rid}) + "\n"
                    return
                event = next(gen)
        except StopIteration:
            stopped = any(e["type"] == "error" for e in events)
            store.save_run(rid, "stopped" if stopped else "done", events)
            m = next((e for e in events if e["type"] == "metrics"), None)
            store.audit(rid, "agent", "stopped" if stopped else "completed",
                        f"{m['verified']}/{m['claims']} quotes verified · {m['tool_calls']} tool calls · "
                        f"{m['latency_ms']} ms · model {m['model']}" if m else "")
        except Exception as e:  # show the error in the UI instead of a broken stream
            err = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            events.append(err)
            store.save_run(rid, "error", events)
            store.audit(rid, "agent", "error", err["message"])
            yield json.dumps(err) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/healthz")
def healthz():
    """Liveness check for Docker/Azure (no token needed, reveals nothing)."""
    return {"ok": True}


# ---------------------------------------------------------------- asking
@app.post("/api/ask")
def ask(body: Ask):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Type a question first.")
    tid = body.thread_id if body.thread_id and store.thread_exists(body.thread_id) else store.create_thread(q)
    memory = store.history(tid)
    rid = store.new_id()
    scope = [d["name"] for d in docs.list_documents()]
    created = store.create_run(rid, tid, q, agent.current_mode(), agent.AGENT_VERSION, scope)
    who = user_name()
    store.audit(rid, who, "asked", q)
    run_event = {"type": "run", "run_id": rid, "thread_id": tid, "created": created, "question": q,
                 "agent_version": agent.AGENT_VERSION, "scope": scope, "followup": bool(memory), "user": who}
    return _stream(rid, agent.run(q, memory), [run_event], intro=[run_event])


@app.post("/api/runs/{rid}/approve")
def approve(rid: str, body: Decision):
    p = paused.pop(rid, None)
    if not p:
        raise HTTPException(404, "This run is no longer waiting (the app may have restarted). Ask again.")
    store.audit(rid, user_name(), "approved" if body.approve else "declined",
                "Showed the low-confidence brief" if body.approve else "Stopped at the approval step")
    return _stream(rid, p["gen"], p["events"], send=body.approve)


# ---------------------------------------------------------------- threads, review, audit
@app.get("/api/threads")
def threads(q: str = ""):
    return store.list_threads(q=q.strip())


@app.get("/api/threads/{tid}")
def thread(tid: str):
    t = store.get_thread(tid)
    if not t:
        raise HTTPException(404, "Conversation not found")
    for r in t["runs"]:
        r["waiting"] = r["id"] in paused
    return t


@app.delete("/api/threads/{tid}")
def delete_thread(tid: str):
    store.delete_thread(tid)
    return {"ok": True}


@app.post("/api/runs/{rid}/review")
def review(rid: str):
    if not store.get_run(rid):
        raise HTTPException(404, "Run not found")
    return store.review(rid, user_name())


@app.get("/api/runs/{rid}/audit")
def audit(rid: str):
    return store.audit_log(rid)


@app.get("/runs/{rid}/report", response_class=HTMLResponse)
def report(rid: str):
    """Print-ready record of one answer (the browser's print dialog saves it as PDF)."""
    r = store.get_run(rid)
    if not r:
        raise HTTPException(404, "Run not found")
    store.audit(rid, user_name(), "exported", "Opened the printable report")
    e = lambda s: html.escape(str(s))
    brief = next((x for x in r["events"] if x["type"] == "brief"), None)
    claims = [x for x in r["events"] if x["type"] == "claim"]
    m = r["metrics"] or {}
    body = e(brief["markdown"]) if brief else "<em>No brief was written.</em>"
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body).replace("\n", "<br>")
    rows = "".join(f"<tr><td>{'✓' if c['verified'] else '✗'}</td><td>{e(c['quote'])}</td>"
                   f"<td>{e(c['doc_name'])}, p.{c['page']}</td></tr>" for c in claims)
    log = "".join(f"<tr><td>{e(a['time'])}</td><td>{e(a['actor'])}</td><td>{e(a['action'])}</td><td>{e(a['detail'])}</td></tr>"
                  for a in store.audit_log(rid))
    status = f"Reviewed by {e(r['reviewed_by'])} at {e(r['reviewed_at'])}" if r["review"] == "reviewed" else "AI draft, not reviewed"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Agentic Researcher report {e(rid)}</title>
<style>body{{font:14px/1.6 system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 24px;color:#111}}
h1{{font-size:20px}} table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0 20px}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}} .meta{{color:#555;font-size:12.5px}}
.s{{display:inline-block;padding:2px 8px;border-radius:99px;background:{'#e6f6ee' if r['review']=='reviewed' else '#fff4e5'}}}
@media print{{.noprint{{display:none}}}}</style></head><body>
<p class="noprint"><button onclick="print()">Save as PDF / Print</button></p>
<p class="meta"><b>Agentic Researcher</b> · answers you can verify</p><h1>{e(r["question"])}</h1>
<p class="meta">Run {e(rid)} · {e(r['created'])} · {e(r['mode'])} mode · agent {e(r['agent_version'])} · model {e(m.get('model','–'))}<br>
<span class="s">{status}</span></p>
<p class="meta">Scope: {e(', '.join(r['scope']))}</p>
<h2>Answer</h2><p>{body}</p>
<h2>Evidence</h2><table><tr><th></th><th>Quote</th><th>Source</th></tr>{rows}</table>
<p class="meta">{m.get('verified','–')}/{m.get('claims','–')} quotes verified · {m.get('tool_calls','–')} tool calls ·
{m.get('latency_ms','–')} ms · {m.get('tokens',0)} tokens · ${m.get('cost_usd',0):.4f}</p>
<h2>Audit log</h2><table><tr><th>Time (UTC)</th><th>Who</th><th>Action</th><th>Detail</th></tr>{log}</table>
<p class="meta">AI-generated from public documents. Every quote was checked against its page; verify before relying on it.</p>
<script>setTimeout(()=>print(),400)</script></body></html>"""


# ---------------------------------------------------------------- documents, evals, status
@app.get("/api/status")
def status():
    return {"app_version": APP_VERSION, "mode": agent.current_mode(), "agent_version": agent.AGENT_VERSION,
            "documents": len(docs.list_documents()), "user": user_name(),
            "anthropic_key": settings.key_status()["set"], "ollama": shutil.which("ollama") is not None}


# ---------------------------------------------------------------- settings
class Key(BaseModel):
    key: str


@app.get("/api/settings")
def get_settings():
    return {"settings": settings.load(), "user": user_name(), "key": settings.key_status(),
            "claude_models": settings.CLAUDE_MODELS, "ollama_models": settings.ollama_models(),
            "paths": {"documents": str(docs.DATA), "database": str(store.DB), "config": str(settings.CONF)},
            "mode": agent.current_mode(), "agent_version": agent.AGENT_VERSION, "app_version": APP_VERSION,
            "claimcheck": claimcheck.info()}


@app.put("/api/settings")
def put_settings(patch: dict):
    try:
        return settings.save(patch)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/settings/key")
def put_key(body: Key):
    try:
        return settings.set_key(body.key)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/settings/key")
def delete_key():
    return settings.clear_key()


@app.get("/api/documents")
def documents():
    return docs.list_documents()


@app.post("/api/documents")
async def upload(file: UploadFile, kind: str = "papers"):
    if kind not in docs.KINDS:
        raise HTTPException(400, "kind must be papers or datasheets")
    name = Path(file.filename or "").name
    data = await file.read()
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        raise HTTPException(400, "Only PDF files can be added.")
    dest = docs.DATA / kind / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"id": f"{kind}/{name}"}


@app.get("/files/{folder}/{name}")
def pdf(folder: str, name: str):
    try:
        return FileResponse(docs.path_of(f"{folder}/{name}"), media_type="application/pdf")
    except FileNotFoundError:
        raise HTTPException(404, "Document not found")


@app.get("/api/evals")
def evals():
    out = []
    for f in sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
        except ValueError:
            continue
        out.append({"file": f.name, "summary": data.get("summary") if isinstance(data, dict) else None,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")})
    return out


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
