"""SQLite storage: conversations (threads), runs, and an append-only audit log.

Every run keeps its full event list, so the UI can redraw any past answer exactly
as it happened, and the audit log records who did what and when.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from .docs import ROOT

DB = ROOT / "logs" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES threads(id), question TEXT NOT NULL,
  created TEXT NOT NULL, status TEXT NOT NULL, review TEXT NOT NULL DEFAULT 'draft',
  reviewed_by TEXT, reviewed_at TEXT, mode TEXT, agent_version TEXT, model TEXT,
  scope TEXT, events TEXT NOT NULL DEFAULT '[]', metrics TEXT);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, time TEXT NOT NULL,
  actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def db() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _run(row) -> dict:
    r = dict(row)
    r["events"] = json.loads(r["events"])
    r["scope"] = json.loads(r["scope"] or "[]")
    r["metrics"] = json.loads(r["metrics"] or "null")
    return r


# ---- threads
def create_thread(title: str) -> str:
    tid, t = new_id(), now()
    with db() as con:
        con.execute("INSERT INTO threads VALUES (?,?,?,?)", (tid, title[:120], t, t))
    return tid


def thread_exists(tid: str) -> bool:
    with db() as con:
        return con.execute("SELECT 1 FROM threads WHERE id=?", (tid,)).fetchone() is not None


def list_threads(limit: int = 200, q: str = "") -> list[dict]:
    """Conversations, newest first. q filters by any question asked in the conversation."""
    like = f"%{q}%"
    with db() as con:
        rows = con.execute("""SELECT t.*, COUNT(r.id) AS runs,
                 SUM(r.review='reviewed') AS reviewed FROM threads t LEFT JOIN runs r ON r.thread_id=t.id
                 WHERE ?='' OR t.title LIKE ? OR t.id IN (SELECT thread_id FROM runs WHERE question LIKE ?)
                 GROUP BY t.id ORDER BY t.updated DESC LIMIT ?""", (q, like, like, limit)).fetchall()
    return [dict(r) for r in rows]


def get_thread(tid: str) -> dict | None:
    with db() as con:
        t = con.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
        if not t:
            return None
        runs = con.execute("SELECT * FROM runs WHERE thread_id=? ORDER BY created", (tid,)).fetchall()
    return {**dict(t), "runs": [_run(r) for r in runs]}


def delete_thread(tid: str) -> None:
    with db() as con:
        con.execute("DELETE FROM runs WHERE thread_id=?", (tid,))
        con.execute("DELETE FROM threads WHERE id=?", (tid,))


def history(tid: str) -> list[dict]:
    """Earlier questions and answers in a thread: the agent's conversation memory."""
    out = []
    for r in (get_thread(tid) or {"runs": []})["runs"]:
        brief = next((e for e in r["events"] if e["type"] == "brief"), None)
        if brief:
            out.append({"question": r["question"], "answer": brief["markdown"]})
    return out


# ---- runs
def create_run(rid: str, tid: str, question: str, mode: str, version: str, scope: list[str]) -> str:
    t = now()
    with db() as con:
        con.execute("""INSERT INTO runs (id, thread_id, question, created, status, mode, agent_version, scope)
                       VALUES (?,?,?,?,?,?,?,?)""", (rid, tid, question, t, "running", mode, version, json.dumps(scope)))
        con.execute("UPDATE threads SET updated=? WHERE id=?", (t, tid))
    return t


def save_run(rid: str, status: str, events: list[dict]) -> None:
    m = next((e for e in events if e["type"] == "metrics"), None)
    with db() as con:
        con.execute("UPDATE runs SET status=?, events=?, metrics=?, model=COALESCE(?, model) WHERE id=?",
                    (status, json.dumps(events), json.dumps(m) if m else None, m and m.get("model"), rid))


def get_run(rid: str) -> dict | None:
    with db() as con:
        r = con.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    return _run(r) if r else None


def review(rid: str, reviewer: str) -> dict:
    t = now()
    with db() as con:
        con.execute("UPDATE runs SET review='reviewed', reviewed_by=?, reviewed_at=? WHERE id=?", (reviewer, t, rid))
    audit(rid, reviewer, "reviewed", "Marked the answer as reviewed")
    return {"review": "reviewed", "reviewed_by": reviewer, "reviewed_at": t}


# ---- audit
def audit(rid: str, actor: str, action: str, detail: str = "") -> None:
    with db() as con:
        con.execute("INSERT INTO audit (run_id, time, actor, action, detail) VALUES (?,?,?,?,?)",
                    (rid, now(), actor, action, detail))


def audit_log(rid: str) -> list[dict]:
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM audit WHERE run_id=? ORDER BY id", (rid,))]
