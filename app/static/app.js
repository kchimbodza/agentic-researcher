/* Agentic Researcher UI: answers you can verify.
   One renderer (Run) draws a research run from its events, both live (streamed) and
   replayed from the database, so a saved answer looks exactly like it did when it ran. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pdfUrl = (doc, page) => `/files/${doc.split("/").map(encodeURIComponent).join("/")}#page=${page}&view=FitH`;
const when = (iso) => new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const icon = (id, cls = "i") => `<svg class="${cls}"><use href="#i-${id}"/></svg>`;
const json = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const send = (url, method, body) => json(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
function toast(msg) {
  const t = Object.assign(document.createElement("div"), { className: "toast", textContent: msg });
  document.body.append(t);
  setTimeout(() => t.remove(), 3500);
}

const STEPS = [["planner", "Plan"], ["researcher", "Search"], ["critic", "Check"], ["writer", "Write"]];
const EXAMPLES = [
  "What is the ESP32 deep-sleep current?",
  "What latency and energy does the bathroom radar fall detector report on a Raspberry Pi?",
  "How many PoE ports does the switch have, and what is its power budget?",
  "Which fall detection accuracy does mmWave radar reach?",
];

let status = {};
let current = null;   // open conversation id
let busy = false;

/* ================================================================ one research run */
class Run {
  constructor(row, isFollowup) {
    this.row = row;
    this.claims = [];
    this.cites = [];
    this.approvalPending = false;
    const el = (this.el = document.createElement("article"));
    el.className = "run";
    el.innerHTML = `
      <div class="record">
        <span class="chipst run" data-k="status">● Running</span>
        <span class="mono" data-k="id"></span><span class="mono" data-k="time"></span><span class="mono" data-k="ver"></span>
        <div class="acts">
          <button class="btn pri hidden" data-a="review">${icon("check")}Mark reviewed</button>
          <button class="btn hidden" data-a="export">${icon("dl")}Export PDF</button>
          <button class="btn" data-a="audit">${icon("list")}Audit log</button>
        </div>
      </div>
      ${isFollowup ? `<p class="kicker">Follow-up</p>` : ""}
      <h1 class="q" data-k="q"></h1>
      <div class="scope" data-k="scope"></div>
      <div class="stepper">${STEPS.map(([id, n], i) => `<div class="step" data-step="${id}"><div class="n">${i + 1}</div><b>${n}</b><span>waiting</span></div>`).join("")}</div>
      <div class="approve hidden" data-k="approve"></div>
      <section class="answer">
        <div class="ansbody">
          <h2><span class="badge wait" data-k="badge">${icon("check")}</span>Answer</h2>
          <div class="read wait" data-k="brief">The agents are working…</div>
        </div>
        <div class="metrics" data-k="metrics">
          ${[["clock", "Time"], ["ok", "Verified"], ["dollar", "Cost"], ["tool", "Tool calls"]].map(([ic, l]) =>
            `<div class="m">${icon(ic)}<div><b data-m="${l}">–</b><span>${l}</span></div></div>`).join("")}
          <div class="m">${icon("cpu")}<div><b class="sm" data-m="Model">–</b><span>Model · <span data-m="Tokens">0</span> tokens</span></div></div>
        </div>
      </section>
      <section class="card ev hidden" data-k="evCard">
        <h2><span class="l">${icon("layers")}Evidence</span><button class="linkbtn" data-a="steps">Show agent steps →</button></h2>
        <ol class="steps hidden" data-k="steps"></ol>
        <table class="tbl" data-k="ev"></table>
      </section>`;
    el.addEventListener("click", (e) => this.click(e));
    this.k = (k) => el.querySelector(`[data-k="${k}"]`);
    this.m = (k) => el.querySelector(`[data-m="${k}"]`);
  }

  click(e) {
    const a = e.target.closest("[data-a]")?.dataset.a;
    const cite = e.target.closest("[data-cite]");
    const claim = e.target.closest("[data-claim]");
    if (cite) return openSource(this, this.claimForCite(+cite.dataset.cite));
    if (claim) return openSource(this, +claim.dataset.claim);
    if (a === "review") return this.review();
    if (a === "export") return window.open(`/runs/${this.row.id}/report`, "_blank");
    if (a === "audit") return openAudit(this.row.id);
    if (a === "steps") {
      const s = this.k("steps"); s.classList.toggle("hidden");
      e.target.textContent = s.classList.contains("hidden") ? "Show agent steps →" : "Hide agent steps ↑";
    }
    if (a === "yes" || a === "no") return this.decide(a === "yes");
  }

  claimForCite(n) {
    const c = this.cites.find((x) => x.n === n);
    const i = c ? this.claims.findIndex((x) => x.verified && x.doc === c.doc && x.page === c.page) : 0;
    return Math.max(i, 0);
  }

  setStatus() {
    const s = this.k("status"), r = this.row;
    const show = (a, on) => this.el.querySelector(`[data-a="${a}"]`).classList.toggle("hidden", !on);
    if (r.status === "running") { s.className = "chipst run"; s.textContent = "● Running"; }
    else if (r.status === "waiting") { s.className = "chipst draft"; s.textContent = "● Waiting for your approval"; }
    else if (r.status === "error") { s.className = "chipst bad"; s.textContent = "✗ Error"; }
    else if (r.status === "stopped") { s.className = "chipst bad"; s.textContent = "■ Stopped"; }
    else if (r.review === "reviewed") { s.className = "chipst done"; s.textContent = `✓ Reviewed by ${r.reviewed_by} · ${when(r.reviewed_at)}`; }
    else { s.className = "chipst draft"; s.textContent = "● AI draft · needs review"; }
    show("review", r.status === "done" && r.review !== "reviewed");
    show("export", r.status === "done" || r.status === "stopped");
  }

  apply(ev) {
    if (this.approvalPending && !["paused", "approval"].includes(ev.type)) this.resolveApproval();
    switch (ev.type) {
      case "run":
        Object.assign(this.row, { id: ev.run_id, thread_id: ev.thread_id, created: ev.created });
        this.k("q").textContent = ev.question;
        this.k("id").textContent = `Run ${ev.run_id}`;
        this.k("time").textContent = when(ev.created);
        this.k("ver").textContent = `agent ${ev.agent_version}`;
        this.k("scope").textContent = `Searched ${ev.scope.length} documents: ${ev.scope.join(", ")} · Public documents only`;
        this.setStatus();
        break;
      case "step": {
        const st = this.el.querySelector(`[data-step="${ev.agent}"]`);
        st.className = `step ${ev.status === "start" ? "run" : "done"}`;
        st.querySelector("span").textContent = ev.status === "start" ? "working…"
          : `${ev.detail}${ev.ms != null ? ` · ${ev.ms < 1000 ? ev.ms + " ms" : (ev.ms / 1000).toFixed(1) + " s"}` : ""}`;
        break;
      }
      case "plan":
        ev.subquestions.forEach((q, i) => this.k("steps").insertAdjacentHTML("beforeend", `<li>Plan · lookup ${i + 1}: ${esc(q)}</li>`));
        break;
      case "tool":
        this.k("steps").insertAdjacentHTML("beforeend",
          `<li>${esc(ev.agent)} · <span class="tool">${esc(ev.name)}</span>(“${esc(ev.args.query ?? "")}”) → ${esc(ev.summary)}</li>`);
        break;
      case "claim": {
        const i = this.claims.push(ev) - 1;
        const n = this.claims.filter((c) => c.verified).length;
        this.k("evCard").classList.remove("hidden");
        this.k("ev").insertAdjacentHTML("beforeend", ev.verified
          ? `<tr class="v" data-claim="${i}"><td class="n">${n}</td><td class="ck">✓</td><td>“${esc(ev.quote)}”</td><td class="where">${esc(short(ev.doc_name))} · p.${ev.page}<span>↗</span></td></tr>`
          : `<tr class="x" data-claim="${i}"><td class="n">–</td><td class="ck">✗</td><td>“${esc(ev.quote)}”<br><em class="why">Removed: ${esc(ev.reason || "failed the checks")}</em></td><td class="where">${esc(short(ev.doc_name))} · p.${ev.page}</td></tr>`);
        break;
      }
      case "approval":
        this.approvalPending = true;
        this.k("approve").className = "approve";
        this.k("approve").innerHTML = `<p><b>Your approval is needed.</b> ${esc(ev.reason)}</p>
          <button class="btn pri" data-a="yes">${icon("check")}Show answer</button><button class="btn" data-a="no">Stop here</button>`;
        break;
      case "paused": this.row.status = "waiting"; this.setStatus(); break;
      case "brief": this.renderBrief(ev); break;
      case "metrics":
        this.m("Time").textContent = `${(ev.latency_ms / 1000).toFixed(2)} s`;
        this.m("Verified").textContent = `${ev.verified}/${ev.claims}`;
        this.m("Cost").textContent = `$${ev.cost_usd.toFixed(3)}`;
        this.m("Tool calls").textContent = ev.tool_calls;
        this.m("Model").textContent = ev.model;
        this.m("Tokens").textContent = ev.tokens.toLocaleString();
        this.k("badge").classList.remove("wait");
        this.row.status = "done"; this.setStatus();
        break;
      case "error":
        this.k("brief").className = "read wait";
        this.k("brief").textContent = ev.message;
        this.row.status = /Stopped/.test(ev.message) ? "stopped" : "error";
        this.setStatus();
        break;
    }
  }

  resolveApproval() {
    this.approvalPending = false;
    this.k("approve").className = "approve done";
    this.k("approve").innerHTML = `The matches were weak, so the agents asked before answering. Your decision is in the audit log.`;
  }

  renderBrief(ev) {
    this.cites = ev.citations;
    const known = new Set(ev.citations.map((c) => c.n));
    let h = esc(ev.markdown)
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/(^|\s)_(.+?)_(?=\s|$)/g, '$1<em class="why">$2</em>')
      .replace(/\s?\[(\d+)\]/g, (m, n) => (known.has(+n) ? `<a class="c" data-cite="${n}" title="Open source ${n}">${n}</a>` : m));
    h = h.split(/\n\n+/).map((b) => b.split("\n").every((l) => l.startsWith("- "))
      ? `<ul>${b.split("\n").map((l) => `<li>${l.slice(2)}</li>`).join("")}</ul>` : `<p>${b.replace(/\n/g, "<br>")}</p>`).join("");
    this.k("brief").className = "read";
    this.k("brief").innerHTML = h;
  }

  async review() {
    try {
      Object.assign(this.row, await json(`/api/runs/${this.row.id}/review`, { method: "POST" }));
      this.setStatus(); loadThreads(); toast("Marked as reviewed. Recorded in the audit log.");
    } catch (e) { toast(e.message); }
  }

  async decide(approve) {
    this.resolveApproval();
    this.row.status = "running"; this.setStatus();
    await streamInto(this, fetch(`/api/runs/${this.row.id}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approve }) }));
  }
}
const short = (n) => (n.length > 30 ? n.slice(0, 28) + "…" : n);

/* ================================================================ asking + streaming */
async function streamInto(run, respPromise) {
  busy = true; $("#sendBtn").disabled = true;
  try {
    const resp = await respPromise;
    if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || resp.statusText);
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        run.apply(ev);
        if (ev.type === "run" && current !== ev.thread_id) { current = ev.thread_id; setHash(`t=${current}`); loadThreads(); }
      }
      $("#msgs").scrollTo({ top: $("#msgs").scrollHeight, behavior: "smooth" });
    }
  } catch (e) {
    run.apply({ type: "error", message: e.message });
  } finally {
    busy = false; $("#sendBtn").disabled = false; loadThreads();
  }
}

$("#askForm").onsubmit = (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question || busy) return;
  $("#question").value = ""; autosize();
  if (!current) $("#msgs").innerHTML = "";
  const run = new Run({ status: "running", review: "draft" }, !!$("#msgs .run"));
  $("#msgs").append(run.el);
  $("#question").placeholder = "Ask a follow-up question…";
  streamInto(run, fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: current }) }));
};
const autosize = () => { const t = $("#question"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; };
$("#question").addEventListener("input", autosize);
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#askForm").requestSubmit(); });

/* ================================================================ navigation + conversations */
function setHash(h) { history.replaceState(null, "", `#${h}`); }

function showView(name) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  $$(".nav").forEach((n) => n.classList.toggle("on", n.dataset.view === name));
  closeDrawer();
  if (name === "documents") loadDocs();
  if (name === "evals") loadEvals();
  if (name === "settings") loadSettings();
}
$$("[data-view]").forEach((n) => (n.onclick = () => {
  const v = n.dataset.view;
  if (v === "thread") return current ? openThread(current) : newQuestion();
  current = null; setHash(v); showView(v); loadThreads();
}));

let searchTimer;
$("#search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadThreads, 200); });

async function loadThreads() {
  const q = $("#search").value.trim();
  const list = await json(`/api/threads${q ? `?q=${encodeURIComponent(q)}` : ""}`).catch(() => []);
  const today = new Date().toDateString();
  const groups = q ? { [`Results for “${q}”`]: list } : { Today: [], Earlier: [] };
  if (!q) list.forEach((t) => groups[new Date(t.updated).toDateString() === today ? "Today" : "Earlier"].push(t));
  $("#threads").innerHTML = Object.entries(groups).filter(([, v]) => v.length).map(([g, v]) =>
    `<div class="h">${esc(g)}</div>` + v.map((t) => `
      <div class="it ${t.id === current ? "on" : ""}" data-thread="${t.id}" title="${esc(t.title)}">
        <span class="t">${esc(t.title)}</span>${t.reviewed && t.reviewed === t.runs ? `<span class="rv" title="All answers reviewed">✓</span>` : ""}
        <button class="del" data-del="${t.id}" title="Delete conversation">${icon("x")}</button></div>`).join("")).join("")
    || `<div class="none">${q ? "No matches." : "No research yet. Ask your first question."}</div>`;
}

$("#threads").onclick = async (e) => {
  const del = e.target.closest("[data-del]");
  if (del) {
    e.stopPropagation();
    if (!confirm("Delete this conversation? Its answers are removed; the audit log is kept.")) return;
    await json(`/api/threads/${del.dataset.del}`, { method: "DELETE" });
    if (current === del.dataset.del) newQuestion();
    return loadThreads();
  }
  const t = e.target.closest("[data-thread]");
  if (t) openThread(t.dataset.thread);
};

async function openThread(id) {
  let t;
  try { t = await json(`/api/threads/${id}`); } catch { return newQuestion(); }
  current = id; setHash(`t=${id}`); showView("thread");
  $("#msgs").innerHTML = "";
  t.runs.forEach((r, i) => {
    const run = new Run({ ...r }, i > 0);
    $("#msgs").append(run.el);
    r.events.forEach((ev) => run.apply(ev));
    Object.assign(run.row, { status: r.status, review: r.review, reviewed_by: r.reviewed_by, reviewed_at: r.reviewed_at });
    if (r.status === "waiting" && !r.waiting) run.row.status = "stopped";
    if (run.approvalPending && !r.waiting) run.resolveApproval();
    run.setStatus();
  });
  $("#question").placeholder = "Ask a follow-up question…";
  $("#msgs").scrollTop = $("#msgs").scrollHeight;
  loadThreads();
}

function newQuestion() {
  current = null; setHash("new"); showView("thread");
  $("#question").placeholder = "Ask a research question…";
  $("#msgs").innerHTML = `
    <div class="welcome">
      <img src="icon.svg" width="64" height="64" alt="">
      <h1>Agentic <em>Researcher</em></h1>
      <p class="tagline">Answers you can verify.</p>
      <p>AI agents plan, search and write; a critic checks every quote against its page; a person signs off.
         Ask about your ${status.documents ?? ""} datasheets and papers.</p>
      <div class="how">
        <div><b><span>1</span>Plan</b>Splits your question into lookups.</div>
        <div><b><span>2</span>Search</b>Finds the pages that answer each lookup.</div>
        <div><b><span>3</span>Check</b>Keeps only quotes that really appear on the page.</div>
        <div><b><span>4</span>Write</b>Writes the answer with numbered citations.</div>
      </div>
      <div class="chips">${EXAMPLES.map((x) => `<button type="button" class="chip">${esc(x)}</button>`).join("")}</div>
    </div>`;
  $("#msgs .chips").onclick = (e) => {
    if (e.target.classList.contains("chip")) { $("#question").value = e.target.textContent; autosize(); $("#question").focus(); }
  };
  loadThreads();
  $("#question").focus();
}
$("#newBtn").onclick = newQuestion;

/* ================================================================ drawer: source, audit log, help */
function drawerMode(title, body) {
  $("#dTitle").textContent = title;
  [$("#dTabs"), $("#dQuote"), $("#dPdf"), $("#dOpen")].forEach((x) => x.classList.toggle("hidden", !!body));
  $("#dBody").classList.toggle("hidden", !body);
  if (body) $("#dBody").innerHTML = body;
  $("#drawer").classList.add("open"); $("#drawer").setAttribute("aria-hidden", "false");
}
function closeDrawer() { $("#drawer").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true"); }
$("#dClose").onclick = closeDrawer;
document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());

function openSource(run, idx) {
  const c = run.claims[idx];
  if (!c) return;
  let n = 0;
  $("#dTabs").innerHTML = run.claims.map((x, i) => x.verified
    ? `<button data-i="${i}" class="${i === idx ? "on" : ""}">[${++n}] ${esc(short(x.doc_name))} · p.${x.page}</button>` : "").join("");
  $("#dTabs").onclick = (e) => e.target.dataset.i && openSource(run, +e.target.dataset.i);
  $("#dQuote").className = `quote ${c.verified ? "" : "bad"}`;
  $("#dQuote").innerHTML = `${c.verified ? '<b style="color:var(--ok)">✓ Quote verified on this page</b>' : `<b style="color:var(--bad)">✗ Removed by the critic: ${esc(c.reason || "failed the checks")}</b>`}
    <br>“${esc(c.quote)}”<br><span class="src">${esc(c.doc_name)} · page ${c.page}</span>`;
  $("#dPdf").src = pdfUrl(c.doc, c.page);
  $("#dOpen").href = pdfUrl(c.doc, c.page);
  drawerMode(`Source · ${short(c.doc_name)} · p.${c.page}`);
}

async function openAudit(runId) {
  const log = await json(`/api/runs/${runId}/audit`);
  drawerMode(`Audit log · run ${runId}`, `
    <div class="card"><table class="tbl"><tr><th>Time</th><th>Who</th><th>Action</th><th>Detail</th></tr>${
      log.map((a) => `<tr><td class="where">${esc(when(a.time))}</td><td>${esc(a.actor)}</td><td><b>${esc(a.action)}</b></td><td>${esc(a.detail)}</td></tr>`).join("")}</table></div>
    <p>Append-only record stored in <code>logs/app.db</code>. Export PDF includes this log.</p>`);
}

$("#helpBtn").onclick = () => drawerMode("Help", `
  <h3>What this app does</h3>
  <p><b>Agentic Researcher: answers you can verify.</b> You ask a question about your documents. Four agents work in turn:
     <b>Plan</b> splits the question, <b>Search</b> finds the pages, <b>Check</b> keeps only quotes that really appear on the cited page,
     and <b>Write</b> writes the answer with numbered citations.</p>
  <h3>Verified vs reviewed</h3>
  <p><b>Verified</b> means the critic found the exact quote on the page (automatic).
     <b>Reviewed</b> means a person read the answer and signed it off with <i>Mark reviewed</i>.</p>
  <h3>Checking a source</h3>
  <p>Click a citation number or an evidence row to open the PDF at that page, with the checked quote shown above it.</p>
  <h3>Audit trail</h3>
  <p>Every run records who asked, what the agents did, any approval decision, the review and exports.
     <i>Audit log</i> shows it; <i>Export PDF</i> prints the answer, evidence and log for the file.</p>
  <h3>Low-confidence answers</h3>
  <p>If the matches are weak, the agents pause and ask before showing an answer. Your choice is logged.</p>
  <h3>Models</h3>
  <p>The agent uses the model chosen in Settings (Claude or a local Ollama model) and falls back to the other if the first fails or the per-run cost cap is reached.
     If no model is reachable it runs in <b>preview mode</b>: keyword search and quote checks only, listing verified quotes instead of a written answer.</p>
  <h3>Guardrails</h3>
  <p>The critic removes a claim if its quote is not on the page, if the quote contains instructions aimed at the AI (prompt injection),
     if the claim has numbers or product names the quote does not contain, or (when installed) if the fine-tuned claim checker says the quote does not support it.
     The writer then drops any sentence whose figures are not in the cited evidence.</p>
  <h3>Keyboard</h3>
  <p><code>Ctrl+Enter</code> send · <code>Esc</code> close this panel</p>`);

/* ================================================================ documents + evaluation */
async function loadDocs() {
  const docs = await json("/api/documents");
  $("#docCount").textContent = docs.length;
  $("#docTable").innerHTML = docs.length
    ? `<tr><th>Document</th><th>Type</th><th>Pages</th><th>Size</th></tr>` + docs.map((d) => `
      <tr><td><a href="${pdfUrl(d.id, 1)}" target="_blank">${esc(d.name)}</a></td><td><span class="kind ${d.kind}">${d.kind}</span></td>
      <td class="where">${d.pages}</td><td class="where">${(d.size_kb / 1024).toFixed(1)} MB</td></tr>`).join("")
    : `<tr><td><div class="empty"><strong>No documents yet</strong>Add a PDF above.</div></td></tr>`;
}
async function upload(files) {
  for (const f of files) {
    const fd = new FormData(); fd.append("file", f);
    try { await json(`/api/documents?kind=${$("#kindSel").value}`, { method: "POST", body: fd }); toast(`Added ${f.name}`); }
    catch (e) { toast(e.message); }
  }
  loadDocs();
}
const drop = $("#drop");
$("#fileIn").onchange = (e) => upload(e.target.files);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); };
$("#kindSel").onclick = (e) => e.preventDefault();
$("#kindSel").onmousedown = (e) => e.stopPropagation();

async function loadEvals() {
  const list = await json("/api/evals");
  $("#evalList").innerHTML = list.length ? list.map((e) => `
    <div class="card evalrow"><b>${esc(e.file)}</b> <span class="mono muted">${esc(e.modified)}</span>
      ${e.summary ? `<dl>${Object.entries(e.summary).map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>`
                  : `<div class="muted">No "summary" block in this file.</div>`}</div>`).join("")
    : `<div class="empty"><strong>No evaluation results yet</strong>Phase 1 adds the baseline scores and Phase 3 the agent's.
       Files saved to <code>evals/results/*.json</code> with a <code>"summary"</code> block appear here.</div>`;
}

/* ================================================================ settings */
async function loadSettings() {
  const s = await json("/api/settings");
  const st = s.settings;
  $("#sName").value = st.reviewer_name || s.user;
  $("#confPath").textContent = s.paths.config;
  $("#keyState").className = `pill ${s.key.set ? "ok" : "no"}`;
  $("#keyState").textContent = s.key.set ? `✓ Key saved ${s.key.hint}` : "No key saved";
  $("#clearKey").classList.toggle("hidden", !s.key.set);
  $("#sProvider").value = st.provider;
  $("#sClaude").innerHTML = s.claude_models.map((m) => `<option ${m === st.anthropic_model ? "selected" : ""}>${m}</option>`).join("");
  const local = s.ollama_models.length ? s.ollama_models : [st.ollama_model];
  if (!local.includes(st.ollama_model)) local.unshift(st.ollama_model);
  $("#sOllama").innerHTML = local.map((m) => `<option ${m === st.ollama_model ? "selected" : ""}>${esc(m)}</option>`).join("")
    + (s.ollama_models.length ? "" : `<option disabled>(Ollama not running)</option>`);
  $("#sCap").value = st.cost_cap_usd;
  $("#sChecker").value = st.claim_checker;
  $("#sTransport").value = st.tool_transport;
  $("#checkerInfo").textContent = s.claimcheck.available
    ? `Fine-tuned claim checker installed (${s.claimcheck.base_model || "model"}, trained ${s.claimcheck.trained || "?"}).`
    : "Fine-tuned claim checker not installed (models/claimcheck/); the critic uses its rule checks.";
  $("#about").innerHTML = [
    ["App", `Agentic Researcher ${s.app_version}`], ["Mode", `${s.mode} · agent ${s.agent_version}`],
    ["Documents", s.paths.documents], ["Database & audit log", s.paths.database], ["Settings", s.paths.config],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
}
async function saveSetting(patch, msg) {
  try { await send("/api/settings", "PUT", patch); toast(msg); refreshStatus(); }
  catch (e) { toast(e.message); }
}
$("#saveName").onclick = () => saveSetting({ reviewer_name: $("#sName").value }, "Name saved. New reviews use it.");
$("#saveModels").onclick = () => saveSetting({ provider: $("#sProvider").value, anthropic_model: $("#sClaude").value,
  ollama_model: $("#sOllama").value, cost_cap_usd: +$("#sCap").value,
  claim_checker: $("#sChecker").value, tool_transport: $("#sTransport").value }, "Model settings saved.");
$("#saveKey").onclick = async () => {
  try { await send("/api/settings/key", "POST", { key: $("#sKey").value }); $("#sKey").value = ""; toast("Key saved privately."); loadSettings(); }
  catch (e) { toast(e.message); }
};
$("#clearKey").onclick = async () => {
  if (!confirm("Remove the saved API key from this computer?")) return;
  await json("/api/settings/key", { method: "DELETE" }); toast("Key removed."); loadSettings();
};

/* ================================================================ theme, status, start */
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  const dark = (document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  $("#themeBtn").innerHTML = icon(dark ? "moon" : "sun");
}
try { applyTheme(localStorage.getItem("theme")); } catch { applyTheme(); }
$("#themeBtn").onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch {}
};

async function refreshStatus() {
  try {
    status = await json("/api/status");
    $("#sysDot").className = "dot on"; $("#sysText").textContent = "System online";
    $("#sysVer").textContent = `v${status.app_version} · ${status.mode} mode`;
    $("#modeChip").textContent = status.mode === "agent" ? `Agent ${status.agent_version}` : "Preview mode";
    $("#modeChip").classList.toggle("agent", status.mode === "agent");
    $("#modeChip").title = status.mode === "agent" ? "" : "Keyword search and quote checks only, no AI model yet (Phase 2 adds the agent)";
    $("#docCount").textContent = status.documents;
    $("#avatar").textContent = status.user.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";
    $("#avatar").title = `${status.user} · Settings`;
  } catch {
    $("#sysDot").className = "dot off"; $("#sysText").textContent = "Server offline";
  }
}
setInterval(refreshStatus, 30000);

(async () => {
  await refreshStatus();
  const h = location.hash.slice(1);
  const preset = new URLSearchParams(location.search).get("q");
  if (h.startsWith("t=")) await openThread(h.slice(2));
  else if (["documents", "evals", "settings"].includes(h)) { showView(h); loadThreads(); }
  else newQuestion();
  if (preset) { $("#question").value = preset; $("#askForm").requestSubmit(); }
})();
