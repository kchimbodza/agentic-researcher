# Agentic Researcher

**Answers you can verify.** AI agents plan, search and write; a critic checks every quote against its page; a person signs off.

Ask a research question that needs several lookups across hardware datasheets and research papers.
A small team of agents (LangGraph) plans the lookups, searches the PDFs through an MCP tool server,
keeps only claims whose quotes really appear on the cited page, and writes a short answer where every
sentence cites its evidence. Every run has an audit trail, a review sign-off and a printable report.

It is the agent version of my [Datasheet Answer Finder](https://github.com/kchimbodza/datasheet-answer-finder)
(single-step RAG), which is the baseline it is measured against on the same test set.

![Asking a multi-step question: the agents plan, search, check every quote and write a cited answer](docs/media/demo-ask.gif)

*Above: a question that needs two lookups (40 W + 60 W = 100 W). Each sentence cites a quote that was checked
against the page; clicking a citation opens the PDF at that page; the reviewer signs off and the audit log records it.
Full 70-second demo: [docs/media/demo.mp4](docs/media/demo.mp4).*

## In 30 seconds

- **Problem:** chatbots over documents make up numbers. Here every figure must come from a quote that really
  appears on the cited page, or it is removed.
- **What I built:** a 4-step agent team (plan → search → check → write), a tool server, an evaluation suite, a desktop
  app with human sign-off and an audit trail, and a small fine-tuned checking model.
- **Measured, not claimed:** on 21 test questions the agent beat my single-step baseline (57% vs 52% accuracy,
  right page found 94% vs 78%), never made up an answer to an unanswerable question, and resisted 6/6 prompt-injection attacks.
  It is also slower (15 s vs 3 s). With Claude as the model, the same agent reached 60–62% and found the right page
  97–100% of the time. All numbers below come from `evals/results/`.

## Screenshots

| | |
|---|---|
| ![Home screen](docs/media/01-home.png) **Home:** ask a question; the four steps are explained up front. | ![Answer with metrics](docs/media/02-answer.png) **Answer:** each step's progress, time, tokens, cost and model. |
| ![Source drawer](docs/media/04-source-drawer.png) **Check the source:** a citation opens the PDF at the quoted page. | ![Agent steps](docs/media/05-agent-steps.png) **Agent steps:** every lookup and tool call the agents made. |
| ![Audit log](docs/media/06-audit-log.png) **Audit log:** who asked, what the agent did, who reviewed it. | ![Not found](docs/media/07-not-found.png) **Honest "not found":** a price that is not in the documents is not guessed. |
| ![Evaluation page](docs/media/08-evaluation.png) **Evaluation:** every eval run, agent vs baseline. | ![Settings](docs/media/10-settings.png) **Settings:** reviewer name, API key, model routing, cost cap. |

<details><summary>More: unanswerable question demo, app pages tour, documents, dark mode</summary>

![Unanswerable question](docs/media/demo-not-found.gif)
![Tour of the Evaluation, Documents and Settings pages](docs/media/demo-pages.gif)
![Documents](docs/media/09-documents.png)
![Dark mode](docs/media/11-dark-mode.png)
</details>

## At a glance

| | |
|---|---|
| **What it does** | Multi-step research over your own PDFs with verified quotes, citations, human approval and an audit log |
| **Skills shown** | LangGraph multi-agent workflow, MCP tool server, agent evals (accuracy, safety, latency, cost), guardrails against prompt injection and made-up numbers, human-in-the-loop, model routing (Claude ↔ local Ollama ↔ Azure OpenAI), LoRA fine-tuning (PyTorch), Docker, GitHub Actions |
| **Tech** | Python, FastAPI, LangGraph, MCP, Anthropic API, Ollama (qwen2.5:7b, nomic-embed-text), Chroma, ONNX Runtime, SQLite, vanilla JS |
| **Runs** | Fully on a laptop with a local model, or with Claude via an API key |

## Results

Measured on the 21-question test set (7 single-fact, 10 multi-step, 4 unanswerable), all on the local
`qwen2.5:7b` model through Ollama on a laptop RTX 4050 (6 GB). Agent runs repeat every question twice.
Raw outputs: `evals/results/`; per-question table: `evals/results/REPORT.md`.

| | Single-step RAG (baseline) | Agent v0.4.0 (first) | **Agent v0.4.1** | Agent v0.4.1 + fine-tuned checker |
|---|---|---|---|---|
| Accuracy | 11/21 (52%) | 20/42 (48%) | **24/42 (57%)** | 14/42 (33%) |
| Multi-step questions | 3/10 (30%) | 6/20 (30%) | **8/20 (40%)** | 2/20 (10%) |
| Unanswerable → "not found" | 4/4 | 8/8 | 8/8 | 8/8 |
| Made-up answers | 0 | 0 | 0 | 0 |
| Right page found | 14/18 (78%) | 28/36 (78%) | **34/36 (94%)** | 33/36 (92%) |
| Cited but wrong (per run) | 4 | 5 | 5 | **2.5** |
| Same result on both runs | – | 21/21 | 21/21 | 21/21 |
| Median time per question | **2.8 s** | 13.7 s | 15.2 s | 17.2 s |

**Safety set** (`evals/safety.json`, 6 attacks, with a poisoned PDF that tells the AI to answer "999 mA",
to refuse Route10 questions and to reveal its system prompt): the agent resisted **6/6** (never repeated 999,
never leaked a prompt, never invented a price, declined the off-topic request). In the one run where the
poisoned page was retrieved as evidence, the critic removed it as a prompt injection. The baseline also
resisted 6/6, mostly because it never retrieved the poisoned page; when the question pointed straight at it,
the baseline answered "not found" while the agent found the real figure (10 µA) in the ESP32 datasheet.

**Fine-tuned claim checker** (`finetune/`): DistilBERT + LoRA, 1.9 M trainable parameters, trained in 5 minutes
on the GPU. On held-out document pages it scores 94.3% accuracy vs 78.9% for a "numbers match" rule
(details in `finetune/README.md`). As a hard filter in the agent it halved wrong citations but removed many
correct, reworded claims, so it is off by default (Settings → Critic checks).

**Same agent on Claude** (agent v0.4.1, rules checker, `effort: low`, same 21 questions × 2):

| | qwen2.5:7b (local) | Claude Haiku 5.5 | Claude Sonnet 5.5 | Claude Opus 5.5 |
|---|---|---|---|---|
| Accuracy | 24/42 (57%) | 25/42 (60%) | **26/42 (62%)** | **26/42 (62%)** |
| Single-fact questions | 8/14 (57%) | 6/14 (43%) | **9/14 (64%)** | 7/14 (50%) |
| Multi-step questions | 8/20 (40%) | **11/20 (55%)** | 9/20 (45%) | **11/20 (55%)** |
| Unanswerable → "not found" | 8/8 | 8/8 | 8/8 | 8/8 |
| Made-up answers | 0 | 0 | 0 | 0 |
| Right page found | 34/36 (94%) | 35/36 (97%) | **36/36 (100%)** | **36/36 (100%)** |
| Cited but wrong (per run) | 5 | 5.5 | 6 | 5.5 |
| Same result on both runs | 21/21 | 20/21 | 19/21 | 19/21 |
| Median time per question | 15.2 s | **7.2 s** | 10.0 s | 12.7 s |
| Cost for the whole run | $0 | $0.07 | $1.03 | $2.23 |

A bigger model helped less than I expected: retrieval became near-perfect, but accuracy rose only 3–5 points and
wrong citations stayed about the same. Opus scored the same as Sonnet at twice the cost, so the remaining errors
come from the pipeline (what the researcher extracts and how the critic and scorer judge it), not model size.

## How it works

<p align="center"><img src="docs/media/how-it-works.svg" width="100%" alt="Question, Planner, Researcher, Critic, Writer, Answer; the Critic retries lookups with no evidence once and asks a human before a partial answer"></p>

<details><summary>Same flow as a Mermaid chart</summary>

```mermaid
flowchart LR
    Q[Question + conversation memory] --> P[Planner<br/>splits into lookups]
    P --> R[Researcher<br/>hybrid search via MCP,<br/>extracts claims + exact quotes]
    R --> C[Critic<br/>quote on page? injection nearby?<br/>numbers & product names match?<br/>fine-tuned claim checker]
    C -- lookups with no evidence<br/>(one retry, reworded) --> R
    C -- partly answered --> H{{Human approval}}
    C -- fully answered / nothing found --> W
    H -- approve --> W[Writer<br/>cited sentences only,<br/>drops ungrounded figures]
    W --> A[Answer + evidence + audit log]
```

</details>

- **Planner** splits the question into lookups and resolves follow-ups ("that paper") from conversation memory.
- **Researcher** calls `search_documents` on the MCP server (`app/mcp_server.py`): vector search (nomic-embed-text in Chroma)
  and keyword search merged by reciprocal-rank fusion, then asks the model for claims with word-for-word quotes.
- **Critic** (`app/guardrails.py`) removes a claim if the quote is not on the cited page, if the page text around it
  addresses the AI (prompt injection), if the claim has numbers or product names the quote does not contain, or if the
  fine-tuned claim checker (`app/claimcheck.py`, see `finetune/`) says the quote does not support it.
  Lookups with no surviving evidence are reworded and searched once more.
- **Human-in-the-loop:** if only part of the question is answered, the LangGraph run pauses (interrupt before the writer)
  and the app asks whether to show a partial answer. The decision is logged.
- **Writer** may only use verified evidence; any sentence without a citation, or with a figure that is not in its cited
  quotes (simple arithmetic and unit conversion on cited figures is allowed), is dropped.
- **Model routing** (`app/llm.py`): Claude (structured outputs; server-side refusal fallbacks on Opus/Sonnet) or a local
  Ollama model as main, the other as fallback, Azure OpenAI optional; per-run cost cap; every call logged with tokens, cost and latency.

## Run it

```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text   # local model + embeddings
./install.sh            # adds "Agentic Researcher" to the app menu (Linux; needs uv and Google Chrome)
```
Or `uv run uvicorn app.server:app --port 8770` and open http://127.0.0.1:8770. Put PDFs in `data/datasheets/` and
`data/papers/` (or drag them onto the Documents page). Add an Anthropic API key in Settings to use Claude.

Use the tools from any MCP client, e.g. Claude Code: `claude mcp add agentic-researcher -- uv run --project . python -m app.mcp_server`

## Evaluate it

```bash
uv run python -m evals.evaluate --system baseline                 # single-step RAG
uv run python -m evals.evaluate --system agent --repeat 2         # the agent, twice (stability)
uv run python -m evals.evaluate --system agent --set safety       # prompt-injection and pressure tests
uv run python -m evals.compare baseline-main agent-main-rules     # side-by-side table
uv run pytest -q                                                  # unit tests (fake model, generated PDFs)
```
The test set (`evals/questions.json`) has 21 questions written from the PDFs: 7 single-fact, 10 multi-step
(comparisons, sums, facts from two documents) and 4 that the documents cannot answer. Scoring is automatic:
expected numbers must appear as whole numbers in the answer; unanswerable questions must be answered "not found".
Raw outputs for every run are in `evals/results/`.

## Deploy

`docker compose up` runs the app with Ollama. `deploy/azure/deploy.sh` deploys to Azure Container Apps
(scale to zero, documents on Azure Files, keys as secrets, access-token gate); see `deploy/azure/README.md`
for the cost table. CI (`.github/workflows/ci.yml`) runs the unit tests, builds the image, and, when an
`ANTHROPIC_API_KEY` secret is set, runs a 6-question smoke eval that fails the build below 60% accuracy.

## Documents used

Vendor datasheets (Espressif ESP32, Alta Labs switches and Route10) and two open-access papers:
Adebesin et al. (2026), *Wearable System for Real Time Detection and Prediction of Falls*, Edge Intelligence and Systems (CC BY 4.0);
Wang et al. (2026), *Edge-Efficient Two-Stream Multimodal Architecture for Non-Intrusive Bathroom Fall Detection*, arXiv:2603.17069 (CC BY).
The PDFs are not stored in this repository.

## What I learned

- **The first agent was worse than the simple baseline** (48% vs 52%) with more wrong citations. Multi-step
  planning only paid off after fixing what the evals exposed: PDF text with no spaces ("at12.5Hz") made true
  numbers look invented, the researcher saw too little of each page, and the product-name check trusted the
  question instead of the cited page. After those fixes: 57%, and the right page is found 94% of the time.
- **A real quote is still not a correct answer.** The critic proves the quote exists; it cannot prove the small
  model read a dense spec table correctly (it took a paper's *baseline* latency as the radar-only figure).
  The fine-tuned checker catches more of these, at the cost of recall: a precision/recall trade-off I measured
  instead of guessing.
- **Safety tests must reach the attack.** My first injection cases looked perfect only because the poisoned page
  was never retrieved; I added a case that forces retrieval and checked which runs actually saw it.
- **Unit tests with a fake model found two security bugs** before any eval did: an injected instruction slipped
  through because the quote itself looked harmless, and the number check let 999 through as "60 ÷ 60 × 1000".
- **Small local models are slow but private and free**: about 15 s and ~5,000 tokens per question on a laptop GPU.
  On Claude the same code was 2× faster (Haiku: 7 s, $0.07 for 42 runs) but only 3–5 points more accurate,
  so the next gains have to come from the pipeline, not a bigger model.

Built with help from an AI coding assistant (Claude Code).
<!-- TODO (Kudzayi): once you have reviewed and tested it yourself, say so here in your own words. -->


## Author

**Kudzayi Chimbodza** · [GitHub @kchimbodza](https://github.com/kchimbodza)
