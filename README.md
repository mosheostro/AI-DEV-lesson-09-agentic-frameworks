# NovaOps Supervisor — LangGraph agents over MCP, with a human on the write

An IT-support assistant built as a **supervisor over specialist sub-agents**. An employee writes
in — *"I'm E010 and Webex says I'm not licensed — can you help?"* — and the supervisor decides
which specialists to consult, runs them in parallel against a local MCP server, merges their
evidence, and may propose one state-changing action: filing an access request. That action only
happens after deterministic checks **and** an explicit human approval.

The project demonstrates:

- **LangGraph orchestration** — structured-output planning, parallel fan-out with `Send` and a
  reducer, `interrupt()` for human approval, and a SQLite checkpointer for durable pauses.
- **MCP as the tool boundary** — every tool is discovered at runtime from the NovaOps MCP server
  through `langchain-mcp-adapters`; nothing about the server is imported or reimplemented.
- **Safety around a single write** — read-only specialists, a deterministic pre-approval check,
  a fail-closed approval gate, exact checkpointed arguments, duplicate protection, and explicit
  recovery rules when the outcome of a write is uncertain.

This is an educational / portfolio project, not production software.

## Architecture

```
plan ──► worker × k (parallel) ──► act ──► verify ──► review (human) ──► write ──► respond
  │                                 │ │      │           └─ reject ──────────────► respond
  ├─ empty plan ───────────────────►┘ │      └─ check fails ─────────────────────► respond
  └─ invalid plan twice ─► end        └─ nothing proposed ─► end (answer from act)
```

| Step | What it does |
|---|---|
| **plan** | The model picks the specialists this request needs (`knowledge`, `records`, `eligibility`, or none) as structured output from a closed set; invalid output is retried once, then the run ends with a fixed error. A bulk access request (more than one employee, or "all software") ends here with a fixed reply. |
| **worker** | Each chosen specialist is its own agent with 2–3 **read-only** MCP tools, at most 3 tool calls and a step limit; they run in one parallel superstep and their findings merge through a reducer. A failed or incomplete specialist disables the write path. |
| **act** | The only step that can propose `create_access_request`, and only when the planner judged a filing relevant and every finding is complete. |
| **verify** | Deterministic, read-only checks before anyone is asked: the employee ID must exist exactly as the server will file it; the software must be named in the employee's request, must match exactly one subscription by exact name (the server itself matches names by substring), and must be over its seat limit. Otherwise nothing is filed and the employee gets a fixed reply: it asks for the employee ID or the exact software name, or — when seats are free — says no request is needed and to open an IT ticket. An employee ID that a specialist's lookup did not find gets the same fixed reply even when nothing was proposed. |
| **review** | `interrupt()` shows the approver the original request, the evidence, the resolved system and its seats, and the exact proposed arguments; the business justification is marked as written by the model, to check against the request. Only an explicit approval proceeds; anything else is a rejection. |
| **write** | Sends the checkpointed arguments unchanged — the model never regenerates them — at most once per employee + software within a run; a server result naming a different system is flagged. |
| **respond** | Writes the final reply with no tools bound. User-visible replies are capped at `ANSWER_MAX_TOKENS` (800); a reply cut off at that limit is discarded, retried once, then replaced by a fixed text. |

**Persistence and recovery.** The CLI checkpoints every run to `checkpoints.sqlite` (next to
`supervisor.py`), so a request paused at the approval prompt can be resumed in a new process.
A run that stopped anywhere else is **never** resumed automatically: if a write may have reached
the server, the CLI says the outcome is uncertain and requires manual verification instead of
retrying it.

The full contract — state, node rules, approval payload, error policy, test plan and decisions —
is in [`SPEC.md`](SPEC.md).

## Requirements

- **Python 3.11 or newer** (the code uses 3.11 features). Developed and tested with Python 3.14.3
  on Windows 11.
- An **AWS account with Amazon Bedrock access** in your region to:
  - the chat model `us.amazon.nova-2-lite-v1:0` (used by the supervisor), and
  - the embedding model `amazon.titan-embed-text-v2:0` (used by the MCP server's knowledge-base
    search).
- Dependencies from `requirements.txt` (LangChain, LangGraph, `langchain-aws`,
  `langchain-mcp-adapters`, the MCP SDK pinned to `mcp<2`, FAISS, and
  `langgraph-checkpoint-sqlite==3.1.2`).

## Setup

### Option A — setup script

From the repository root, in **Git Bash** (Windows), macOS, Linux or WSL:

```bash
source setup.sh
```

In PowerShell, `.\setup.ps1` finds Git Bash and hands over to `setup.sh`.

The script creates `.env` from `.env.example` if it is missing (fill in your credentials and run
it again), creates and activates `.venv`, installs `requirements.txt`, and finally calls the
Bedrock chat and embedding models once to confirm access. Use `source`, not `./setup.sh`, so the
virtual environment stays active in your shell.

### Option B — manual

```bash
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
```

### Configuration (`.env`)

`supervisor.py` requires these variables and stops with exit code 2, naming only the missing
ones, if any is absent:

| Variable | Example / meaning |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | your AWS credentials (never commit them) |
| `AWS_REGION` | `us-east-1` |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-2-lite-v1:0` |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` |
| `MCP_SERVER_URL` | `http://127.0.0.1:9876/mcp` |

`.env.example` also lists a few variables from the wider course (`GOOGLE_API_KEY`,
`GEMINI_MODEL_ID`, `OPENSEARCH_COLLECTION`) that this project does not use. `.env` is
git-ignored.

## Running

Start the MCP server in one terminal and leave it running. It builds its knowledge-base index on
startup and serves on `http://127.0.0.1:9876/mcp`:

```bash
cd server && python server.py
```

Run the supervisor from the repository root in a second terminal:

```bash
python supervisor.py                       # the Webex demo request, once
python supervisor.py --demo                # same as above
python supervisor.py "request text"        # one request, then exit
python supervisor.py -- "-request text"    # a request that starts with "-"
python supervisor.py -i                    # interactive: several requests (exit/quit to leave)
python supervisor.py --resume THREAD_ID    # continue a request paused at its approval prompt
python supervisor.py --status [THREAD_ID]  # saved requests (or one) and the next step
python supervisor.py --help
```

- **Approval prompt:** answer `approve`, `yes`, `y` or `ok` — or `reject`, `no`, `n` (with an
  optional reason). Anything else asks again. To decide later press **Ctrl-C** (or end the input:
  Ctrl-Z then Enter on Windows, Ctrl-D on macOS/Linux); the printed `--resume` command continues
  the request in a new process.
- **`--help` and `--status`** need no server, credentials or model: `--help` reads nothing, and
  `--status` only reads `checkpoints.sqlite` to classify each saved request (awaiting approval,
  finished, or stopped — with the safe next step).
- **Exit codes:** 0 done, paused, help or status · 1 runtime error · 2 configuration or usage
  error, or unknown thread · 3 not resumed automatically (stopped outside an approval pause) ·
  130 interrupted while running.

The server keeps its database in memory: access requests filed during a session disappear when
the server restarts.

## Testing

From the repository root, with the dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The suite (107 tests, standard-library `unittest`) is fully **offline**: a scripted chat model and
fake MCP tools stand in for Bedrock and the server, persistence tests use temporary SQLite files,
and no `.env`, credentials or network access are needed. It covers planning, parallel fan-out,
specialist limits, the pre-approval check (employee ID, software name, seat limit), approval and
rejection input, duplicate writes, truncated replies, the CLI modes, Ctrl-C handling,
pause/resume across processes, and uncertain-write recovery.

There are no automated live tests. End-to-end checks against a running server and real Bedrock
are described in `SPEC.md` §12 ("Live checks"); they need the setup above, and an approved request
really files a row on the local server.

## Limitations

These are documented in `SPEC.md` (§8 and §13); the main ones:

- **At-least-once, not exactly-once.** The server has no idempotency key and no tool to read
  access requests. Duplicate protection covers one run only; if a process dies after the server
  commits but before the checkpoint is saved, the outcome is unknown and must be verified by hand.
- **Name matching.** The server resolves software names by substring. `verify` proves the name is
  an exact subscription name, but filing searches all systems, so a system without a subscription
  whose name contains the same text could still be chosen — this is only detected after filing.
- **The request must name the software literally.** Descriptive requests ("my video-call app"),
  aliases and non-English names are not filed; the employee is asked for the exact name.
- **Model behaviour is not deterministic.** The planner may choose a different team, a specialist
  may run out of its tool budget (which disables filing), and the model's own wording can still
  mention a substring candidate when one request mixes a valid and an unmatched name.
- **Reply wording is guided by the prompt, not guaranteed.** The model is told never to claim that
  access was granted, never to name internal tools, and never to invent a justification or a
  rejection reason; only the fixed replies and the `Outcome:` line are written by code. The
  approver must check the model-written justification against the request.
- **One employee per request.** A bulk request — several employees ("all employees", "E001 to
  E012") or software not named one by one ("all software") — gets a fixed reply asking for one
  request per employee; the planner model decides what is bulk. Quantities ("x 2") are not
  represented, and one employee with several named systems can still exhaust a specialist's
  3-call budget, which disables filing.
- **Durability lives in the CLI.** `build_graph(tools)` on its own pauses in memory; the CLI adds
  the SQLite checkpointer.
- **The approval decision is not recorded on the server,** and the server's data is in memory.

## Repository layout

| Path | Contents |
|---|---|
| `supervisor.py` | the supervisor graph and the CLI |
| `SPEC.md` | the authoritative specification: contracts, decisions, tests, limitations |
| `tests/` | offline test suite and its fakes |
| `server/`, `data/` | the NovaOps MCP server and its seed data, policies and knowledge base |
| `setup.sh`, `setup.ps1` | environment setup and Bedrock access check |
| `CLAUDE.md` | build conventions used while developing with a coding agent |
| `PROMPT.md` | the original exercise brief this build started from |

## Attribution

The NovaOps MCP server (`server/`), its dataset (`data/`), the setup scripts and the exercise brief
(`PROMPT.md`) come from the **AI Development course** (Lessons 8–9, agentic frameworks). The
supervisor, its specification and its tests were built for this project on top of them.
