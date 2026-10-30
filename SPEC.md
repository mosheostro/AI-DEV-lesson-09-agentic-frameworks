# NovaOps Supervisor — Implementation Specification

> **Standalone repository note.** This specification was written while the project lived inside
> the AI Development course's reference repository. Mentions of `../code/` (the course's reference
> implementations), `validate-lab/` (the course's grading rubric), `.README.orig.md` (a local backup
> of the original course README), and "this repository's history" / the `origin` URL in §14
> describe that original context; none of them is part of this repository. `README.md` documents
> how to install, run and test this standalone copy.
Status: **agreed contracts — implemented** in `supervisor.py`, tested in `tests/`. This document is the **authoritative
implementation contract** for `supervisor.py` and its tests. `PROMPT.md` names *what* to build;
`CLAUDE.md` sets the build conventions; this file records *how*, and why — and wins over the
`PROMPT.md` stub wherever they differ. If code and this document disagree, fix one of them —
never leave both.

Sources it was reconciled against: `PROMPT.md`, `CLAUDE.md`, `README.md`, `validate-lab/SKILL.md`
(the validator is that rubric alone — there is no executable checker), `server/` (read-only),
and the reference shapes in `../code/03-human-in-the-loop/graph.py` and
`../code/04-parallel-supervisor/graph.py`. Library behaviour below was checked against the
installed versions (langgraph 1.2.17, langchain 1.4.7, langchain-aws 1.9.0,
langchain-mcp-adapters 0.3.2, mcp 1.30.0) and the source of langgraph-checkpoint-sqlite 3.1.2 /
aiosqlite 0.22.1.

---

## 1. Decision log

| # | Decision | Why | Cost / risk |
|---|---|---|---|
| 1 | After a human decision, a separate `respond` node writes the answer: one model call, **no tools bound**, plain-text prompt. | A rejected (or completed) write can never be proposed again. | Differs in shape from the reference (`reject → model`). The rubric only requires that every pending call is answered and that the model still answers; it names no route. |
| 2 | Planner retries once on invalid output; a second failure ends the run with a controlled error. | An invalid plan must never enable a write path. | One extra model call in the failure case. |
| 3 | A valid empty plan goes straight to `act`, no specialists. | Trivial requests don't pay for a team. | — |
| 4 | `knowledge` gets `search_knowledge_base`, `list_policies`, `get_policy`. | Policies are files, not part of the KB vector index. | Overlaps `eligibility` on the policy tools. |
| 5 | CLI checkpoints to a fixed `checkpoints.sqlite` beside `supervisor.py`. | Approval pauses survive a process restart. | — |
| 6 (D1) | `build_graph(tools)` with no checkpointer defaults to **`InMemorySaver`**; the CLI always passes an explicit SQLite saver. | **Verified:** an `AsyncSqliteSaver` that the caller never closes hangs interpreter shutdown (aiosqlite 0.22.1 runs a non-daemon worker thread). Reproduced: the graph ran and `asyncio.run` returned, then the process never exited (killed by a 15–20 s timeout); the same script with `await conn.close()` exited in under 1 s. | Programmatic callers that want persistence must pass a checkpointer they close themselves. |
| 7 (D2) | A thread that stopped with pending nodes but **no pending interrupt** is never resumed automatically. | That state can mean the server committed a write whose checkpoint was never saved. | Manual verification required; documented in §9. |
| 8 | Specialist failures are recorded as findings with a status; any non-`ok` finding disables the write path. | Never present failed evidence as success; never write on incomplete evidence. | Read-only answers may be less complete. |
| 9 | Duplicate-write protection lives in graph state (`writes_done`), scoped to one thread. | The server has no idempotency key. | At-least-once, not exactly-once (§8). |
| 10 | Hand-written argument parsing; help checked first; an unknown `-…` option is a usage error, `--` passes a request that starts with `-`. | Flags were being sent to the model as requests; `argparse` would also change `--resume` parsing and reject requests that start with `-`. | A request that starts with `-` needs `--`. |
| 11 | `--resume` and `--status` classify threads through an inspection graph (placeholder tools and model that raise if called). | Status must not touch Bedrock or MCP, and `next`/`interrupts` are only defined by a compiled graph. | It must keep the real graph's shape — so it is built by the same `build_graph`. |
| 13 | A read-only `verify` node between `act` and `review` checks every proposed filing: the software is named in the employee's request, and the server's subscription lookup has exactly one subscription whose name equals the requested software (case and surrounding spaces ignored) **and** `over_limit` true. Otherwise nothing reaches the approver and nothing is filed. After filing, a server result naming a different system is reported as a warning. | Observed live: `software: "AI"` was filed as **MailGuard** (the server matches names by substring and takes the first alphabetically), and the model filed although seats were available — the seat rule lived only in the prompt. | One read call per distinct proposed name; one more node; the filing rule is now code, not model judgment. A filing can still land on a different system that has no subscription (§8). |
| 14 | Specialists never read a substring match as the requested software: the records prompt reports seat status only for an exact `system_name` row, the eligibility prompt names the software as written, and an `ExactNameLookups` tool-call middleware prefixes every `check_software_subscription` result that has no exact-name row with `NO EXACT MATCH for '<name>'; these rows are substring matches only (unconfirmed candidates, not '<name>'): <rows>.` | Observed live: "Mail" was reported as "access to Mail via the MailGuard subscription" and once as "matches MailGuard exactly"; prompts alone did not hold in 5 live runs. | The final reply (from `act` when nothing is proposed) can still suggest a candidate — §13 risk 7. |
| 15 | When a name has no exact system, the employee's answer is a fixed text written by code (§4 "Answer rules"), never the model's: no candidate row is named and nothing is offered. | Live ("for emploees E013, add AI license x 2", and "AI"/"Mail" runs): replies named MailGuard as "the only partial match" and offered to file for it, in 3 of 3 and 4 of 5 runs, although every finding marked it unconfirmed. | The reply is a template; candidates are visible to the operator only (trace, approval warning). |
| 16 | User-visible replies (`act`, `respond`) are generated with an explicit `max_tokens=ANSWER_MAX_TOKENS` (800); a reply whose stop reason is `max_tokens` is unusable — discarded before its tool calls are read, retried once, then replaced by a fixed text with the outcome from state. | Live thread `case-20261026143937-b419`: one plain-text block, `stopReason: max_tokens`, 1000 output tokens (Bedrock's implicit default): a complete answer, then run-on reasoning ("Okay, I need to figure out…"), cut mid-sentence. Not reproduced in 3 further runs (all `end_turn`, 86–152 tokens). | One extra model call when it happens; the longest normal live answer seen was about 350 tokens. |
| 17 | Business-rule fixes from live runs: `verify` also requires the proposed `employee_id` to be an exact id returned by `get_employee` (the server files only for an exact, case-sensitive id); only unmatched lookups for names the employee actually wrote drive a fixed answer; a known flag after a request, or behind leading spaces, is a usage error; every reply prompt forbids claiming access was approved/granted/provisioned, naming internal checks or tools, and asking for internal documents; the approval prompt marks the justification as model-written. | Live: `employee_id: SAAS` was approved, then refused by the server; a lookup of "Confluence" (never asked for) replaced the reply to "what SW can I get?"; `"Mail needed" --resume <id>` became request text; replies said "access will be automatically granted", asked for an "eligibility report", named `create_access_request`, and one justification contradicted the request ("not for personal or family use" for a manager's daughter). | One more read-only lookup per distinct employee; the prompt rules are not guaranteed (§13 risk 11). |
| 18 | A bulk access request — more than one employee (a list, a range, a team, "all employees") or software not named one by one ("all software") — gets a fixed reply from `plan`: the planner reports `bulk` in its structured output, and `may_file_request` + `bulk` ends the run with `BULK_ANSWER`, no specialists, no model text. One employee with several named systems is not bulk. | Live: "E013 … all sw" became a Salesforce proposal blocked as `not_named`; "all employees … webex" got an unverified seat claim and the backwards rule "confirm there are available seats"; "E001 to E012 … webex" exhausted the records budget and was told to "try again later". Owner chose a fixed reply over bulk filing. | Bulk detection is the model's judgement (structured output, not text matching); a missed bulk request falls back to the existing checks. |
| 19 | `verify` also requires the employee's request to identify the proposed employee: its `employee_id` or `email` appears in the request as a whole word (case-insensitive), or its `full_name` does and a `get_employee(full_name)` lookup returns exactly one row with that full name. Otherwise `NOT FILED` with a fixed answer asking for the employee ID. | Live: "For emploee maya from novaops.com …" — the seed data has two Mayas (E001 Cohen, E020 Rosen; the server documents duplicate first names as intentional), the model chose E001, the approver approved and AR020 was filed. In a replay the model proposed E010 for the same request. | A request with only a first name or a nickname cannot be filed; the employee must give the ID, the email or the full name. |
| 20 | A requester the request does not identify as an employee is told nothing about systems or employees: with an employee fact (`no_employee`, `unidentified`, or a lookup's unknown or ambiguous employee) the fixed answer states only the employee facts and asks for the employee ID; `verify` stops a proposal's checks at an employee failure, so no seat details reach the `NOT FILED` reason or the `Outcome:` line, and the `unidentified` reason omits the guessed id; an employee lookup for a name the requester wrote that matches several people, none of whom the request identifies, is an `unidentified` fact even when nothing was proposed. | Owner: "Not employee — no need to share the inside info." Live: "For emploee COHEN …" got Salesforce's seat counts and an IT-ticket hint, and the outcome named E017; a "maya" reply listed "Maya Cohen and Maya Rosen". | Read-only answers for an unidentified requester are still the model's text unless a lookup was ambiguous or found no one. |
| 12 | Interactive mode is a flag (`-i`); no argument stays the Webex demo. Pausing an approval ends the session; a failed request returns to the prompt. | `PROMPT.md` and the validator run `python supervisor.py` as the Webex case. | — |

---

## 2. Graph

```
START → plan ─┬─ error ──────────────────────────────────────────────► END
              ├─ k ≥ 1: Send("worker", …) × k ─► worker ─(barrier)─► act
              └─ k = 0 ─────────────────────────────────────────────► act
act ─┬─ no tool_calls ─► END
     └─ tool_calls ────► verify ─┬─ any check fails (NOT FILED results) ─► respond ─► END
                                 └─ all pass ─► review ─┬─ approve ─► write ─► respond ─► END
                                                        └─ reject ────────────► respond ─► END
```

| Case | Path |
|---|---|
| Normal fan-out | `plan` → one `Send` per chosen worker (one parallel superstep) → `act` waits for all |
| Empty plan | `plan` → `act` |
| Invalid plan (2 failed attempts) | `plan` → END with `error` and `answer` set |
| Read-only request, or write path disabled | `act` (no tools bound) → END |
| `act` decides not to file | `act` (write tool bound, no call) → END |
| Proposal fails the pre-approval check | `act` → `verify` (a `NOT FILED` result per call) → `respond` → END — no interrupt |
| Approved write | `act` → `verify` → `review` (interrupt) → resume approve → `write` → `respond` → END |
| Rejected write | `act` → `verify` → `review` (interrupt) → resume reject (denials added) → `respond` → END |

Routing: `route_after_plan` returns `END`, a `list[Send]`, or `"act"` (path map
`["worker", "act", END]`); `add_edge("worker", "act")` is the fan-in barrier;
`route_after_act` returns `"verify"` if the last `AIMessage` has tool calls, else `END`;
`verify` and `review` return a `Command`.

---

## 3. State

```python
class Finding(TypedDict):
    agent: str
    status: Literal["ok", "incomplete", "failed"]
    summary: str
    seconds: float
    tool_calls: int
    unmatched: list[str]    # names whose lookup this worker saw marked NO EXACT MATCH (cleaned, §4)

class WorkerInput(TypedDict):          # the Send payload — workers never see supervisor state
    name: str
    request: str

class SupervisorState(TypedDict, total=False):
    request: str                                          # replace
    plan: list[str]                                       # replace
    may_file_request: bool                                # replace
    findings: Annotated[list[Finding], operator.add]      # reducer: parallel branches append
    messages: Annotated[list[AnyMessage], add_messages]   # act / review / write / respond only
    checks: dict[str, dict]                               # replace; tool_call_id → resolved subscription, by `verify`
    name_problems: list[dict]                             # replace; by `verify`: {"kind": "no_exact"|"not_named", "name": str}
    writes_done: dict[str, str]                           # replace; written only by `write`
    error: str                                            # replace; planner failure
    answer: str                                           # replace; final text
```

`total=False` and `.get(key, default)` in every node, so `{"request": q}` alone is a valid
input. `initial_state(request)` builds the full dict. Worker transcripts stay inside each worker;
only a `Finding` reaches the supervisor.

---

## 4. Node contracts

| Node | Reads | Writes | Side effects |
|---|---|---|---|
| `plan` | `request` | `plan`, `may_file_request` — or `error`, `answer` (planner failure) or `answer` (bulk, D18) | ≤ 2 Bedrock calls, no tools |
| `worker` | `WorkerInput` | `findings: [Finding]` | Bedrock + read-only MCP calls through its own tools; never raises |
| `act` | `request`, `findings`, `may_file_request` | `messages` (+ `answer` if no tool call) | 1 Bedrock call |
| `verify` | last `AIMessage.tool_calls` | `checks` + `Command(goto="review")`, or `Command(goto="respond", update={"messages": NOT FILED results})` | ≤ 1 read-only `check_software_subscription` per distinct proposed software; never a write |
| `review` | last `AIMessage`, `findings`, `checks` | `Command(goto="write")` or `Command(goto="respond", update={"messages": denials})` | **None.** `interrupt()` is its first effectful line; the node re-runs from the top on resume |
| `write` | last `AIMessage.tool_calls`, `writes_done` | one `ToolMessage` per call, `writes_done` | ≤ 1 MCP `create_access_request` per unique key |
| `respond` | `request`, `findings`, outcome `ToolMessage`s | `answer`, `messages` | 1 Bedrock call, no tools, plain-text prompt |

**Write-path rule (in `act`).** Bind `[create_access_request]` only if `may_file_request` is
true **and** there is at least one finding **and** every finding has status `ok`. Otherwise
bind no tools; the prompt tells the model to say what evidence is missing and that nothing was
filed. The `act` prompt mirrors the reference: file only if the employee is eligible and the
subscription is over its seat limit.

**Pre-approval check (in `verify`, D13).** For each proposed `create_access_request` call, `verify`
calls the read-only `check_software_subscription(software)` (one call per distinct normalised
name) and parses its rows (each MCP text block is one JSON row). The call passes only if:
1. `employee_id` and `software` are non-empty strings;
1a. `employee_id` is exactly (case-sensitive, untrimmed) the `employee_id` of a row returned by the
   read-only `get_employee(employee_id)` — the server's `create_access_request` accepts only an
   exact id, so anything else would be approved and then refused (observed live: `SAAS`). A failure
   is recorded in `name_problems` as `no_employee`;
1b. the request identifies that employee (D19): the row's `employee_id` or `email` appears in the
   request as a whole word (case-insensitive), or its `full_name` does and the read-only
   `get_employee(full_name)` returns exactly one row with that full name (case-insensitive). The
   seed data repeats first names on purpose, so a first name alone never identifies anyone. A
   failure is recorded in `name_problems` as `unidentified`;
2. `software` appears in the employee's request as a whole word (case-insensitive) — the model may
   substitute a system the employee never named (observed live: an "AI" request, Webex articles
   from the knowledge search, a Webex proposal). A request that only describes a tool cannot be
   filed. This is checked before the lookup. The result is
   `NOT FILED: the proposed system '<software>' was never named in your request. Please tell us the
   exact name of the software you need.` and `respond` appends that sentence (without the
   `NOT FILED: ` prefix) verbatim to the model's reply, so the employee always reads which system
   was proposed, that they never named it, and that they should name the software;
3. exactly one row's `system_name` equals `software` after trimming and lower-casing — a partial
   match (`AI` → MailGuard, `Zoom` → ZoomInfo) or no row fails;
4. that row's `over_limit` is `true` — when seats are available, no access request is needed; the
   failure is recorded in `name_problems` as `seats_available` with the system and its seat numbers.

An employee failure (1a or 1b) ends that call's checks — no subscription lookup, and its reason
names no system (D20). Otherwise all of a call's software failures are reported together. If every call passes, the matched rows are stored in `checks` and the run goes to `review`. If any
call fails, **no** call is sent for approval: each gets
`ToolMessage("NOT FILED: <reason>", status="error")` (a call that passed gets
`NOT FILED: another proposed request in the same turn failed the pre-approval check`) and the run
goes to `respond`. An MCP error result from the lookup fails the check; a transport exception
propagates (the run stops before any write, so recovery reports "no write attempted"). Non-write
tool calls are left to `write`, which refuses them as before. The arguments are never changed.
A failure under point 2 is recorded in `name_problems` as `not_named`, under point 3 as
`no_exact` (with the proposed name), so `respond` can decide its answer from state.

**Answer rules (D15, D16).** Decided in code from tool results and state, never from model prose.
- *Names.* `clean_name(raw)`: newlines → spaces, markdown characters (`` ` * _ # > [ ] ( ) ~ | ``)
  removed, whitespace collapsed, clipped to 60 characters. Applied to every name a fixed text shows.
- *Unmatched names* come from the `ExactNameLookups` marker in each worker's own tool results
  (`Finding.unmatched`), de-duplicated case-insensitively, and count **only if the name appears in
  the employee's request** as a whole word — a specialist may look up a name nobody asked for
  (observed live: "Confluence" for "what SW can I get?").
- *Ambiguous employees* (D20): an employee lookup in a worker's own tool results whose query the
  requester wrote, that returned several rows, none of which the request identifies (id, email,
  or a full name only one of those rows has) → `Finding.ambiguous_employees` → an `unidentified`
  fact (`Your request does not identify the employee exactly.`).
- *Employee facts come alone* (D20): when any employee fact is present, every other fact is
  dropped, so the answer is the employee facts, `Nothing was filed.` and `Please tell us the
  employee ID.` — nothing about systems or seats.
- *Unknown employees* come the same way from the `NO EMPLOYEE FOUND` marker
  (`Finding.unknown_employees`), under the same "written in the request" rule, and give a
  `no_employee` fact even when nothing was proposed — observed live: "For new emploee (E099) …"
  stopped at the records tool limit, so `verify` never ran, and the reply called E099 "eligible".
- *Trigger (i):* `act`'s usable reply proposes no write and some finding has unmatched names or unknown employees →
  `answer` is the fixed text. *Trigger (ii):* `verify` blocked the turn and `name_problems` is
  non-empty → `respond` makes no model call and `answer` is the fixed text.
- *Fixed text:* one fact per name, each stated once — `No system named exactly '<N>' was found.`
  for an unmatched or `no_exact` name, `The proposed system '<P>' was never named in your request.`
  for a `not_named` proposal, `No employee with the ID '<E>' was found.` for a `no_employee` proposal,
  `Your request does not identify the employee exactly.` for an `unidentified` proposal (an
  employee fact: it is followed by `Please tell us the employee ID.`) —
  and, for `seats_available`, `<System> has seats available (<a> of <l> in use), so no access request was
  filed — one is only needed when a subscription is over its seat limit. Nothing has been assigned to
  you yet: open an IT ticket to have a seat assigned (your manager approves it).` (grounded in the
  dataset's license-assignment articles; observed live: the model promised "you will be provisioned") —
  then, unless every fact is `seats_available`, once `Nothing was filed.`, then `Please tell us the employee ID.` if an employee fact is present
  and `Please tell us the exact name of the software you need.` if a `no_exact`/`not_named` fact is present. It
  never names a candidate row and never offers to file.
- *Reply rules (prompts, D17).* `act` and `respond` prompts say: never claim access was approved,
  granted, provisioned or assigned; never mention internal checks, specialists or tool names; never
  ask the employee for internal reports or documents. `act` also files only for the employee named,
  for their own work, with the reason they gave, never inventing a justification. `respond` gives,
  after a rejection, only the reason the approver stated. When filing is disabled the employee is
  told an internal check could not finish and to try again later. (A rule about quantities —
  "x 2" — was tried and removed: live, it made the model refuse over-limit filings.)
- *Mixed case* (a proposal passed `verify`, but a finding has unmatched names): the normal flow
  runs; after `respond`'s model reply the fixed facts for the unmatched names are appended, then
  `Please tell us the exact name of the software you need.` once. The approval prompt shows a
  warning per unmatched name (§7).
- *Output limit and truncation:* `act` and `respond` call the model with
  `max_tokens=ANSWER_MAX_TOKENS` (800). A reply whose `response_metadata` stop reason
  (`stopReason`/`stop_reason`/`finish_reason`) is `max_tokens` or `length` is unusable: it is
  discarded before its `tool_calls` are read (a cut-off tool call is malformed), a trace line is
  printed, and the call is retried once with the same input. If the retry is also cut off, `act`
  answers `Sorry — I could not produce a complete answer. Outcome: <outcome from state>.` (or the
  fixed text above when trigger (i) applies) and proposes nothing; `respond` answers `Sorry — I could
  not produce a complete reply. Outcome: <outcome from state>.` Truncation is never detected from
  the text. The planner's error answer is already fixed text.

---

## 5. Planner contract

```python
class Plan(BaseModel):
    workers: list[Literal[*WORKER_SPECS]]   # closed set built from the spec dict
    may_file_request: bool                 # the employee lacks access to a system and wants it
    bulk: bool = False                     # more than one employee, or software not named one by one
```

- Call: `model.with_structured_output(Plan, include_raw=True)` — returns
  `{raw, parsed, parsing_error}`. `parsed is None` (no tool call, unknown name, bad shape) is
  a failure.
- Retry once with the same messages. Second failure → `error="planner_failed"`, a short
  `answer` apologising that the request could not be routed, END. No workers, no tools, no write.
- Bedrock API errors are raised before parsing, are not retried, and end the run (§11).
- Duplicate names are removed, order kept. An empty list is valid.
- *Bulk (D18).* `may_file_request` and `bulk` both true → `plan` returns `plan=[]`,
  `may_file_request=False` and `answer=BULK_ANSWER`, and the run ends: no specialist, no `act`, no
  lookups, no model text, exit 0, outcome `no access request was proposed`. `BULK_ANSWER` is
  `Each access request covers one employee and software they name one by one. Nothing was filed.
  Please send a separate request for each employee, naming each software you need (for example:
  "<employee ID> needs <software>").` A bulk question with no filing (`may_file_request` false) is
  answered normally.
- Prompt: one line per specialist (knowledge — how-to and policy text; records — the employee's
  record and seat status; eligibility — whether they may use it and who approves), "only the
  relevant ones", an empty list allowed for trivial requests.

---

## 6. Specialists and tool access

| Specialist | Tools | Prompt purpose |
|---|---|---|
| knowledge | `search_knowledge_base`, `list_policies`, `get_policy` | how-to and policy facts |
| records | `get_employee`, `check_software_subscription` | employee record and seat status |
| eligibility | `get_employee`, `list_policies`, `get_policy` | may they use it; who approves (single most relevant policy) |

- Every prompt ends: *"At most 3 tool calls. Facts only, 2–3 sentences. Do not decide or file
  anything."*
- Each worker is `create_agent(model, its_tools, system_prompt=…,
  middleware=[ToolCallLimitMiddleware(run_limit=3, exit_behavior="continue")])`, invoked with
  `config={"recursion_limit": 12}` as a backstop. Measured: 12 steps fit 3 sequential tool
  calls plus the report (11 steps); a 4th attempted call exhausts it. Either way that worker
  is `incomplete` (§11).
- Discovery: `MultiServerMCPClient({"novaops": {"url": os.environ["MCP_SERVER_URL"],
  "transport": "streamable_http"}}).get_tools()`; discovered names are printed. The build never
  imports `server/`. The current `server/` (including the owner's local change in
  `server/server.py`) is a fixed dependency: the build never modifies, reverts or reimplements it.
- **Exact names (D14).** The server's lookup matches by substring (`AI`, `ai`, `Mail` → MailGuard;
  `Zoom` → ZoomInfo; `up` → CloudBackup, Laptop Fleet via its vendor, SupportDesk — probed live).
  The records prompt reports seat/subscription status only for a row whose `system_name` equals the
  requested software (ignoring case); otherwise it says "no exact match for <name>" and lists the
  rows only as unconfirmed candidates. The eligibility prompt names the software exactly as the
  employee wrote it and never substitutes a different system. Every specialist runs with the
  `ExactNameLookups` middleware: a `check_software_subscription` result with no exact-name row is
  prefixed `NO EXACT MATCH for '<name>'; these rows are substring matches only (unconfirmed
  candidates, not '<name>'): <system names or none>.` before the specialist's model reads it, and
  a `get_employee` result with no rows is prefixed `NO EMPLOYEE FOUND for '<query>'.` (read into
  `Finding.unknown_employees`). It only compares names and counts rows; it does not change the rows
  or reimplement the server.
- `WRITE_TOOL = "create_access_request"` is the one tool name with special meaning. The server
  marks no tool read-only (no `readOnlyHint`), so the write tool cannot be discovered — it must
  be declared client-side.
- `build_graph` raises `ValueError` if a spec names an undiscovered tool, if `WRITE_TOOL` was not
  discovered, or if any spec contains `WRITE_TOOL`.
- A worker's ToolNode holds only its own tools: a hallucinated write call gets an error
  `ToolMessage`, and nothing reaches the server. `act` holds only the write tool, and its calls
  go through `review` → `write`, never through a ToolNode.

---

## 7. Approval contract

Interrupt payload (raised by `review`):

```python
{"kind": "approval_request",
 "question": "Approve filing this access request?",
 "requests": [{"tool_call_id": str, "tool": str, "args": dict,      # args verbatim from the checkpointed AIMessage
               "resolved": {"system_name", "seat_limit", "active_seats", "over_limit"}}],  # from `verify`
 "evidence": [{"agent": str, "status": str, "summary": str}],
 "request": str,                                                  # the employee's original request text
 "unmatched": [str]}                                              # cleaned names with no exact system (mixed case)
```

- Resume value: `{"approved": True}` or `{"approved": False, "reason": str}`. Only
  `decision.get("approved") is True` approves; anything else is a rejection (fail closed).
- **Approve** → `Command(goto="write")`.
- **Reject** → for **every** call in the last `AIMessage`, a
  `ToolMessage(content=f"DENIED by approver: {reason}. The access request was not filed.",
  tool_call_id=call["id"], name=call["name"], status="error")`, then `Command(goto="respond")`.
  `write` never runs.
- On resume, LangGraph re-runs only `review`; `plan`, `worker` and `act` are already in the
  checkpoint and are not re-executed.

---

## 8. Write safety and duplicates

- **System check after filing.** If the server's result names a `system_name` that differs from
  the approved `software` (trimmed, lower-cased), the result is kept in `writes_done` but
  prefixed `WARNING: filed for '<system>', not the approved '<software>' — ask IT to correct it.`
  The outcome line and the reply show it. Nothing can undo it (no cancel tool).
- **Exact arguments.** `write` sends each checkpointed call unchanged:
  `tool.ainvoke({"type": "tool_call", "id": c["id"], "name": c["name"], "args": c["args"]})`.
  The model is never asked to regenerate them.
- **Duplicate key:** `employee_id.strip().upper() + "|" + software.strip().lower()`.
  - First call for a key → server call; if the result `status != "error"`, its content is stored
    in `writes_done[key]`. Server error results are not stored — no row was written.
  - Each later call with that key → `ToolMessage("DUPLICATE: already filed in this run — <first
    result>", tool_call_id=<its own id>)`, no server call.
  - A call missing `employee_id` or `software` → error `ToolMessage`, no server call.
  - A call naming any tool other than `WRITE_TOOL` → error `ToolMessage`, not executed.
- **Limits — the server cannot provide more than this:**
  - The record is per thread; the same request run twice is two threads and two access requests.
  - The server has no idempotency key. A crash after the server commits but before `write`'s
    checkpoint is saved leaves the outcome unknown. Delivery is **at-least-once, never
    exactly-once** — which is why §9 refuses to auto-resume such a thread.
  - Different spellings the server resolves to one system (substring match) produce different keys.
  - The server resolves names by substring (`LIKE '%name%'`, first alphabetically). `verify`
    proves the name is an exact subscription name, but filing searches **all** systems: a system
    without a subscription whose name contains the same text and sorts earlier would still be
    chosen. No tool lists all systems, so this is only detected after filing (warning above).
  - No tool reads `access_requests`: the seed's AR001 (E010 / Webex, `blocked`) is invisible, so
    an approval files a new row (AR005).
  - The server's database is in memory; filed rows vanish when the server restarts.
  - The approve/reject decision is not recorded server-side (no approvals tool; `approver_id`
    stays NULL).

---

## 9. Persistence, recovery and CLI

**Checkpointers (D1).**
- `build_graph(tools, model=None, checkpointer=None)`: an explicitly supplied checkpointer is
  always used as given. `InMemorySaver()` is used **only** when `checkpointer` is `None` —
  interrupt/resume then works within the process, and the process exits cleanly. Callers wanting
  persistence pass a saver they own and close. `AsyncSqliteSaver.__init__` calls
  `asyncio.get_running_loop()`, so a caller-built SQLite saver must be created inside a running
  event loop.
- CLI: every invocation (new request or `--resume`) opens its own saver with
  `async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:` and builds a fresh graph
  with `build_graph(tools, model=model, checkpointer=saver)`. The `async with` closes the
  connection on every exit path — completion, pause, error, Ctrl-C. Default
  `db_path = CHECKPOINT_DB = Path(__file__).resolve().parent / "checkpoints.sqlite"`. The saver
  uses WAL mode, so `-wal`/`-shm` files appear beside it (all gitignored).

**CLI entry point.** `async def cli(argv, *, tools=None, model=None, db_path=CHECKPOINT_DB,
ask=ask_human) -> int` returns the exit code; `main()` is `sys.exit(asyncio.run(cli(sys.argv[1:])))`.
`tools=None` → `load_tools()`; `model=None` → `get_model()`; `ask(payload) -> decision` is the
approval prompt and raises `EOFError`/`KeyboardInterrupt` to pause. Every prompt (`ask` and the
interactive request prompt) runs in a daemon thread, so a Ctrl-C (which `asyncio.run` delivers
as task cancellation) is handled like end-of-input, and a prompt still blocked on `input()`
cannot hold the process open. These keyword arguments are the only injection points the tests
use (plus a scripted `builtins.input` for the real prompts); tests observe exit codes, printed
output, the fake server's call log and the stored answer — not node internals.

**Thread IDs.** `case-<UTC yyyymmddHHMMSS>-<4 hex>`, printed when a request starts and at every
pause. Every new request — one-shot, demo, or each request of an interactive session — gets a new
thread; a thread id is reused only by `--resume`/`--status`. Reusing one for a new request would
append `findings` twice.

**Argument parsing (D10).** Hand-written and checked in this order, before any configuration,
model, MCP connection or checkpoint file is touched:

| argv | Mode |
|---|---|
| first argument `-h` / `--help` | print usage (every mode, the approval answers, the `--` rule), exit 0 |
| *(none)* or `--demo` alone | **demo**: the Webex request (`DEFAULT_REQUEST`), once |
| `-i` / `--interactive` alone | **interactive** session |
| `--resume THREAD_ID` | **resume** that thread |
| `--status` / `--status THREAD_ID` | **status** of all threads / of one thread |
| `--` followed by words | **one-shot** request made of the words (lets a request start with `-`) |
| any other first argument starting with `-` (leading spaces ignored) | unknown option: usage, exit 2 — a mistyped flag is never sent to the model |
| a request followed by a separate known flag (`"text" --resume id`) | usage, exit 2 — the flag is never folded into the request text |
| anything else | **one-shot**: the arguments joined by spaces are the request |

A known flag given with the wrong number of arguments (`--demo x`, `--resume`, `--status a b`)
→ usage, exit 2. A request that merely *contains* `--help` (`"what does --help do"`) is one
argument that does not start with `-`, so it stays a request.

**Approval prompt.** The CLI prints the question, the thread id, the employee's original request
(`Employee's request: …` — from the payload, so `--resume` prints it too), one
`Warning: no system named exactly '<N>' exists — evidence about it is unconfirmed.` line per
unmatched name, each specialist's evidence, the server system `verify` matched, and
every proposed call as the tool name plus one `key: value` line per argument — exactly the
checkpointed arguments, which the human cannot edit; `business_justification` is labelled
`(written by the model — check it against the employee's request)`. `ask_human` reads one line, trimmed and
lower-cased:
- `approve`, `yes`, `y`, `ok` → `{"approved": True}`;
- `reject`, `no`, `n` → asks for an optional reason → `{"approved": False, "reason": …}`
  (empty reason → `"rejected by approver"`);
- anything else, including an empty line → prints the accepted answers and asks again. It never
  becomes a decision by default;
- end-of-input / Ctrl-C → pause (below).

The CLI resumes with `ainvoke(Command(resume=decision), config)` and keeps driving while the
result carries `__interrupt__`, so each interrupt is shown and answered; it reports only the
final state. The `review` node is unchanged: only `decision.get("approved") is True` approves.

**Outcome line.** After the answer the CLI prints one `Outcome:` line derived from the final
state, never from the model's prose: `access request filed — <server result>` (from
`writes_done`), `rejected by the approver — nothing was filed` (denial `ToolMessage`s),
`the server refused the write — nothing was filed` (error result), `no access request was
proposed`, or `the request could not be routed — nothing was looked up or filed` (planner failure).

**Modes.**
- **demo / one-shot** — load configuration, MCP tools and the model, open the saver, run one new
  thread to a terminal state, print answer + outcome, exit. Exit 1 on a runtime error or planner
  failure.
- **interactive** — load tools and model and open the saver **once**, then loop on the request
  prompt `Request (exit/quit to leave):`. Each non-empty line other than `exit`/`quit` (any case)
  is run as a new thread, exactly as one-shot. An empty line re-prompts. Ends with exit 0 on
  `exit`, `quit`, end-of-input or Ctrl-C at the request prompt.
  - A runtime error in one request is printed with its checkpoint classification (case 4
    wording) and the thread id; the session returns to the prompt and the next request gets a new
    thread. The failed thread is never retried — an uncertain write included.
  - End-of-input / Ctrl-C at an approval prompt pauses that thread and **ends the session**
    (exit 0), so no new request starts while that approval is unresolved.
- **resume** — needs the checkpoint file; a missing file or unknown thread → "unknown thread",
  exit 2, and the file is not created. The thread is classified first with the inspection graph
  (below); only a thread with a pending interrupt loads MCP tools and the model, shows the stored
  payload, asks, and resumes. Completed `plan`/`worker`/`act` work is never re-executed — only
  `review` re-runs. The four cases are listed below.
- **status** — read-only: never loads `.env`, MCP tools or the model, and writes no checkpoint
  (opening the saver only runs its idempotent table setup — verified: row counts unchanged). A missing
  checkpoint file → `No saved threads.` (exit 0) and the file is not created. Threads are listed
  with the supported `AsyncSqliteSaver.alist(None)` (root namespace only) and each is read with
  `aget_state` on the inspection graph. Per thread: id, request, state, and the next action.
  `--status THREAD_ID` shows one thread (unknown → exit 2) including the proposed arguments of a
  pending approval. States and the next action:
  - **awaiting approval** → `python supervisor.py --resume <id>`;
  - **finished** (filed / rejected / no request proposed / could not be routed) → nothing to do;
    a filed request is reported as filed and no new write is suggested;
  - **stopped — outcome uncertain** → verify manually with IT; neither `--resume` nor re-running
    the request is suggested;
  - **stopped — already filed** → do not file it again;
  - **stopped — nothing filed** → a new request is safe.

  Limitation: the checkpoint records only what the client saw. An uncertain write cannot be
  resolved from it, and nothing on the server can be checked (no read tool, §8).

**Inspection graph (D11).** `resume` and `status` read checkpoints through `aget_state` on the
same compiled graph shape, built by `build_graph` with placeholder tools and a placeholder model
that raise if ever invoked. `StateSnapshot.next`/`.interrupts` are derived by LangGraph from the
graph, so reading them through a compiled graph is the supported mechanism; the existence of a
SQLite file or row proves nothing.

**`--resume` cases** (from `aget_state`):
  1. no stored values → "unknown thread", exit 2;
  2. a pending interrupt → show `snapshot.interrupts[0].value`, ask, resume;
  3. finished (`snapshot.next` empty) → print the stored `answer` and outcome, exit 0, nothing
     executes;
  4. **pending nodes but no pending interrupt (D2)** → never resumed, nothing executes, exit 3.
     The message depends on what the checkpoint proves:
     - **uncertain write** — `"write"` is among the pending nodes: the approved call may or may
       not have reached the server (it could have committed before the checkpoint was saved). Say
       so, list the approved arguments, state that no tool can read `access_requests`, and require
       manual verification before any new attempt. No write call is issued.
     - **write completed** — `"write"` is not pending and `writes_done` is non-empty (e.g. a crash
       in `respond`): the request *was* filed. Print the stored result and tell the operator not to
       file it again.
     - **no write attempted** — `"write"` is not pending and `writes_done` is empty (e.g. a crash
       or Ctrl-C in `plan`, `worker`, `act`, or `respond` after a rejection): nothing was filed,
       and starting a new request is safe.
- An exception raised inside `write` during a live run prints the uncertain-write warning and
  exits 1; a later `--resume` of that thread lands in the uncertain-write case above.
- **Ctrl-C while the graph is running** (not at a prompt) → the run stops; the CLI prints
  `Interrupted.`, the thread id and its case-4 classification, closes the saver and exits 130
  (an interactive session ends too). Never a raw traceback, never a success message.
- Rationale: the server has no idempotency key and no read tool, so an interrupted write is
  indistinguishable from a committed one (at-least-once, §8). Retrying could file a duplicate; a
  human must decide.

**Exit codes.** 0 done, paused, help, status, or interactive session ended · 1 runtime error
(planner failure, Bedrock, MCP) in a one-shot / demo / resume run · 2 configuration or usage
(missing variable, unknown option, unknown thread) · 3 not resumed automatically — the thread
stopped outside an approval pause (case 4) · 130 interrupted by Ctrl-C while running.

---

## 10. Model and configuration

- `load_config()` = `load_dotenv(find_dotenv())`, called on first need by `get_model()`,
  `load_tools()` and the CLI modes that use the model or MCP — never at import and never for
  `--help` or `--status`. Prints nothing; never overrides a variable already set.
- Required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`,
  `BEDROCK_EMBEDDING_MODEL_ID`, `MCP_SERVER_URL` — the same list as `CLAUDE.md`. The tracked
  `.env.example` (placeholders only) lists `MCP_SERVER_URL=http://127.0.0.1:9876/mcp`; the local
  `.env` gets the same line at implementation time.
- `require_env()` raises `RuntimeError` naming the **missing variables only** — never values.
  Called by `get_model()` and by the CLI (which turns it into exit 2).
- `get_model()` → `ChatBedrockConverse(model=os.environ["BEDROCK_MODEL_ID"],
  region_name=os.environ["AWS_REGION"])`. Credentials reach boto3 from the environment; they are
  never passed or printed. One model instance serves every node.
- Importing `supervisor` starts no CLI, opens no connection, reads no `.env`, and needs no
  environment.

---

## 11. Error policy

| Failure | Handling |
|---|---|
| Worker exceeds its tool limit, hits `recursion_limit`, or ends without text | Finding `incomplete` |
| Any other exception inside a worker (Bedrock, MCP transport) | Finding `failed`, summary `"<Type>: <first 200 chars>"` |
| MCP server error result (e.g. unknown policy name) | Not an exception: the adapter returns an error `ToolMessage` to the worker, which may recover |
| Any non-`ok` finding | Kept and shown with its status; the write path is disabled |
| Missing configuration | Exit 2 |
| MCP unreachable, discovery failure, spec validation failure | Exit 1 |
| Planner invalid twice | Controlled END, exit 1 |
| Bulk filing request (D18) | `plan` ends the run with the fixed `BULK_ANSWER`, exit 0; nothing looked up or filed |
| Bedrock error in `plan` / `act` / `respond` | Exit 1; a later `--resume` exits 3 with "no write attempted" or "write completed" (§9 case 4) |
| Transport error in `write` | Exit 1 with the uncertain-write warning; a later `--resume` exits 3 and issues no write (§9 case 4) |
| Any runtime error in an interactive request | Printed with its case-4 classification; the session returns to the prompt; the failed thread is never retried |
| Ctrl-C while the graph runs | `Interrupted.` + classification, saver closed, exit 130 |
| Unknown option / wrong flag arity | Usage, exit 2, nothing loaded |
| Proposed software is not an exact subscription name, or has seats available | `verify`: `NOT FILED` results, no approval prompt, reply explains |
| Server files a different system than approved | Warning in the result, reply and outcome; not undone |
| A name has no exact system (lookup marker, or `verify` name failure) | Fixed answer (§4 Answer rules); the model's text is not used |
| A user-visible reply stops at `max_tokens` | Discarded unread, retried once, then a fixed text with the outcome |
| Proposed employee id not found exactly | `verify`: `NOT FILED`, fixed answer asking for the employee ID; no approval prompt |
| Request does not identify the proposed employee (first name only, a full name two people share) | `verify`: `NOT FILED`, fixed answer asking for the employee ID; no approval prompt; no system details in the reply or the outcome line |
| A specialist's lookup of a name the requester wrote matches several employees, none identified | Fixed answer asking for the employee ID, even when nothing was proposed; no employee names are listed |
| A specialist's employee lookup returns no one (id written in the request) | Fixed answer asking for the employee ID, even when filing was disabled; the model's text is not used |
| Proposed software has free seats | `verify`: `NOT FILED`, fixed answer (no request needed, nothing assigned, open an IT ticket); no model text |

---

## 12. Test plan

Offline tests use `unittest.IsolatedAsyncioTestCase` (no pytest). A scripted `BaseChatModel`
picks its reply by a role marker in the system prompt; its `bind_tools` records bound tool
names; its `with_structured_output` returns scripted `{raw, parsed, parsing_error}`. Fake async
tools carry the server's tool names and record every call. Graph-level tests drive
`build_graph(...)` with `InMemorySaver` (the default); every persistence and recovery test (T14,
T15, T16b, T19) drives `cli(...)` against a temporary SQLite file, so each `cli` call opens a
fresh saver and builds a fresh graph — exactly what a process restart does. Assertions use exit
codes, printed output, the fake server's call log and the stored answer, not node internals.
The fakes (`tests/fakes.py`) force `LANGSMITH_TRACING=false` before importing `supervisor`, so a
local `.env` that enables tracing cannot make the offline suite touch the network. Run:
`python -m unittest discover -s tests -v`.

| ID | Verifies | Assertion |
|---|---|---|
| T1 | Closed planner set | `Plan`'s literal equals `WORKER_SPECS` keys; an unknown name fails parsing |
| T2 | Planner retry / failure | invalid → valid proceeds; invalid twice → `error` set, 0 worker / act / write calls, no interrupt |
| T3 | Empty plan | 0 workers run, `act` binds no tools, 0 writes |
| T4 | Spec validation | write tool in a spec, unknown tool, or undiscovered write tool → `ValueError` |
| T4c | Exact-name specialist rules | the records prompt reports status only for an exact `system_name` row and otherwise says "no exact match for" and lists unconfirmed candidates; the eligibility prompt names the software as written and never substitutes another system |
| T4b | Eligibility prompt bounds | the prompt keeps the employee lookup conditional, one `get_policy` for the single relevant policy, no repeated calls, a concise finding |
| T5 | Specialists cannot write | a worker model emits `create_access_request` → server write count 0 |
| T6 | Real parallelism + reducer | each worker's model awaits `asyncio.Barrier(3)` under `asyncio.timeout(2)` — a sequential run cannot pass; all 3 findings `ok`, peak concurrency 3, all merged |
| T7 | Read-only path | no `__interrupt__`, `act` binds no tools, 0 writes, answer present |
| T8 | Gate before write | Webex → `__interrupt__` with E010 / Webex in `args`; 0 writes at that point |
| T9 | Exact approved args | approve → exactly 1 server call, args deep-equal the payload args; `act` model called once in total |
| T10 | Rejection | 2 proposed calls → 2 denials with matching `tool_call_id`s, 0 writes, no second interrupt, `respond` bound no tools, answer present; resume values `{"approved": "yes"}`, `{"approved": 1}`, `{"approved": False}`, `"y"` all reject (`{}` and `None` are not usable LangGraph resume values — they never reach `review`) |
| T11 | Duplicates | 5 identical calls approved → 1 server write; 5 `ToolMessage`s with matching ids, 4 `DUPLICATE`; graph ends cleanly |
| T12 | Failed specialist | one worker raises → `failed`, others `ok`, write tool not bound, 0 writes |
| T13 | Incomplete specialist | 4 calls in one turn → 3 run, 1 blocked → `incomplete`, `tool_calls` 3, write path disabled; a worker that never stops → recursion limit → `incomplete` |
| T14 | Restart persistence (SQLite) | `cli([request])` with `ask` raising `EOFError` → exit 0, thread id printed, 0 writes; then a second `cli(["--resume", id])` on the same `db_path` (fresh saver, fresh graph, same thread id) with `ask` approving → exit 0, exactly 1 write, args equal to those shown at the pause |
| T15 | Completed thread | after T14's flow, `cli(["--resume", id])` again → exit 0, stored answer printed, server call log and model call count unchanged |
| T16a | Default checkpointer (D1) | subprocess: `build_graph(fake_tools, model=fake)` → `app.checkpointer` is an `InMemorySaver`; interrupt + resume works; process exits < 10 s |
| T16b | CLI closes SQLite (D1) | subprocess: `cli(...)` with fakes and a temp `db_path` runs to completion → exits < 10 s (an unclosed aiosqlite saver hangs shutdown, verified); `--resume` on that file finds the thread |
| T16c | Explicit checkpointer honoured | `build_graph(fake_tools, model=fake, checkpointer=given)` → `app.checkpointer is given` |
| T17 | Server error on write | error result not recorded in `writes_done`; run completes |
| T18 | Import is inert | `import supervisor` runs no CLI and makes no network call |
| T19a | Uncertain write (D2) | fake write tool records the call, then raises a transport error → exit 1 with warning; `cli(["--resume", id])` → exit 3, message says outcome uncertain and requires manual verification, **server call log still has exactly 1 write**, `ask` never called |
| T19b | Write completed (D2) | write succeeds, fake model raises in `respond` → exit 1; `--resume` → exit 3, message says the request was filed and shows its result, still exactly 1 write |
| T19c | No write attempted (D2) | fake model raises in `act` → exit 1; `--resume` → exit 3, message says nothing was filed, 0 writes |
| T20 | Unknown thread | `cli(["--resume", <unknown>])` → exit 2 |
| T21 | Approval answers | `ok`/` Yes `/`approve`/`y` approve; `reject`/`no`/`n` reject (with or without a reason); an empty line and `maybe` re-prompt with the accepted answers before a valid answer; end-of-input raises `EOFError` (pause) |
| T22 | Approval end to end (CLI) | one-shot with the real prompt and input `ok` → the prompt shows the thread id, evidence, `create_access_request` and `employee_id: E010`; exactly 1 write whose args equal the payload args; `Outcome: access request filed`. Input `""`, `nope`, `no`, a reason → 0 writes, `Outcome: rejected` |
| T23 | Help and parsing | `--help`/`-h` exit 0 listing every mode, with `load_dotenv`, `get_model`, `load_tools` and the SQLite saver patched to fail; subprocess `python supervisor.py --help` exits 0 with the required variables removed from its environment. `--bogus`, `--demo x`, `--resume`, `--status a b` → exit 2, nothing loaded. `-- -x` is a request |
| T24 | Demo and one-shot | no argument and `--demo` both run `DEFAULT_REQUEST` once; `"what's the remote work policy?"` runs that request once; each prints one new thread id |
| T25 | Interactive session | scripted input: Webex request, `ok`, policy request, `exit` → exit 0, two different thread ids, 1 write, both answers, prompt shown again after each answer; `quit` and end-of-input also end it; an empty line re-prompts; end-of-input at an approval → `Paused`, the next scripted request is never read, then `--resume` approves → 1 write; a model error in the first request → error + "Nothing was filed", the second request runs on a new thread, the failed one is not retried; an uncertain write → warning, back to the prompt, still exactly 1 write |
| T26 | Status | no file → `No saved threads.`, file not created; awaiting approval → the exact `--resume <id>` command and the proposed args; filed → reported filed, no resume or new write suggested; rejected → not filed; uncertain write → manual verification, no `--resume` suggested; unknown id → exit 2; `get_model`/`load_tools` patched to fail throughout |
| T28 | Pre-approval check | a request naming `AI` but a `Webex` proposal → no interrupt, no lookup, 0 writes, the result and the reply say "'Webex' was never named in your request" and ask for the exact name; `webex`/`WEBEX`/`WeBeX` in the request pass; the payload carries the original `request`, printed as `Employee's request:` at the prompt and again on `--resume`; for the probes `AI`, `ai`, `Mail`, `Zoom`, `up` a records specialist reading the lookup sees `NO EXACT MATCH for '<probe>'` with the candidates and `not '<probe>'`, and the run ends with no interrupt and no write; an exact `webex` lookup is not marked; the model proposing `MailGuard` for an `AI` request is blocked as never named; proposing `AI` literally is blocked as not an exact system name; whole-word match (`Webexpress` does not name Webex); `AI` (server would pick MailGuard), `Zoom` (ZoomInfo) and an unknown name, each named in the request → no interrupt, 0 writes, `NOT FILED` naming the reason, `respond` bound no tools, `Outcome: not filed`; exact `MailGuard` with seats available → not filed, seats reason; `webex` (case differs) → interrupt whose request shows `resolved.system_name` Webex and 42/40 seats; Webex + AI proposed together → neither reaches approval; a server that files `Cisco Webex` for an approved `Webex` → filed with `WARNING`; the lookup raising mid-run → exit 1, `--resume` exit 3 "nothing filed" |
| T29 | Truncated replies | scripted stop reason `max_tokens`: `respond` cut off once then fine → the second reply is the answer and a retry trace line is printed; cut off twice → `Sorry — I could not produce a complete reply. Outcome: …` with the outcome from state; an `act` reply cut off twice that contains a tool call → no `verify` lookup, no interrupt, 0 writes, fixed answer; `act` and `respond` calls carry `max_tokens=800` |
| T30 | Fixed answers for unmatched names | scripted models that try to suggest the candidate: trigger (i) for `AI`/`Mail` (MailGuard) and `Zoom` (ZoomInfo) → answer equals the fixed text, no candidate name, 0 writes; trigger (ii) `AI` proposed literally (`no_exact`) and `MailGuard` proposed for an `AI` request (`not_named`) → fixed text, no model call in `respond`; both triggers in one run → each fact once and the request for the exact name once; mixed case (Webex proposed and approved, `AI` unmatched) → model reply + appended fixed facts, approval prompt warns about `AI`; a hostile name (newline, markdown, instruction-like text) is cleaned and clipped to 60 characters |
| T31 | Business rules from live runs | unknown employee `SAAS` → no interrupt, 0 writes, fixed answer asking for the employee ID, no `respond` model call; `" e010 "` (not exact) blocked; unknown employee + non-exact software → only the employee fact and its ask (D20); a records lookup of `Confluence` for "what SW can I get?" does not replace the model's answer; `"Mail needed" --resume <id>`, `"  --resume <id> …"`, `"text" -i`, `"x" --status` → exit 2 with nothing loaded, while `-- please explain --resume` is a request; reply prompts carry the D17 rules; the approval prompt labels the model-written justification; free seats (MailGuard 70/80) → the fixed free-seats answer, no `respond` model call ; a records lookup of `E099` that returns no one, with records over its tool limit → the fixed `No employee with the ID 'E099' was found.` answer, 0 writes; lookups that find someone, or of an id the employee never wrote, leave the model's answer alone |
| T32 | Bulk requests | the three live bulk requests ("E013 … all sw", "all employees … webex", "E001 to E012 … webex") with `bulk=True` → `BULK_ANSWER`, only the planner called, no tool call, no interrupt, outcome `no access request was proposed`; `bulk=True` with `may_file_request=False` → normal answer; `Plan.bulk` defaults to false and its description names "more than one employee" |
| T33 | The request must identify the employee | the live "maya from novaops.com" request with an E001 proposal → no interrupt, 0 writes, `Your request does not identify the employee exactly. Nothing was filed. Please tell us the employee ID.`, no `respond` call; the ID, the email or the unique full name in the request → interrupt; a full name shared by two employees → blocked. The fake server has a second Maya (E020) and emails, like the seed data. D20: "COHEN … webex and MailGuard" → only the employee fact, outcome `not filed — the request does not identify the employee exactly.`, no subscription lookup; an unknown employee plus a non-exact software → only the employee fact; a records lookup of "maya" (two rows) with a model reply listing both → the fixed answer, while "I'm Maya Cohen …" or "I'm E001 (maya) …" keeps the model's answer |
| T27 | Ctrl-C and a fresh process (subprocess, real `asyncio.run`) | SIGINT raised while the approval prompt waits → `Paused`, exit 0, < 10 s; SIGINT during a model call → `Interrupted.`, exit 130, < 10 s; a process hard-killed at the approval prompt, then `--resume` in a new process approves → exactly 1 write |

Live checks (server running, real Bedrock):

| ID | Check |
|---|---|
| L1 | Server stopped → connection error, no answer |
| L2 | Server up → 6 tools discovered; grep shows no `db` / `rag` / `server` imports |
| L3 | Webex, approve → record with `pending_approval`; answer cites seats and role; per-worker timings show wall time ≈ slowest worker |
| L4 | Webex, reject → "not filed"; trajectory shows no write |
| L5 | "what's the remote work policy?" → knowledge only, no interrupt |
| L6 | Pause, kill the process, `--resume` in a fresh process → completes |
| L7 | `validate-lab/SKILL.md` Checks 1–3 scored by hand |

---

## 13. Validator mapping and open risks

| Rubric item | Where it is met | Tests |
|---|---|---|
| Per-request team selection (10) | §5 | T1–T3, L5 |
| Specialists with narrow toolsets (10) | §6 | T4, T5, T13 |
| Parallel fan-out + reducer merge (12) | §2, §3 | T6, L3 |
| Human gate on the write, incl. reject (10) | §4, §7, §9 | T7–T10, T21, T22, T25, T28 |
| Durable pause (5) | §9 | T14–T16c, T19a–c, T20, T25–T27, L6 |
| MCP discovery, one provider seam (3) | §6, §10 | T18, L1, L2 |
| 2a real MCP client · 2b interrupt only for the write · 2c reject writes nothing | §6, §7, §8 | L1–L4, T7–T10 |
| Check 3 semantic | §5, §6, `respond` | L3–L5 |
| Bonus: persistent checkpointer, no-specialist path, specs as data, richer payload, timings | §9, §3, §6, §7 | T3, T14, L3 |

Open risks — reported, not assumed away:
1. **Route shape.** `respond` satisfies the written requirement, but a grader comparing against
   the reference `human_review` may expect `reject → model`. Fallback with the same safety:
   reject → `act` with no tools bound once a decision exists.
2. **Default checkpointer.** `build_graph(tools)` alone pauses in memory; durability lives in
   the CLI. A grader reading only `build_graph` may score the durable-pause bonus lower.
3. **Model nondeterminism.** Nova Lite may plan wrongly or decline to propose the write, so 2b
   can fail on an individual live run. Mitigated by prompts, not guaranteed. Observed in live
   checks: 1 of 5 Webex runs did not reach the gate — the eligibility specialist attempted a 4th
   tool call (a second policy), was marked `incomplete`, and the write path was disabled as §11
   requires (the employee was told nothing was filed). The eligibility prompt now says to read
   the employee record only if needed, call `get_policy` once for the single policy relevant to
   the requested software, and avoid repeated calls (T4b); limits and failure handling are
   unchanged. After the change: 20 of 20 live Webex runs reached the gate (eligibility `ok`,
   3 calls each), with 0 writes before approval and 1 `pending_approval` record after it.
4. **Layout.** The rubric assumes a separate clean build repo; here `SDD/` sits inside the
   reference repo. The rubric allows the build path to be `.`; `../code/` is present for comparison.
5. **Decisions live here, not in `PROMPT.md`.** The stub suggests folding decisions into the
   prompt itself; this build records them in `SPEC.md` and `PROMPT.md` carries a one-line pointer.
   A grader looking only at `PROMPT.md` sees the stub.
6. **Request-name rule limits (D13, §4 point 2).** The proposed software must appear literally in
   the employee's request, so:
   - **descriptive-only requests cannot be filed** ("my video-call app says I'm not licensed") —
     the employee is asked to name the software;
   - **aliases need the literal name** ("Teams" does not name "Microsoft Teams" unless the request
     contains the system's exact name as written in the catalog);
   - **non-English requests will not match English system names** unless the employee writes the
     system's name as it appears in the catalog.
7. **Reply after a substring-only lookup — resolved by D15.** Before D15, 4 of 5 live replies
   suggested a candidate although every finding marked it unconfirmed; the answer is now a fixed
   text whenever trigger (i) or (ii) applies.
8. **Mixed case.** When one name passes `verify` and another is unmatched, the employee reads the
   model's reply followed by the fixed facts. The model's own text can still mention a candidate
   or repeat the eligibility specialist's text about the unmatched name; the approval prompt warns
   the operator, but the reply text itself is not filtered.
9. **No `records` specialist in the plan.** Only `records` performs the subscription lookup, so
   without it no marker exists, trigger (i) cannot fire, and the model writes the reply (a proposal
   is still checked by `verify`, so trigger (ii) and the write safety still hold).
10. **Eligibility text about a nonexistent system.** The eligibility specialist cannot see the
    subscription lookup, so it may describe an approval path for "AI". Under D15 that text never
    reaches the employee when trigger (i) or (ii) applies, and it reaches an approval prompt only in
    the mixed case, below the unmatched-name warning (§7). It can reach the employee in the mixed
    case (risk 8) and without `records` (risk 9).
11. **Reply wording is prompt-level.** The D17 reply rules (no "approved/granted" claims, no internal
    names, no invented justification) are instructions to the model, not guarantees; only the fixed
    answers (D15) and the `Outcome:` line are written by code. The approver sees the original request
    next to the model-written justification and must reject a mismatch. Live after D17: justifications
    quoted the employee's own words, no reply named internal tools or asked for an "eligibility
    report", but a rejected request's reply still gave its own reason ("over its seat limit") instead
    of the approver's, and one reply claimed an unknown employee "currently has access to Jira".
    Filing rate for an over-limit request is nondeterministic: "add WEBEX license x 2" reached the
    gate in 5 of 7 runs with the final prompt (3 of 4 with the pre-D17 prompt).
12. **One employee, one software, one request.** The server files one row per call; quantities
    ("x 2") are not represented. Bulk requests get a fixed reply (D18), but only when the planner
    flags them; one employee with several named systems is allowed and can still exceed a
    specialist's 3-tool-call budget, which marks it `incomplete` and disables filing.
13. **Beneficiary.** A request for someone who is not an employee (e.g. a relative) has no exact
    `employee_id` and cannot be filed for that person; if the model files it under the requesting
    employee's id, only the approver can catch it.

---

## 14. Security and publishing

- `README.md` had a GitHub personal access token embedded in its clone URL. It has been removed
  from `README.md`. The unmodified original is kept locally as `.README.orig.md`, which is
  gitignored and must never be committed (`git add -f` would override the ignore — don't).
- The token is also present in **this repository's history** (commit `82d8255`) and in
  `.git/config`'s `origin` URL. Removing it from the working file does not remove it from
  history, so a portfolio repository must start from a fresh history — not a push or fork of
  this one — and must never reuse the `origin` URL.
- **Revoke the token if it is still valid** (GitHub → Settings → Developer settings → Personal
  access tokens). Removal from files is not revocation.
- `.env`, `.venv/`, IDE folders, checkpoint databases and the README backup are excluded by
  `SDD/.gitignore`, which also covers the case of `SDD/` being published as its own repository
  (the parent `.gitignore` would then no longer apply).
- Secret values are never printed by the build or its tooling.

---

## 15. Files and dependencies

| File | Change | Phase |
|---|---|---|
| `SPEC.md` | this document | done |
| `.gitignore` | new; secrets, venvs, Python caches, IDE, checkpoints, README backup | done |
| `requirements.txt` | `langgraph-checkpoint-sqlite==3.1.2`; `mcp<2` kept | done |
| `README.md` | token removed from the clone URL (one line) | done |
| `.README.orig.md` | local, ignored backup of the original README | done |
| `CLAUDE.md` | points to `SPEC.md` as the authoritative contract; server is a fixed dependency incl. the owner's change; `MCP_SERVER_URL` required; `.env.example` described as tracked placeholders | done |
| `PROMPT.md` | one comment line pointing to `SPEC.md` | done |
| `.env.example` | appended `MCP_SERVER_URL=http://127.0.0.1:9876/mcp` (not a secret) | done |
| `server/server.py` | owner's local change; fixed dependency — never inspected, modified or reverted by the build | owner |
| `supervisor.py` | new | done |
| `tests/test_supervisor.py`, `tests/fakes.py` | new; offline suite (T1–T20, T4b) and its fakes | done |
| `tests/test_cli.py` | new; CLI acceptance tests T21–T27 | CLI fixes |
| `.env` | appended `MCP_SERVER_URL=http://127.0.0.1:9876/mcp` (local, ignored) | done |
| `README.md` | rewritten for the standalone repository; "Running" lists the CLI modes | publication |

Install (done): `pip install "langgraph-checkpoint-sqlite==3.1.2" "mcp<2"` —
adds `aiosqlite 0.22.1` and `sqlite-vec 0.1.9`; upgrades nothing (installed
`langgraph-checkpoint 4.3.0` satisfies its `>=4.3.0,<5` range). `setup.sh` installs from
`requirements.txt`, so a fresh setup picks the new pin up automatically.
