"""NovaOps assistant: a supervisor over specialist sub-agents, with a human on the write.

    plan     ── the model picks the specialists THIS request needs (structured output)
      │  Send × k                                    (k = 0 → straight to act)
      ▼
    worker   ── knowledge · records · eligibility, one parallel superstep; each is its own
      │         create_agent with its own few read-only tools; findings merge via operator.add
      ▼
    act      ── reads the merged evidence; may propose create_access_request
      │
    review   ── interrupt(): a human approves or rejects the proposed call(s)
      ├─ approve → write   (sends the checkpointed args unchanged, once per employee+software)
      └─ reject  → one denial ToolMessage per pending call
      ▼
    respond  ── the final answer to the employee (no tools bound)

The contracts behind every choice are in SPEC.md.

Run (with the MCP server running in another terminal):
    python supervisor.py                       # the Webex case
    python supervisor.py "some other request"
    python supervisor.py --resume <thread-id>  # continue a paused approval
"""
import asyncio
import json
import operator
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field

REQUIRED_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "BEDROCK_MODEL_ID",
                "BEDROCK_EMBEDDING_MODEL_ID", "MCP_SERVER_URL")
DEFAULT_REQUEST = "I'm E010 and Webex says I'm not licensed — can you help?"
CHECKPOINT_DB = Path(__file__).resolve().parent / "checkpoints.sqlite"

# The server marks no tool as read-only, so the one state-changing tool has to be named here.
WRITE_TOOL = "create_access_request"
# Read-only lookup `verify` uses to prove a proposed software name is exact and over its seat limit.
SUBSCRIPTION_TOOL = "check_software_subscription"
# Read-only lookup `verify` uses to prove the employee exists; filing needs the exact id.
EMPLOYEE_TOOL = "get_employee"
NOT_NAMED = "was never named in your request"
# Output cap for the replies the employee reads (act, respond); longest normal live answer ~350 tokens.
ANSWER_MAX_TOKENS = 800
ASK_NAME = "Please tell us the exact name of the software you need."
ASK_EMPLOYEE = "Please tell us the employee ID."
UNIDENTIFIED = "Your request does not identify the employee exactly."
BULK_ANSWER = ("Each access request covers one employee and software they name one by one. Nothing was "
               "filed. Please send a separate request for each employee, naming each software you need "
               "(for example: \"<employee ID> needs <software>\").")

# The bound is what keeps a sub-agent from wandering; ToolCallLimitMiddleware enforces it.
_BOUND = " At most 3 tool calls. Facts only, 2–3 sentences. Do not decide or file anything."
WORKER_SPECS = {
    "knowledge": {
        "tools": ["search_knowledge_base", "list_policies", "get_policy"],
        "prompt": "You are the Knowledge specialist. Search the IT knowledge base for how-to "
                  "facts; for policy questions, find the single most relevant policy with "
                  "list_policies and read it with get_policy." + _BOUND,
    },
    "records": {
        "tools": ["get_employee", "check_software_subscription"],
        "prompt": "You are the Records specialist. Look up the employee's record and the "
                  "software's seat/subscription status, and report those facts. The lookup matches "
                  "names partially, so report seat/subscription status only for a row whose "
                  "system_name equals the requested software (ignoring case). If no row matches "
                  "exactly, say \"no exact match for <name>\" and list the returned rows only as "
                  "unconfirmed candidates, never as the requested software." + _BOUND,
    },
    "eligibility": {
        "tools": ["get_employee", "list_policies", "get_policy"],
        "prompt": "You are the Eligibility specialist. Report whether the employee's role may use "
                  "the requested software and who approves access. Read the employee record only "
                  "if the request does not already give their role. Use list_policies once, then "
                  "read the single policy most relevant to the requested software: call get_policy "
                  "only once, never for a second policy. Avoid repeated or unnecessary tool calls; "
                  "give a concise finding from the evidence you have. Name the software exactly as "
                  "the employee wrote it and never substitute a different system." + _BOUND,
    },
}

PLANNER_PROMPT = (
    "You are the supervisor of a NovaOps IT specialist team. Choose which specialists to "
    "consult for the employee's request — only the relevant ones:\n"
    "- knowledge: how-to facts from the IT knowledge base, and what a policy document says\n"
    "- records: the employee's record and a software's seat/subscription status\n"
    "- eligibility: whether the employee may use a software, and who approves it\n"
    "Return an empty list if the request needs no lookups (e.g. a greeting). Set "
    "may_file_request to true only if the employee lacks access to a system and wants it."
)
# Live replies claimed "access will be automatically granted", asked employees for an
# "eligibility report" and named internal tools; these rules apply to every reply.
_REPLY_RULES = (
    " Never say that access is approved, granted, provisioned or assigned — you can only file a "
    "request that a human must approve. Never mention internal checks, specialists or tool names, "
    "and never ask the employee for internal reports or documents."
    # Live: "Webex is at its seat limit, so I cannot file a request for you".
    " A subscription being over its seat limit is the reason a request is filed, never a reason "
    "it cannot be."
)
ACT_PROMPT = (
    "You are the NovaOps supervisor. Your team gathered the evidence below. If the employee is "
    "eligible and the subscription is over its seat limit, file an access request with "
    "create_access_request (a human approves it before it is sent). Otherwise answer the "
    "employee clearly from the evidence." + _REPLY_RULES +
    " File only for the employee named in the request, for their own work, with the reason they "
    "gave — never invent a justification."
)
_ACT_NO_FILING = (
    "\n\nFiling is disabled for this request because an internal check could not finish. Answer "
    "what you can from the evidence, say plainly that no access request was filed, and suggest "
    "that they try again later."
)
_ACT_READ_ONLY = "\n\nNo access request is needed for this request; just answer it."
RESPOND_PROMPT = (
    "You are the NovaOps supervisor writing the final reply to the employee. Use the evidence "
    "and the outcome of the proposed access request. Say plainly whether a request was filed "
    "(and its id and status) or that it was not filed and why. If the approver rejected it, say so "
    "and give only the reason the approver stated. Be brief and grounded." + _REPLY_RULES
)


class Plan(BaseModel):
    workers: list[Literal[*WORKER_SPECS]] = Field(
        description="The specialists to consult for this request; may be empty.")
    may_file_request: bool = Field(
        description="True only if the employee lacks access to a system and wants it.")
    bulk: bool = Field(default=False, description=(
        "True if the request is for more than one employee (a list, a range, a team, 'all employees') "
        "or for software not named one by one ('all software', 'everything')."))


class Finding(TypedDict):
    agent: str
    status: Literal["ok", "incomplete", "failed"]
    summary: str
    seconds: float
    tool_calls: int
    unmatched: list[str]
    unknown_employees: list[str]
    ambiguous_employees: list[str]


class WorkerInput(TypedDict):
    name: str
    request: str


class SupervisorState(TypedDict, total=False):
    request: str
    plan: list[str]
    may_file_request: bool
    findings: Annotated[list[Finding], operator.add]   # reducer: parallel branches append here
    messages: Annotated[list[AnyMessage], add_messages]
    checks: dict[str, dict]                             # tool_call_id → subscription row `verify` matched
    name_problems: list[dict]                           # verify's name failures: {"kind", "name"}
    writes_done: dict[str, str]                         # duplicate key → first server result
    error: str
    answer: str


def initial_state(request: str) -> SupervisorState:
    return {"request": request, "plan": [], "may_file_request": False, "findings": [],
            "messages": [], "writes_done": {}}


def require_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing environment variables: {', '.join(missing)}")


def load_config() -> None:
    # Loaded on first need, not at import, so `--help` and `--status` never read .env.
    load_dotenv(find_dotenv())


def get_model():
    """The one provider seam: swap this function to change the chat model."""
    load_config()
    require_env()
    return ChatBedrockConverse(model=os.environ["BEDROCK_MODEL_ID"],
                               region_name=os.environ["AWS_REGION"])


async def load_tools() -> list:
    load_config()
    client = MultiServerMCPClient(
        {"novaops": {"url": os.environ["MCP_SERVER_URL"], "transport": "streamable_http"}})
    return await client.get_tools()


def trace(node: str, message: str) -> None:
    print(f"  [{node:11}] {message}")


def text_of(message) -> str:
    return message.text if hasattr(message, "text") else str(message)


def evidence_text(findings: list[Finding]) -> str:
    return "\n\n".join(f"[{f['agent']} — {f['status']}]\n{f['summary']}" for f in findings) or "(none)"


def duplicate_key(args: dict) -> str | None:
    employee, software = args.get("employee_id"), args.get("software")
    if not (isinstance(employee, str) and employee.strip() and isinstance(software, str) and software.strip()):
        return None
    return f"{employee.strip().upper()}|{software.strip().lower()}"


def json_rows(message) -> list[dict]:
    """JSON rows of a tool result; the MCP adapter returns each row as its own text block."""
    blocks = message.content if isinstance(message.content, list) else [message.content]
    rows = []
    for block in blocks:
        try:
            value = json.loads(block.get("text", "") if isinstance(block, dict) else str(block))
        except ValueError:
            continue
        rows.extend(value if isinstance(value, list) else [value])
    return [row for row in rows if isinstance(row, dict)]


def mark_substring_rows(message: ToolMessage, software: str) -> ToolMessage:
    """Prefixes a subscription lookup with NO EXACT MATCH when no row's name equals `software`."""
    wanted = software.strip()
    rows = json_rows(message)
    if message.status == "error" or any(str(r.get("system_name", "")).strip().lower() == wanted.lower() for r in rows):
        return message
    names = ", ".join(str(r.get("system_name")) for r in rows) or "none"
    return prefixed(message, f"NO EXACT MATCH for '{wanted}'; these rows are substring matches only "
                             f"(unconfirmed candidates, not '{wanted}'): {names}.")


def mark_missing_employee(message: ToolMessage, query: str) -> ToolMessage:
    """Prefixes an employee lookup that returned no one with NO EMPLOYEE FOUND."""
    if message.status == "error" or json_rows(message):
        return message
    return prefixed(message, f"NO EMPLOYEE FOUND for '{query.strip()}'.")


def prefixed(message: ToolMessage, note: str) -> ToolMessage:
    content = message.content
    content = [{"type": "text", "text": note}, *content] if isinstance(content, list) else f"{note}\n{content}"
    return message.model_copy(update={"content": content})


_MARKER = re.compile(r"NO EXACT MATCH for '(.*)'; these rows are substring matches only", re.DOTALL)
_NO_EMPLOYEE = re.compile(r"NO EMPLOYEE FOUND for '(.*)'\.$", re.DOTALL)
_MARKDOWN = re.compile(r"[`*_#>\[\]()~|]")


def plain(raw) -> str:
    """Newlines and markdown removed, whitespace collapsed — not clipped."""
    name = _MARKDOWN.sub("", str(raw).replace("\r", " ").replace("\n", " "))
    return " ".join(name.split())


def clean_name(raw) -> str:
    """A model-supplied name made safe to quote in a fixed answer."""
    return plain(raw)[:60].strip()


def named_in(name, request) -> bool:
    """`name` appears in the employee's request as a whole word (case-insensitive)."""
    wanted = plain(name).lower()
    return bool(wanted) and re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", plain(request).lower()) is not None


def unmatched_in(messages, marker=_MARKER) -> list[str]:
    """Names whose lookup ExactNameLookups marked, read from tool results only."""
    names = []
    for m in messages:
        if isinstance(m, ToolMessage):
            first = m.content[0] if isinstance(m.content, list) and m.content else m.content
            match = marker.match(first.get("text", "") if isinstance(first, dict) else str(first))
            if match:
                names.append(plain(match.group(1)))
    return names


def unmatched_names(state: dict, key: str = "unmatched") -> list[str]:
    """Unmatched lookups for names the employee actually wrote. A specialist may look up a name
    nobody asked for (live: "Confluence" for "what SW can I get?"); that must not drive the answer."""
    names = []
    for finding in state.get("findings", []):
        for name in finding.get(key, []):
            shown = clean_name(name)
            if named_in(name, state.get("request", "")) and shown.lower() not in (n.lower() for n in names):
                names.append(shown)
    return names


def reversed_name(name: str) -> str:
    """The words of a multi-word name in reverse order ("cohen daniel" -> "daniel cohen"), else ""."""
    words = str(name).split()
    return " ".join(reversed(words)) if len(words) > 1 else ""


def full_name_in(full: str, request: str) -> bool:
    return named_in(full, request) or bool(reversed_name(full)) and named_in(reversed_name(full), request)


def named_by(person: dict, rows: list[dict], request: str) -> bool:
    """The request names this employee by id, email, or a full name no other row in `rows` has.
    The server matches names by substring and the seed data repeats first names on purpose."""
    if named_in(person.get("employee_id", ""), request) or named_in(person.get("email") or "", request):
        return True
    full = str(person.get("full_name") or "")
    return full_name_in(full, request) and len(
        [r for r in rows if str(r.get("full_name", "")).lower() == full.lower()]) == 1


async def identified(person: dict, request: str, find) -> bool:
    full = str(person.get("full_name") or "")
    return named_by(person, await find(full) if full_name_in(full, request) else [], request)


def ambiguous_in(messages, request: str) -> list[str]:
    """Employee lookups for a name the employee wrote that matched several people, none of whom
    the request identifies (live: "maya" — the reply then listed both Mayas)."""
    queries = {c["id"]: str(c["args"].get("query", ""))
               for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    names = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == EMPLOYEE_TOOL and m.status != "error":
            rows, query = json_rows(m), queries.get(m.tool_call_id, "")
            if len(rows) > 1 and named_in(query, request) and not any(named_by(r, rows, request) for r in rows):
                names.append(plain(query))
    return names


def name_facts(state: dict) -> list[tuple[str, str]]:
    """(kind, sentence) per unmatched or blocked name, each fact once (SPEC §4 Answer rules)."""
    facts, seen = [], set()

    def add(kind, name, sentence):
        if (kind, name.lower()) not in seen and (kind, sentence) not in facts:
            seen.add((kind, name.lower()))
            facts.append((kind, sentence))

    for name in unmatched_names(state):
        add("no_exact", name, f"No system named exactly '{name}' was found.")
    # Live: E099 was found by no lookup, nothing was proposed, and the reply called E099 "eligible".
    for name in unmatched_names(state, "unknown_employees"):
        add("no_employee", name, f"No employee matching '{name}' was found.")
    for name in unmatched_names(state, "ambiguous_employees"):
        add("no_employee", name, UNIDENTIFIED)
    for problem in state.get("name_problems", []):
        name = clean_name(problem["name"])
        if problem["kind"] == "no_exact":
            add("no_exact", name, f"No system named exactly '{name}' was found.")
        elif problem["kind"] == "no_employee" and named_in(problem["name"], state.get("request", "")):
            add("no_employee", name, f"No employee with the ID '{name}' was found.")
        elif problem["kind"] in ("no_employee", "unidentified"):
            # Live: a model-invented id ("E?not_found") was echoed back; only ids the employee wrote are.
            add("no_employee", name, UNIDENTIFIED)
        elif problem["kind"] == "seats_available":
            # Live: the model promised "you will be provisioned"; nothing is assigned by this assistant.
            add("seats_available", name,
                f"{name} has seats available ({problem.get('active')} of {problem.get('limit')} in use), so no "
                "access request was filed — one is only needed when a subscription is over its seat limit. "
                "Nothing has been assigned to you yet: open an IT ticket to have a seat assigned (your manager "
                "approves it).")
        else:
            add("not_named", name, f"The proposed system '{name}' {NOT_NAMED}.")
    # Owner: someone not identified as an employee is told nothing about systems or seats.
    employee = [fact for fact in facts if fact[0] == "no_employee"]
    if employee:
        # The generic sentence adds nothing next to a specific one.
        return [fact for fact in employee if fact[1] != UNIDENTIFIED] or employee[:1]
    return facts


def asks_for(facts: list[tuple[str, str]]) -> list[str]:
    asks = [ASK_EMPLOYEE] if any(kind == "no_employee" for kind, _ in facts) else []
    return asks + ([ASK_NAME] if any(kind in ("no_exact", "not_named") for kind, _ in facts) else [])


def fixed_answer(facts: list[tuple[str, str]]) -> str:
    # The free-seats sentence already says nothing was filed and what to do instead.
    closing = ["Nothing was filed."] + asks_for(facts) if any(k != "seats_available" for k, _ in facts) else []
    return " ".join([sentence for _, sentence in facts] + closing)


def truncated(reply) -> bool:
    # Structural: the provider's stop reason, never the text. Bedrock says "max_tokens".
    meta = getattr(reply, "response_metadata", None) or {}
    return any(meta.get(key) in ("max_tokens", "length") for key in ("stopReason", "stop_reason", "finish_reason"))


async def answer_call(runnable, messages, node: str):
    """A user-visible reply that was not cut off, or None after one retry."""
    for attempt in (1, 2):
        reply = await runnable.ainvoke(messages)
        if not truncated(reply):
            return reply
        trace(node, f"reply cut off at the output limit ({ANSWER_MAX_TOKENS} tokens) — "
                    + ("retrying once" if attempt == 1 else "using the fixed reply"))
    return None


class ExactNameLookups(AgentMiddleware):
    """The server matches names by substring ("AI" -> MailGuard), so a specialist reads a lookup
    with no exact-name row only after it is marked as unconfirmed candidates."""

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if request.tool_call["name"] == SUBSCRIPTION_TOOL and isinstance(result, ToolMessage):
            return mark_substring_rows(result, str(request.tool_call["args"].get("software", "")))
        if request.tool_call["name"] == EMPLOYEE_TOOL and isinstance(result, ToolMessage):
            query = str(request.tool_call["args"].get("query", ""))
            flipped = reversed_name(query)
            # The server matches a name's words in order (live: "cohen daniel" found no one).
            if flipped and result.status != "error" and not json_rows(result):
                retry = await handler(request.override(
                    tool_call={**request.tool_call, "args": {**request.tool_call["args"], "query": flipped}}))
                if isinstance(retry, ToolMessage) and retry.status != "error" and json_rows(retry):
                    return prefixed(retry, f"No match for '{query.strip()}'; found by the name in reverse order ('{flipped}').")
            return mark_missing_employee(result, query)
        return result


def build_graph(tools: list, model=None, checkpointer=None):
    by_name = {t.name: t for t in tools}
    for required in (WRITE_TOOL, SUBSCRIPTION_TOOL, EMPLOYEE_TOOL):
        if required not in by_name:
            raise ValueError(f"the MCP server did not offer {required}")
    for name, spec in WORKER_SPECS.items():
        if WRITE_TOOL in spec["tools"]:
            raise ValueError(f"specialist {name!r} must not hold {WRITE_TOOL}")
        unknown = [t for t in spec["tools"] if t not in by_name]
        if unknown:
            raise ValueError(f"specialist {name!r} names undiscovered tools: {unknown}")
    if model is None:
        model = get_model()
    write_tool, subscription_tool = by_name[WRITE_TOOL], by_name[SUBSCRIPTION_TOOL]
    employee_tool = by_name[EMPLOYEE_TOOL]

    workers = {
        name: create_agent(
            model, [by_name[t] for t in spec["tools"]], system_prompt=spec["prompt"],
            # "continue" answers over-limit calls with an error ToolMessage instead of running them.
            middleware=[ToolCallLimitMiddleware(run_limit=3, exit_behavior="continue"), ExactNameLookups()])
        for name, spec in WORKER_SPECS.items()
    }

    async def plan(state: SupervisorState) -> dict:
        # include_raw=True turns a malformed answer into parsed=None instead of an exception.
        chooser = model.with_structured_output(Plan, include_raw=True)
        messages = [SystemMessage(PLANNER_PROMPT), HumanMessage(state["request"])]
        for attempt in (1, 2):
            picked = (await chooser.ainvoke(messages))["parsed"]
            if picked is not None:
                break
            trace("plan", f"attempt {attempt}: the model returned no valid plan")
        else:
            return {"error": "planner_failed",
                    "answer": "Sorry — I could not work out how to route your request, so nothing "
                              "was looked up or filed. Please rephrase it or contact IT directly."}
        chosen = list(dict.fromkeys(picked.workers))
        if picked.may_file_request and picked.bulk:
            # Live: bulk requests exhausted the specialists' tool budget or became a request for a
            # system nobody named; one request covers one employee, so nothing is looked up.
            trace("plan", "bulk access request — fixed answer, no lookups")
            return {"plan": [], "may_file_request": False, "answer": BULK_ANSWER}
        trace("plan", f"team: {', '.join(chosen) or '(none)'}   may_file_request={picked.may_file_request}")
        return {"plan": chosen, "may_file_request": picked.may_file_request}

    def route_after_plan(state: SupervisorState):
        if state.get("error") or state.get("answer"):
            return END
        sends = [Send("worker", WorkerInput(name=n, request=state["request"])) for n in state.get("plan", [])]
        return sends or "act"

    async def worker(payload: WorkerInput) -> dict:
        name, started = payload["name"], time.perf_counter()
        trace(name, f"working ... (tools: {', '.join(WORKER_SPECS[name]['tools'])})")
        status, summary, used, unmatched, unknown, ambiguous = "ok", "", 0, [], [], []
        try:
            result = await workers[name].ainvoke(
                {"messages": [HumanMessage(payload["request"])]}, config={"recursion_limit": 12})
            messages = result["messages"]
            unmatched = unmatched_in(messages)
            unknown = unmatched_in(messages, _NO_EMPLOYEE)
            ambiguous = ambiguous_in(messages, payload["request"])
            results = [m for m in messages if isinstance(m, ToolMessage)]
            blocked = [m for m in results if str(m.content).startswith("Tool call limit exceeded")]
            used = len(results) - len(blocked)
            last = messages[-1]
            if isinstance(last, AIMessage) and not last.tool_calls:
                summary = text_of(last).strip()
            if blocked or not summary:
                status = "incomplete"
        except GraphRecursionError:
            status = "incomplete"
        except Exception as exc:
            status, summary = "failed", f"{type(exc).__name__}: {str(exc)[:200]}"
        seconds = time.perf_counter() - started
        trace(name, f"{status} in {seconds:.1f}s ({used} tool calls)")
        return {"findings": [Finding(agent=name, status=status, summary=summary or "No report produced.",
                                     seconds=round(seconds, 2), tool_calls=used,
                                     unmatched=list(dict.fromkeys(unmatched)),
                                     unknown_employees=list(dict.fromkeys(unknown)),
                                     ambiguous_employees=list(dict.fromkeys(ambiguous)))]}

    async def act(state: SupervisorState) -> dict:
        findings = state.get("findings", [])
        may_file = bool(state.get("may_file_request"))
        can_file = may_file and bool(findings) and all(f["status"] == "ok" for f in findings)
        note = "" if can_file else (_ACT_NO_FILING if may_file else _ACT_READ_ONLY)
        context = HumanMessage(f"Request: {state['request']}\n\nEvidence:\n{evidence_text(findings)}")
        actor = (model.bind_tools([write_tool]) if can_file else model).bind(max_tokens=ANSWER_MAX_TOKENS)
        # A cut-off reply is dropped before its tool_calls are read: a truncated call is malformed.
        reply = await answer_call(actor, [SystemMessage(ACT_PROMPT + note), context], "act")
        facts = name_facts(state)
        if reply is None:
            answer = (fixed_answer(facts) if facts else
                      f"Sorry — I could not produce a complete answer. Outcome: {outcome_of(state)}.")
            return {"messages": [context, AIMessage(answer)], "answer": answer}
        trace("act", f"write tool bound: {can_file}   proposed calls: {len(reply.tool_calls)}")
        update = {"messages": [context, reply]}
        if not reply.tool_calls:
            # Trigger (i): a name had no exact system and nothing was proposed — the model's
            # text is not used, so it can never point the employee at a substring match.
            update["answer"] = fixed_answer(facts) if facts else text_of(reply)
            if facts:
                trace("act", "fixed answer (model text not used): " + " ".join(s for _, s in facts))
        return update

    def route_after_act(state: SupervisorState):
        last = state["messages"][-1]
        return "verify" if isinstance(last, AIMessage) and last.tool_calls else END

    async def verify(state: SupervisorState) -> Command:
        # The server resolves names by substring ("AI" -> MailGuard), so a proposal reaches the
        # approver only if its name is an exact subscription name that is over its seat limit.
        pending = state["messages"][-1].tool_calls
        lookups, people, checks, failures, problems = {}, {}, {}, {}, []
        for call in pending:
            if call["name"] != WRITE_TOOL:
                continue
            if duplicate_key(call["args"]) is None:
                failures[call["id"]] = "employee_id and software are required"
                continue
            software, employee = call["args"]["software"].strip(), call["args"]["employee_id"]
            wanted, reasons, row = software.lower(), [], None
            # Filing accepts only an exact, case-sensitive employee id (live: "SAAS" was approved,
            # then refused by the server), so the employee must be found exactly as proposed.
            async def find(query):
                if query not in people:
                    people[query] = json_rows(await employee_tool.ainvoke(
                        {"type": "tool_call", "id": f"verify-employee-{call['id']}-{len(people)}",
                         "name": EMPLOYEE_TOOL, "args": {"query": query}}))
                return people[query]

            person = next((r for r in await find(employee) if str(r.get("employee_id")) == employee), None)
            if person is None:
                reasons.append(f"no employee with the ID '{employee}' was found")
                problems.append({"kind": "no_employee", "name": employee})
            elif not await identified(person, state["request"], find):
                # Live: "maya" matched two employees and the model filed for one of them.
                # The guessed id is not shown: it would tell the requester who matched.
                reasons.append("the request does not identify the employee exactly")
                problems.append({"kind": "unidentified", "name": employee})
            if reasons:
                # Owner: no system details for a requester who is not identified as an employee.
                failures[call["id"]] = "; ".join(reasons)
                continue
            # The model may substitute a system the employee never named (live: an "AI" request
            # became a Webex proposal), so the name must appear in the request as a whole word.
            if not named_in(software, state["request"]):
                reasons.append(f"the proposed system '{software}' {NOT_NAMED}. "
                               "Please tell us the exact name of the software you need")
                problems.append({"kind": "not_named", "name": software})
            else:
                if wanted not in lookups:
                    lookups[wanted] = await subscription_tool.ainvoke(
                        {"type": "tool_call", "id": f"verify-{call['id']}", "name": SUBSCRIPTION_TOOL,
                         "args": {"software": software}})
                result = lookups[wanted]
                rows = json_rows(result)
                exact = [r for r in rows if str(r.get("system_name", "")).strip().lower() == wanted]
                if result.status == "error":
                    reasons.append(f"the subscription lookup failed: {text_of(result)[:200]}")
                elif len(exact) != 1:
                    names = ", ".join(str(r.get("system_name")) for r in rows)
                    reasons.append(f"'{software}' is not an exact system name — "
                                   + (f"the server would match: {names}" if names else "no subscription has that name"))
                    problems.append({"kind": "no_exact", "name": software})
                elif exact[0].get("over_limit") is not True:
                    reasons.append(f"{exact[0]['system_name']} has seats available ({exact[0].get('active_seats')} of "
                                   f"{exact[0].get('seat_limit')} in use), so no access request is needed")
                    problems.append({"kind": "seats_available", "name": exact[0]["system_name"],
                                     "active": exact[0].get("active_seats"), "limit": exact[0].get("seat_limit")})
                else:
                    row = exact[0]
            if reasons:
                failures[call["id"]] = "; ".join(reasons)
                continue
            checks[call["id"]] = {k: row.get(k) for k in ("system_name", "seat_limit", "active_seats", "over_limit")}
        if not failures:
            trace("verify", "passed: " + ", ".join(f"{c['system_name']} {c['active_seats']}/{c['seat_limit']}"
                                                    for c in checks.values()))
            return Command(goto="review", update={"checks": checks})
        trace("verify", "not sent for approval: " + "; ".join(failures.values()))
        not_filed = [ToolMessage(f"NOT FILED: {failures.get(c['id'], 'another proposed request in the same turn failed the pre-approval check')}.",
                                 tool_call_id=c["id"], name=c["name"], status="error") for c in pending]
        return Command(goto="respond", update={"messages": not_filed, "name_problems": problems})

    def review(state: SupervisorState) -> Command:
        # Re-runs from the top on resume, so nothing before interrupt() may have side effects.
        pending = state["messages"][-1].tool_calls
        decision = interrupt({
            "kind": "approval_request",
            "question": "Approve filing this access request?",
            "requests": [{"tool_call_id": c["id"], "tool": c["name"], "args": c["args"],
                          "resolved": state.get("checks", {}).get(c["id"])} for c in pending],
            "evidence": [{"agent": f["agent"], "status": f["status"], "summary": f["summary"]}
                         for f in state.get("findings", [])],
            "request": state.get("request", ""),
            "unmatched": unmatched_names(state),
        })
        if isinstance(decision, dict) and decision.get("approved") is True:
            trace("review", "approved")
            return Command(goto="write")
        reason = (decision.get("reason") if isinstance(decision, dict) else None) or "no reason given"
        trace("review", f"rejected: {reason}")
        # Every pending tool call must get a result, or the transcript is malformed.
        denials = [ToolMessage(f"DENIED by approver: {reason}. The access request was not filed.",
                               tool_call_id=c["id"], name=c["name"], status="error") for c in pending]
        return Command(goto="respond", update={"messages": denials})

    async def write(state: SupervisorState) -> dict:
        done = dict(state.get("writes_done", {}))
        results = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] != WRITE_TOOL:
                results.append(ToolMessage(f"ERROR: {call['name']} is not executable here.",
                                           tool_call_id=call["id"], name=call["name"], status="error"))
                continue
            key = duplicate_key(call["args"])
            if key is None:
                results.append(ToolMessage("ERROR: employee_id and software are required; nothing was filed.",
                                           tool_call_id=call["id"], name=call["name"], status="error"))
            elif key in done:
                results.append(ToolMessage(f"DUPLICATE: already filed in this run — {done[key]}",
                                           tool_call_id=call["id"], name=call["name"]))
            else:
                trace("write", f"calling {WRITE_TOOL} {call['args']}")
                # The approved call is sent exactly as checkpointed; the model never regenerates it.
                result = await write_tool.ainvoke(
                    {"type": "tool_call", "id": call["id"], "name": call["name"], "args": call["args"]})
                if result.status != "error":
                    filed = text_of(result)
                    system = next((str(r["system_name"]) for r in json_rows(result) if r.get("system_name")), None)
                    approved = call["args"]["software"].strip()
                    if system and system.strip().lower() != approved.lower():
                        filed = (f"WARNING: filed for '{system}', not the approved '{approved}' — "
                                 f"ask IT to correct it. Server result: {filed}")
                        result = ToolMessage(filed, tool_call_id=result.tool_call_id, name=result.name)
                    done[key] = filed
                trace("write", f"server: {text_of(result)[:160]}")
                results.append(result)
        return {"messages": results, "writes_done": done}

    async def respond(state: SupervisorState) -> dict:
        # Plain text in, no tools bound: a decided write can never be proposed again.
        outcome = "\n".join(f"- {text_of(m)}" for m in state["messages"] if isinstance(m, ToolMessage))
        prompt = (f"Request: {state['request']}\n\nEvidence:\n{evidence_text(state.get('findings', []))}"
                  f"\n\nOutcome of the proposed access request:\n{outcome}")
        facts = name_facts(state)
        if state.get("name_problems"):
            # Trigger (ii): verify blocked a name — a fixed answer; the model is not asked.
            answer = fixed_answer(facts)
            trace("respond", "fixed answer: " + " ".join(s for _, s in facts))
            return {"answer": answer, "messages": [AIMessage(answer)]}
        reply = await answer_call(model.bind(max_tokens=ANSWER_MAX_TOKENS),
                                  [SystemMessage(RESPOND_PROMPT), HumanMessage(prompt)], "respond")
        answer = (text_of(reply) if reply is not None else
                  f"Sorry — I could not produce a complete reply. Outcome: {outcome_of(state)}.")
        if facts:   # mixed case: the filed request stays; each unmatched name is still stated
            answer = "\n\n".join([answer, " ".join([s for _, s in facts] + asks_for(facts))])
        return {"answer": answer, "messages": [reply if reply is not None else AIMessage(answer)]}

    graph = StateGraph(SupervisorState)
    graph.add_node("plan", plan)
    graph.add_node("worker", worker)
    graph.add_node("act", act)
    graph.add_node("verify", verify, destinations=("review", "respond"))
    graph.add_node("review", review, destinations=("write", "respond"))
    graph.add_node("write", write)
    graph.add_node("respond", respond)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, ["worker", "act", END])
    graph.add_edge("worker", "act")   # fan-in barrier: waits for every branch
    graph.add_conditional_edges("act", route_after_act, ["verify", END])
    graph.add_edge("write", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


# ── CLI ─────────────────────────────────────────────────────────────────────────

USAGE = """\
Usage:
  python supervisor.py                       run the Webex demo request once
  python supervisor.py --demo                same as above
  python supervisor.py "request text"        run one request, then exit
  python supervisor.py -- "-request text"    a request that starts with "-"
  python supervisor.py -i | --interactive    several requests in one session (exit/quit to leave)
  python supervisor.py --resume THREAD_ID    continue a request paused at its approval prompt
  python supervisor.py --status [THREAD_ID]  list saved requests (or show one) and the next step
  python supervisor.py -h | --help           show this help

At an approval prompt, answer approve / yes / y / ok, or reject / no / n.
To pause at the prompt press Ctrl-C (or end the input: Ctrl-Z then Enter on Windows, Ctrl-D on
macOS/Linux); --resume continues the request later.
Every mode except --help and --status needs the NovaOps MCP server (cd server && python server.py)."""

APPROVE_ANSWERS = ("approve", "yes", "y", "ok")
REJECT_ANSWERS = ("reject", "no", "n")
REQUEST_PROMPT = "\nRequest (exit/quit to leave): "


_FLAGS = ("-h", "--help", "--demo", "-i", "--interactive", "--resume", "--status", "--")


def parse_args(argv: list[str]) -> tuple[str, str | None] | None:
    """(mode, value) for SPEC.md §9's argument table, or None for a usage error."""
    first, rest = (argv[0], argv[1:]) if argv else ("", [])
    if first in ("-h", "--help"):
        return "help", None
    if not argv or (first == "--demo" and not rest):
        return "demo", DEFAULT_REQUEST
    if first in ("-i", "--interactive") and not rest:
        return "interactive", None
    if first == "--resume" and len(rest) == 1:
        return "resume", rest[0]
    if first == "--status" and len(rest) <= 1:
        return "status", rest[0] if rest else None
    if first == "--" and " ".join(rest).strip():
        return "request", " ".join(rest).strip()
    # A flag hidden behind spaces ("  --resume …") or placed after the request ("text" --resume id)
    # is a mistake, never request text; `--` is the one way to send a request that looks like one.
    if first.lstrip().startswith("-") or any(arg.strip() in _FLAGS for arg in rest):
        return None
    return "request", " ".join(argv).strip() or DEFAULT_REQUEST


def new_thread_id() -> str:
    return f"case-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{secrets.token_hex(2)}"


def thread_config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def ask_human(payload: dict) -> dict:
    """Reads the approver's decision; raises EOFError / KeyboardInterrupt to pause."""
    while True:
        answer = input("Decision [approve/yes/y/ok | reject/no/n]: ").strip().lower()
        if answer in APPROVE_ANSWERS:
            return {"approved": True}
        if answer in REJECT_ANSWERS:
            reason = input("Reason for rejecting (optional): ").strip()
            return {"approved": False, "reason": reason or "rejected by approver"}
        print("  Please answer approve, yes, y or ok to file the request — or reject, no or n to refuse it.")


async def ask_in_thread(ask, payload):
    # A daemon thread, so a prompt still blocked on input() after Ctrl-C cannot hold the
    # process open; Ctrl-C cancels this await instead of being swallowed by the event loop.
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(setter, value):
        if not future.done():
            setter(value)

    def run():
        try:
            result = ask(payload)
        except BaseException as exc:
            loop.call_soon_threadsafe(settle, future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(settle, future.set_result, result)

    threading.Thread(target=run, daemon=True).start()
    return await future


def show_approval(payload: dict, thread: str) -> None:
    print(f"\n=== APPROVAL NEEDED — {payload['question']}  (thread {thread}) ===")
    if payload.get("request"):   # absent in approvals paused before this field existed
        print(f"  Employee's request: {payload['request']}")
    for name in payload.get("unmatched", []):
        print(f"  Warning: no system named exactly '{name}' exists — evidence about it is unconfirmed.")
    for item in payload["evidence"]:
        print(f"  evidence [{item['agent']} — {item['status']}] {item['summary']}")
    for item in payload["requests"]:
        print(f"  proposed write: {item['tool']}")
        for key, value in item["args"].items():
            if key == "business_justification":   # the model wrote it; the employee may have said otherwise
                key += " (written by the model — check it against the employee's request)"
            print(f"      {key}: {value}")
        resolved = item.get("resolved")
        if resolved:
            print(f"      -> server system: {resolved['system_name']} ({resolved['active_seats']} of "
                  f"{resolved['seat_limit']} seats in use, over limit)")
    print("  (To decide later: press Ctrl-C — or Ctrl-Z then Enter on Windows, Ctrl-D on macOS/Linux —"
          " then run --resume)")


def outcome_of(values: dict) -> str:
    """The run's result, read from state — never from the model's prose."""
    if values.get("error"):
        return "the request could not be routed — nothing was looked up or filed"
    if values.get("writes_done"):
        return "access request filed — " + "; ".join(" ".join(r.split()) for r in values["writes_done"].values())
    results = [m for m in values.get("messages", []) if isinstance(m, ToolMessage)]
    if any(str(m.content).startswith("DENIED") for m in results):
        return "rejected by the approver — nothing was filed"
    not_filed = [str(m.content) for m in results if str(m.content).startswith("NOT FILED: ")]
    if not_filed:
        return "not filed — " + not_filed[0].removeprefix("NOT FILED: ")
    if results:
        return "the server refused the write — nothing was filed"
    return "no access request was proposed"


def stopped_message(snapshot) -> str:
    """Explains a thread that stopped outside an approval pause (SPEC.md §9, case 4)."""
    values = snapshot.values
    if "write" in snapshot.next:
        last = values["messages"][-1]
        approved = [c["args"] for c in getattr(last, "tool_calls", [])]
        return ("OUTCOME UNCERTAIN: the approved access request may or may not have reached the "
                "server — it can commit before the checkpoint is saved, and no tool can read "
                "access_requests. It will NOT be retried automatically. Verify manually with IT "
                f"before any new attempt. Approved arguments: {approved}")
    if values.get("writes_done"):
        filed = "\n".join(f"  {result}" for result in values["writes_done"].values())
        return f"ALREADY FILED — do not file it again. Server result:\n{filed}"
    return "Nothing was filed in this run. Starting a new request is safe."


def classify(snapshot, thread: str) -> tuple[str, str]:
    """(state, next action) for --status — the same cases --resume acts on."""
    if snapshot.interrupts:
        return "awaiting approval", f"python supervisor.py --resume {thread}"
    if not snapshot.next:
        return f"finished — {outcome_of(snapshot.values)}", "nothing to do"
    if "write" in snapshot.next:
        return ("stopped — outcome uncertain (the approved write may have reached the server)",
                "verify manually with IT before any new attempt; do not resume or re-run it")
    if snapshot.values.get("writes_done"):
        return "stopped — already filed", "do not file it again"
    return "stopped — nothing filed", "starting a new request is safe"


class _InspectionModel(BaseChatModel):
    """Placeholder for reading checkpoints (--status, --resume): the graph is read, never run."""

    @property
    def _llm_type(self) -> str:
        return "inspection-only"

    def _generate(self, *args, **kwargs):
        raise RuntimeError("the inspection graph never calls a model")


def inspection_graph(saver):
    # LangGraph derives a snapshot's `next` and `interrupts` from the compiled graph, so the
    # checkpoint is read through the same graph shape — with tools and a model that never run.
    def refuse(**kwargs):
        raise RuntimeError("the inspection graph never calls a tool")

    names = sorted({t for spec in WORKER_SPECS.values() for t in spec["tools"]} | {WRITE_TOOL})
    tools = [StructuredTool.from_function(func=refuse, name=n, description="inspection only") for n in names]
    return build_graph(tools, model=_InspectionModel(), checkpointer=saver)


async def drive(app, config: dict, ask, inputs=None, pending=None) -> tuple[int, bool]:
    """Runs the thread until it finishes or pauses. Returns (exit code, end the session?)."""
    thread = config["configurable"]["thread_id"]
    try:
        result = {"__interrupt__": pending} if pending else await app.ainvoke(inputs, config)
        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            show_approval(payload, thread)
            try:
                decision = await ask_in_thread(ask, payload)
            except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                print(f"\nPaused. Resume with: python supervisor.py --resume {thread}")
                return 0, True
            result = await app.ainvoke(Command(resume=decision), config)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\nInterrupted. Thread {thread} stopped before finishing.")
        print(stopped_message(await app.aget_state(config)))
        return 130, True
    except Exception as exc:
        print(f"\nError: {type(exc).__name__}: {exc}")
        print(stopped_message(await app.aget_state(config)))
        print(f"Thread: {thread}  (details: python supervisor.py --status {thread})")
        return 1, False
    findings = result.get("findings", [])
    if findings:
        times = [f["seconds"] for f in findings]
        print(f"\nTeam: slowest specialist {max(times):.1f}s, sum {sum(times):.1f}s "
              f"(parallel wall time ~ the slowest)")
    print(f"\nNovaOps: {result.get('answer', '')}")
    print(f"\nOutcome: {outcome_of(result)}")
    return (1 if result.get("error") else 0), False


async def run_request(app, request: str, ask) -> tuple[int, bool]:
    thread = new_thread_id()
    while (await app.aget_state(thread_config(thread))).values:   # never reuse a saved thread
        thread = new_thread_id()
    print(f"Thread: {thread}\nYou: {request}\n")
    return await drive(app, thread_config(thread), ask, inputs=initial_state(request))


async def interactive(app, ask) -> int:
    print("Interactive session — every request runs on its own new thread.")
    while True:
        try:
            line = await ask_in_thread(lambda _: input(REQUEST_PROMPT), None)
        except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
            print("\nSession ended.")
            return 0
        request = line.strip()
        if not request:
            continue
        if request.lower() in ("exit", "quit"):
            print("Session ended.")
            return 0
        code, end_session = await run_request(app, request, ask)
        if end_session:   # an approval was paused (or Ctrl-C): never start another request on top
            return code


async def load_runtime(tools, model):
    """(tools, model) ready to use, or an exit code if configuration or a connection fails."""
    try:
        if tools is None or model is None:
            load_config()
            require_env()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return 2
    try:
        model = model if model is not None else get_model()
        tools = tools if tools is not None else await load_tools()
    except Exception as exc:
        while isinstance(exc, ExceptionGroup):   # the MCP client wraps transport errors in task groups
            exc = exc.exceptions[0]
        print(f"Could not reach the MCP server or the model: {type(exc).__name__}: {exc}")
        return 1
    trace("mcp", f"discovered tools: {', '.join(t.name for t in tools)}")
    return tools, model


async def show_status(saver, thread: str | None) -> int:
    graph = inspection_graph(saver)
    if thread is not None:
        snapshot = await graph.aget_state(thread_config(thread))
        if not snapshot.values:
            print(f"Unknown thread: {thread}")
            return 2
        state, action = classify(snapshot, thread)
        print(f"Thread:  {thread}\nRequest: {snapshot.values.get('request', '')}\nState:   {state}")
        if snapshot.interrupts:
            for item in snapshot.interrupts[0].value["requests"]:
                print(f"Proposed write: {item['tool']}")
                for key, value in item["args"].items():
                    print(f"    {key}: {value}")
        elif snapshot.next:
            print(stopped_message(snapshot))
        print(f"Next:    {action}")
        return 0
    threads = []
    async for item in saver.alist(None):
        configurable = item.config["configurable"]
        if configurable.get("checkpoint_ns", "") == "" and configurable["thread_id"] not in threads:
            threads.append(configurable["thread_id"])
    if not threads:
        print("No saved threads.")
        return 0
    print(f"{len(threads)} saved thread(s):")
    for thread_id in sorted(threads):
        snapshot = await graph.aget_state(thread_config(thread_id))
        state, action = classify(snapshot, thread_id)
        print(f"\n  {thread_id}  {state[:140]}")
        print(f"      request: {str(snapshot.values.get('request', ''))[:100]}")
        print(f"      next:    {action}")
    return 0


async def resume_thread(saver, thread: str, ask, tools, model) -> int:
    snapshot = await inspection_graph(saver).aget_state(thread_config(thread))
    if not snapshot.values:
        print(f"Unknown thread: {thread}")
        return 2
    if not snapshot.interrupts:
        if not snapshot.next:
            print(f"Thread {thread} already finished.\n\nNovaOps: {snapshot.values.get('answer', '')}")
            print(f"\nOutcome: {outcome_of(snapshot.values)}")
            return 0
        print(f"Thread {thread} stopped outside an approval pause and is not resumed automatically.")
        print(stopped_message(snapshot))
        return 3
    runtime = await load_runtime(tools, model)   # only a paused approval needs MCP and the model
    if isinstance(runtime, int):
        return runtime
    app = build_graph(runtime[0], model=runtime[1], checkpointer=saver)
    code, _ = await drive(app, thread_config(thread), ask, pending=(await app.aget_state(thread_config(thread))).interrupts)
    return code


async def cli(argv: list[str], *, tools=None, model=None, db_path=CHECKPOINT_DB, ask=ask_human) -> int:
    parsed = parse_args(argv)
    if parsed is None:
        print(f"Unknown or incomplete arguments: {' '.join(argv)}\n\n{USAGE}")
        return 2
    mode, value = parsed
    if mode == "help":
        print(USAGE)
        return 0
    db_path = Path(db_path)
    if mode in ("status", "resume"):
        if not db_path.exists():   # never create the checkpoint file just to look inside it
            print("No saved threads." if mode == "status" and value is None else f"Unknown thread: {value}")
            return 0 if mode == "status" and value is None else 2
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
            if mode == "status":
                return await show_status(saver, value)
            return await resume_thread(saver, value, ask, tools, model)

    runtime = await load_runtime(tools, model)
    if isinstance(runtime, int):
        return runtime
    tools, model = runtime
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        try:
            app = build_graph(tools, model=model, checkpointer=saver)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        if mode == "interactive":
            return await interactive(app, ask)
        code, _ = await run_request(app, value, ask)
        return code


def main() -> None:
    # Model text can hold characters a Windows console/pipe codec cannot encode; never crash on print.
    sys.stdout.reconfigure(errors="replace")
    try:
        code = asyncio.run(cli(sys.argv[1:]))
    except KeyboardInterrupt:   # Ctrl-C outside a run (e.g. while connecting); the saver is closed
        print("\nInterrupted.")
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
