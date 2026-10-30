"""Offline fakes for the supervisor tests (SPEC.md §12).

- `ScriptedModel` — a chat model that picks its reply by the role marker in the system
  prompt (planner / a specialist / act / respond) and records which tools were bound.
- `FakeServer` — async tools carrying the NovaOps server's tool names; records every call.
"""
import inspect
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Offline means offline: the local .env may enable LangSmith tracing, and load_dotenv never
# overrides a variable that is already set.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import supervisor  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langchain_core.tools import StructuredTool, ToolException  # noqa: E402
from pydantic import ConfigDict, Field  # noqa: E402

WRITE = "create_access_request"
WEBEX_ARGS = {"employee_id": "E010", "software": "Webex",
              "business_justification": "Customer-facing calls require Webex."}
WEBEX_REQUEST = "I'm E010 and Webex says I'm not licensed — can you help?"

_ids = itertools.count(1)


def text(content: str) -> AIMessage:
    return AIMessage(content=content)


def calls(*specs: tuple[str, dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": dict(args), "id": f"call_{next(_ids)}", "type": "tool_call"}
        for name, args in specs])


def cut_off(message: AIMessage) -> AIMessage:
    """The same reply, reported by the provider as stopped at the output-token limit."""
    return message.model_copy(update={"response_metadata": {"stopReason": "max_tokens"}})


def in_turn(*replies):
    """A reply function that returns the given replies in order, then repeats the last."""
    queue = list(replies)

    def reply(messages, bound):
        return queue.pop(0) if len(queue) > 1 else queue[0]
    return reply


def saw_tool_result(messages, prefix: str = "") -> bool:
    return any(isinstance(m, ToolMessage) and str(m.content).startswith(prefix) for m in messages)


def role_of(messages) -> str:
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    if system == supervisor.PLANNER_PROMPT:
        return "plan"
    for name, spec in supervisor.WORKER_SPECS.items():
        if system == spec["prompt"]:
            return name
    if system.startswith(supervisor.ACT_PROMPT):
        return "act"
    if system == supervisor.RESPOND_PROMPT:
        return "respond"
    raise AssertionError(f"unrecognised system prompt: {system[:80]!r}")


def default_act(messages, bound):
    if WRITE in bound:
        return calls((WRITE, WEBEX_ARGS))
    return text("Here is what I found; nothing was filed.")


DEFAULT_REPLIES: dict[str, Callable] = {
    "knowledge": lambda m, b: text("The remote work policy allows two remote days per week."),
    "records": lambda m, b: text("E010 Rachel Stein, CS Specialist. Webex: 42 of 40 seats, over_limit."),
    "eligibility": lambda m, b: text("CS Specialists may use Webex; the IT manager approves."),
    "act": default_act,
    "respond": lambda m, b: text("FINAL ANSWER: your Webex request has been handled."),
}


class ScriptedModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    replies: dict[str, Any] = Field(default_factory=dict)
    plans: list[Any] = Field(default_factory=list)   # planner outputs, consumed in order; last repeats
    calls: list[tuple[str, list[str]]] = Field(default_factory=list)  # (role, bound tool names)
    call_kwargs: list[tuple[str, dict]] = Field(default_factory=list)  # (role, other call kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("async only")

    def bind_tools(self, tools, **kwargs):
        return self.bind(bound_tools=[getattr(t, "name", None) or t["name"] for t in tools])

    async def _agenerate(self, messages, stop=None, run_manager=None, bound_tools=(), **kwargs):
        role = role_of(messages)
        self.calls.append((role, list(bound_tools)))
        self.call_kwargs.append((role, dict(kwargs)))
        reply = self.replies[role](messages, list(bound_tools))
        if inspect.isawaitable(reply):
            reply = await reply
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        assert include_raw, "the planner must request include_raw=True"

        async def plan(messages):
            assert role_of(messages) == "plan"
            self.calls.append(("plan", []))
            parsed = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
            if isinstance(parsed, Exception):
                raise parsed
            return {"raw": AIMessage(content="(scripted)"), "parsed": parsed,
                    "parsing_error": None if parsed is not None else ValueError("invalid plan")}

        return RunnableLambda(plan)

    def roles(self) -> list[str]:
        return [role for role, _ in self.calls]

    def bound_for(self, role: str) -> list[list[str]]:
        return [bound for r, bound in self.calls if r == role]


def make_model(workers=("records", "eligibility"), may_file=True, plans=None, **replies) -> ScriptedModel:
    plans = plans if plans is not None else [
        supervisor.Plan(workers=list(workers), may_file_request=may_file)]
    return ScriptedModel(replies={**DEFAULT_REPLIES, **replies}, plans=list(plans))


# Employees the fake server knows: id -> (full name, role). Filing needs an exact id.
# Two Mayas, like the real seed data (duplicate first names on purpose).
EMPLOYEES = {"E001": ("Maya Cohen", "Customer Success Manager"), "E010": ("Rachel Stein", "CS Specialist"),
             "E013": ("Emma Wilson", "COO"), "E017": ("Daniel Cohen", "Sales Operations Specialist"),
             "E020": ("Maya Rosen", "HR Coordinator")}
# Subscriptions the fake server knows: name -> (seat_limit, active_seats).
SUBSCRIPTIONS = {"MailGuard": (80, 70), "Miro": (25, 30), "Webex": (40, 42), "ZoomInfo": (15, 12)}


def email(full_name: str) -> str:
    return full_name.lower().replace(" ", ".") + "@novaops.example"


class FakeServer:
    """Matches names like the real server: case-insensitive substring, first alphabetically.

    write_mode: "ok" | "server_error" (MCP isError result) | "transport" (connection drops
    after the server received the call — the outcome is unknown to the client).
    lookup_mode: "ok" | "transport" (check_software_subscription raises).
    extra_systems: systems without a subscription that filing can still match.
    extra_employees: id -> (full name, role), added to EMPLOYEES."""

    def __init__(self, write_mode: str = "ok", lookup_mode: str = "ok", extra_systems=(), extra_employees=None):
        self.write_mode, self.lookup_mode = write_mode, lookup_mode
        self.employees = {**EMPLOYEES, **(extra_employees or {})}
        self.systems = sorted([*SUBSCRIPTIONS, *extra_systems])
        self.log: list[tuple[str, dict]] = []

    @property
    def writes(self) -> list[dict]:
        return [args for name, args in self.log if name == WRITE]

    def tools(self) -> list[StructuredTool]:
        log = self.log

        async def list_policies() -> str:
            log.append(("list_policies", {}))
            return json.dumps(["software_access_policy", "remote_work_policy"])

        async def get_policy(name: str) -> str:
            log.append(("get_policy", {"name": name}))
            return f"Policy {name}: managers approve SaaS access."

        async def search_knowledge_base(query: str) -> str:
            log.append(("search_knowledge_base", {"query": query}))
            return "KB: restart the client after a licence change."

        async def get_employee(query: str) -> list:
            log.append(("get_employee", {"query": query}))
            rows = [{"employee_id": eid, "full_name": name, "email": email(name), "role": role}
                    for eid, (name, role) in sorted(self.employees.items(), key=lambda e: e[1][0])
                    # like the server: exact id, exact email (any case), or name substring
                    if eid == query or query.lower() == email(name) or query.lower() in name.lower()]
            return [{"type": "text", "text": json.dumps(row)} for row in rows]

        async def check_software_subscription(software: str) -> list:
            log.append(("check_software_subscription", {"software": software}))
            if self.lookup_mode == "transport":
                raise ConnectionError("connection reset by peer")
            rows = [{"system_name": name, "vendor_name": name, "seat_limit": limit, "active_seats": active,
                     "status": "active", "seats_available": limit - active, "over_limit": active >= limit}
                    for name, (limit, active) in sorted(SUBSCRIPTIONS.items()) if software.lower() in name.lower()]
            return [{"type": "text", "text": json.dumps(row)} for row in rows]   # one MCP text block per row

        async def create_access_request(employee_id: str, software: str, business_justification: str) -> str:
            args = {"employee_id": employee_id, "software": software,
                    "business_justification": business_justification}
            log.append((WRITE, args))
            if self.write_mode == "server_error":
                raise ToolException("Employee 'E999' not found")
            if self.write_mode == "transport":
                raise ConnectionError("connection reset by peer")
            system = next((name for name in self.systems if software.lower() in name.lower()), None)
            if system is None:
                raise ToolException(f"No system matching {software!r}.")
            return json.dumps({"request_id": f"AR{100 + len(self.writes)}", "system_name": system,
                               "status": "pending_approval", **args})

        fns = [list_policies, get_policy, search_knowledge_base, get_employee,
               check_software_subscription, create_access_request]
        return [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                             description=fn.__name__.replace("_", " "),
                                             handle_tool_error=True)
                for fn in fns]
