"""
NovaOps MCP server — policies + knowledge base + database (Exercise 3).

Exercise 1's server exposed two filesystem tools. This one keeps those and adds
four more, each backed by a completely different system:
  - search_knowledge_base()       -> vector search over the IT KB (rag.py)
  - get_employee()                -> a SQL lookup in the NovaOps database (db.py)
  - check_software_subscription() -> a SQL read: SaaS seat usage (db.py)
  - create_access_request()       -> a SQL WRITE: file an access request (db.py)

Six tools, three backends (plain files, a FAISS index, SQLite) — and to the agent
they are all just tools with a name and a schema. That uniformity is the whole
point of MCP. Notice that agent.py and client.py are UNCHANGED from Exercise 2:
only the server grew.

The last two tools set up Lesson 9: check_software_subscription surfaces the
"Webex is over its seat limit" signal, and create_access_request files the request
a human then approves — the approval gate Lesson 9's frameworks orchestrate.

Run:
  python server.py        # serves at http://127.0.0.1:9876/mcp
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as _FastMCPSettings

import db
import rag

POLICIES_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"

# FastMCP's settings model annotates `lifespan` with a forward reference to FastMCP
# itself, which pydantic-settings 2.15+ warns about on every construction. Resolving
# the reference once — which is exactly what the warning asks for — silences it without
# suppressing anything. Harmless to call on older versions.
_FastMCPSettings.model_rebuild()

mcp = FastMCP("novaops-assistant", host="127.0.0.1", port=9876)


@mcp.tool()
def list_policies() -> list[str]:
    """List the names of all available NovaOps company policies.

    Call this first to discover which policies exist before fetching one.
    """
    return sorted(p.stem for p in POLICIES_DIR.glob("*.md"))


@mcp.tool()
def get_policy(name: str) -> str:
    """Return the full text of one NovaOps policy.

    Args:
        name: A policy name as returned by list_policies, e.g. 'remote_work_policy'.
    """
    path = POLICIES_DIR / f"{name}.md"
    if not path.is_file():
        raise ValueError(f"No policy named {name!r}. Call list_policies for valid names.")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_knowledge_base(query: str) -> list[dict]:
    """Search the NovaOps IT knowledge base for articles relevant to a question.

    Use this for how-to and troubleshooting questions — password reset, VPN access,
    MFA, laptop provisioning, and the like. Returns the most relevant articles.

    Args:
        query: A natural-language description of the problem or question.
    """
    return rag.search_kb(query)


@mcp.tool()
def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by employee id, email, or (partial) name.

    Args:
        query: An employee id (e.g. 'E009'), an email, or a name to match.
    """
    return db.get_employee(query)


@mcp.tool()
def check_software_subscription(software: str) -> list[dict]:
    """Check a SaaS subscription's seat usage — is it at or over its seat limit?

    Use this when someone can't get access to a tool (e.g. 'Webex says I'm not
    licensed') to see whether seats are available or the subscription is full.

    Args:
        software: The software or vendor name, e.g. 'Webex' or 'Salesforce'.
    """
    return db.check_software_subscription(software)


@mcp.tool()
def create_access_request(
    employee_id: str, software: str, business_justification: str
) -> dict:
    """File an access request for an employee to a system (status: pending_approval).

    This WRITES to the database. Use it after confirming who the employee is
    (get_employee) and that access is warranted. The request is created pending a
    human approval decision — it does not grant access on its own.

    Args:
        employee_id: The employee id the access is for, e.g. 'E010'.
        software: The software or system to request, e.g. 'Webex'.
        business_justification: A short reason for the request.
    """
    return db.create_access_request(employee_id, software, business_justification)


if __name__ == "__main__":
    # Build the retrieval index up front so the first question isn't slow. The
    # server calls Bedrock (Titan) here — capability, including model calls, lives
    # on the server side, not in the agent.
    print("Building the IT KB index...")
    print(f"Indexed {rag.warm_index()} KB articles.")
    print("NovaOps MCP server → http://127.0.0.1:9876/mcp  (Ctrl-C to stop)")
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        print("Stopping...")
