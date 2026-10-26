"""
The database tool's engine (Exercise 3, server side).

A tiny SQLite layer over the NovaOps seed data. The MCP server calls these plain
functions; the agent never touches SQL. Three tools live here:
  - get_employee()                 read  — look up people
  - check_software_subscription()  read  — seat usage for a SaaS (the over-limit signal)
  - create_access_request()        WRITE — file a request, status pending_approval

The last one is the lab's first write tool. It's what Lesson 9 orchestrates: an
access request that a human then has to approve.

The DB is built in memory from data/database/{schema,seed}.sql on first use, so
there's no database file to create or clean up. Writes therefore live only for the
life of the server process and reset on restart — fine for a demo, not a real store.
"""

import functools
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "database"

# Only non-sensitive profile columns. A production version of this tool would also
# filter by the caller's user group (HR-only fields exist) — that guardrail is the
# subject of the homework, not this lab.
EMPLOYEE_FIELDS = (
    "employee_id, full_name, email, role, department_id, "
    "manager_id, location, employment_type, status"
)


@functools.lru_cache(maxsize=1)
def _conn() -> sqlite3.Connection:
    """Build the in-memory NovaOps DB once from the shipped schema + seed SQL.

    check_same_thread=False because FastMCP runs sync tools in a worker thread,
    which may not be the thread that first built this connection. One shared
    connection also means create_access_request's writes are visible to later
    reads. The agent calls tools one at a time, so no concurrent-write guarding
    is needed here — a real multi-user server would need it."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.executescript((DB_DIR / "seed.sql").read_text(encoding="utf-8"))
    return conn


def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by id, email, or (partial) name.

    Returns EVERY match — the seed data intentionally has duplicate first names
    (two Daniels, two Mayas, two Saras), so a name search can return more than one
    person, which the agent then has to disambiguate.
    """
    like = f"%{query}%"
    rows = _conn().execute(
        f"SELECT {EMPLOYEE_FIELDS} FROM employees "
        "WHERE employee_id = ? OR lower(email) = lower(?) OR lower(full_name) LIKE lower(?) "
        "ORDER BY full_name",
        (query, query, like),
    ).fetchall()
    return [dict(row) for row in rows]


def check_software_subscription(software: str) -> list[dict]:
    """Look up a SaaS subscription's seat usage by software or vendor name.

    Resolves the name against both the system catalog and the vendor list, so
    'Webex', 'webex', or a partial match all work. Returns seat_limit against the
    seats currently in use, plus a computed `over_limit` flag — the signal an
    access workflow reads to decide whether a new seat needs approval.
    """
    like = f"%{software}%"
    rows = _conn().execute(
        "SELECT sys.system_name, v.vendor_name, sub.seat_limit, sub.active_seats, "
        "       sub.status, sub.renewal_date "
        "FROM software_subscriptions sub "
        "JOIN systems sys ON sub.system_id = sys.system_id "
        "JOIN vendors  v  ON sub.vendor_id = v.vendor_id "
        "WHERE lower(sys.system_name) LIKE lower(?) OR lower(v.vendor_name) LIKE lower(?) "
        "ORDER BY sys.system_name",
        (like, like),
    ).fetchall()
    results = []
    for row in rows:
        sub = dict(row)
        sub["seats_available"] = sub["seat_limit"] - sub["active_seats"]
        sub["over_limit"] = sub["active_seats"] >= sub["seat_limit"]
        results.append(sub)
    return results


def create_access_request(
    employee_id: str,
    software: str,
    business_justification: str,
    access_level: str = "standard_user",
) -> dict:
    """File a new access request for an employee to a system; status pending_approval.

    The lab's one WRITE tool. It resolves the software name to a system id, inserts
    a row into access_requests, and returns the created record. It deliberately does
    NOT decide the request — approval is a separate human step, which is exactly the
    gate Lesson 9 builds on top of this tool.
    """
    conn = _conn()

    if conn.execute(
        "SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)
    ).fetchone() is None:
        raise ValueError(f"No employee {employee_id!r}. Look them up with get_employee first.")

    system = conn.execute(
        "SELECT system_id, system_name FROM systems "
        "WHERE lower(system_name) LIKE lower(?) ORDER BY system_name LIMIT 1",
        (f"%{software}%",),
    ).fetchone()
    if system is None:
        raise ValueError(f"No system matching {software!r}.")

    # Human-readable ids (AR005, AR006, ...) continuing the seed's scheme, so demo
    # output reads cleanly — a real system would use a uuid or DB sequence.
    last = conn.execute(
        "SELECT request_id FROM access_requests ORDER BY request_id DESC LIMIT 1"
    ).fetchone()
    next_num = int(last["request_id"][2:]) + 1 if last else 1
    request_id = f"AR{next_num:03d}"

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    conn.execute(
        "INSERT INTO access_requests "
        "(request_id, employee_id, system_id, access_level, business_justification, "
        " status, approver_id, created_at, decision_reason) "
        "VALUES (?, ?, ?, ?, ?, 'pending_approval', NULL, ?, NULL)",
        (request_id, employee_id, system["system_id"], access_level,
         business_justification, created_at),
    )
    conn.commit()
    return {
        "request_id": request_id,
        "employee_id": employee_id,
        "system_id": system["system_id"],
        "system_name": system["system_name"],
        "access_level": access_level,
        "status": "pending_approval",
        "created_at": created_at,
    }
