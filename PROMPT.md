<!-- ──────────────────────────────────────────────────────────────────────────
  HOW TO USE THIS FILE  (for you, the student — NOT part of the prompt)

  Everything inside these <!-- ... --> comments is for you. Your editor greys
  them out, so what's left in plain text is the prompt itself.

  The plain text below is a STUB, not a spec. It names WHAT to build; the HOW —
  who the specialists are and what each is allowed to touch, who decides which
  ones a given request needs, how they run relative to each other and how their
  work comes back together, and what makes a paused run resumable — is the
  exercise, and so are the TRADE-OFFS behind each choice. Don't hand this to
  your agent as-is: fold in your own decisions first, and say WHY (what each
  buys and what it costs). A blank where a decision should be is a wrong answer.

  Setup:
  - Rename CLAUDE.md-example -> CLAUDE.md so your agent reads the conventions.
  - server/ is PROVIDED and finished — Lesson 8's NovaOps MCP server, unchanged.
    Start it, connect to it over MCP, and leave it alone: don't edit it, and
    don't reimplement its tools inside your build.
  - Read what the server actually offers BEFORE you design anything. Its tool
    set is what decides how this work can be divided at all.
  - When it runs: server in one terminal, your build in another. Then invoke
    validate-lab to score it against the reference.

  ▢ YOU DECIDE  (the substance — deliberately absent from the plain prompt)
  - Who the specialists are, and what each one is allowed to touch. A specialist
    holding every tool is not a specialist. Overlap between two of them is real
    duplicated work you pay for — decide what it buys before you allow it.
  - Whether the team is fixed or chosen per request — and if chosen, chosen by
    what, and how you get a reliable machine-readable answer out of a model that
    has to pick from a known set.
  - How the specialists run relative to each other, and how their separate
    results become ONE body of evidence the final step can read. Both halves are
    named primitives in this framework — find them rather than hand-rolling.
  - What the human is shown at the gate, and what happens to the run when they
    say no: the model still owes the employee an answer, and a model that asked
    for a tool call is waiting for a result for it.
  - What makes the pause DURABLE. A run that suspends has to be resumable, and
    that is a property of how you compile the graph — not of the loop in your
    CLI. Ask what state a resume would have to read, and where it lives.
  - How you keep a sub-agent from wandering. Nothing stops a specialist reading
    every policy in the corpus and taking a minute to report — decide what does.
  - And the honest one: for THIS request, is a team of agents actually better
    than a single agent with all the tools? Say what it costs and what it buys.
─────────────────────────────────────────────────────────────────────────── -->
<!-- This build's decisions are recorded in SPEC.md — the authoritative implementation contract. -->

# Build: a NovaOps assistant that runs a team of agents, with a human on the write

An employee writes in:

> *"I'm E010 and Webex says I'm not licensed — can you help?"*

Handling it well takes several different kinds of lookup, and it ends in an action —
filing an access request — that a person should approve before it happens.

Build that assistant as a **supervisor over specialist sub-agents**, with a **human
approval gate** in front of the action.

The NovaOps tools already exist: `server/` is a finished MCP server. Start it in one
terminal and connect to it from your build.

<!-- Deliverable: supervisor.py at the ROOT of this repo, so that both

       python supervisor.py                      # defaults to the Webex case above
       python supervisor.py "some other request"

     run. validate-lab will ask you for the path to this folder. -->
