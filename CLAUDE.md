# CLAUDE.md — build conventions

- `SPEC.md` is the authoritative implementation contract for this build (architecture, state, nodes, approval, persistence, recovery, tests). Read it before changing `supervisor.py` or its tests; if code and `SPEC.md` disagree, fix one of them — never leave both.
- `server/` is a fixed dependency: start it and call its tools over MCP. Never edit it, never import its functions directly, and never hard-code anything about a tool's logic — discover the tools at runtime. The current `server/` — including the owner's local change in `server/server.py` — is that fixed dependency: never modify or revert it.
- Build on LangChain / LangGraph (`langchain`, `langgraph`, `langchain-mcp-adapters`), and let the adapter turn the discovered MCP tools into framework tools rather than hand-writing schemas.
- Keep the MCP SDK pinned to `mcp<2` — `langchain-mcp-adapters` doesn't support the 2.0 SDK yet, and an unpinned `pip install` will silently pull 2.x back in and break every import.
- Load all configuration from the environment via `load_dotenv(find_dotenv())`; never hard-code a value. If a required variable is missing, STOP and ask rather than guess. Required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID`, `MCP_SERVER_URL`. (`.env.example` is tracked and holds placeholder values only; copy it to `.env` and fill in real values. `.env` is gitignored — never commit it or print its values.)
- Reach Bedrock through the framework's chat-model class, and keep that construction in one small function so the provider stays swappable.
- Comment only non-obvious AI/SDK decisions; leave self-evident code uncommented.
