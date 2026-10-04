# agent-harness

A small **agent harness** built on **LangChain + LangGraph** with a **React web UI** and
**PostgreSQL** chat history. It chats through three backends:

- **Claude** via the Anthropic API: a LangChain tool-calling agent compiled as a LangGraph graph.
- **GitHub Copilot** via the official [Copilot SDK](https://github.blog/changelog/2026-04-02-copilot-sdk-in-public-preview/)
  (public preview): Copilot's own agent, wrapped as a LangGraph node.
- **A local LLM** via [Ollama](https://ollama.com): same LangChain tool-calling agent as Claude,
  just pointed at `ChatOllama` instead of `ChatAnthropic`. No API key, no usage billed; quality
  depends on the model you pull.

All three get tools from **MCP servers**, for example the Confluence RAG from
[local-confluence-RAG](https://github.com/tealtadpole/local-confluence-RAG).

```
 Browser (React + Vite)                         FastAPI (agent-harness serve)
 ┌──────────────────────────┐  HTTP + SSE  ┌──────────────────────────────────────────────┐
 │ sessions sidebar         │◄────────────►│ /api/sessions …   TurnRunner (background task)│
 │ chat, streaming, tools   │              │        │                                     │
 └──────────────────────────┘              │        ▼                                     │
                                           │  LangGraph graph per provider                │
                                           │   ├─ claude:  create_agent(ChatAnthropic) ───┼──► Anthropic API
                                           │   ├─ copilot: node → Copilot SDK session ────┼──► GitHub Copilot
                                           │   └─ llama:   create_agent(ChatOllama) ──────┼──► local Ollama server
                                           │        │ tools                               │
                                           │        ▼                                     │
                                           │  MCP servers (stdio) e.g. confluence-rag     │
                                           └──────────┬───────────────────────────────────┘
                                                      ▼
                                   PostgreSQL: harness_sessions, harness_messages (UI history)
                                               + LangGraph checkpoint tables (agent memory)
```

## Requirements

| What | Details |
|---|---|
| **OS** | Linux or macOS (Windows via WSL). |
| **Python** | 3.11+ (with `python3-venv` on Debian/Ubuntu). |
| **Node.js** | 20.19+ or 22.12+ to build the web UI (only needed for building and UI development). |
| **PostgreSQL** | 14+. The included `docker-compose.yml` runs PostgreSQL 17, so you need Docker with Compose. Your user must be able to run `docker` (e.g. be in the `docker` group) or use `sudo docker compose`. Any existing Postgres works too: set `database.url`. |
| **Claude** | An Anthropic API key from [console.anthropic.com](https://console.anthropic.com) in `ANTHROPIC_API_KEY`. Usage is billed per token. |
| **Copilot** (optional) | A GitHub account **with a Copilot plan that allows SDK use**. Each prompt uses your Copilot premium-request quota. On first use the SDK downloads the Copilot runtime (~135 MB) into `~/.cache/github-copilot-sdk`. |
| **Local LLM** (optional) | [Ollama](https://ollama.com) installed and running (`ollama serve`), with at least one model pulled (`ollama pull llama3.2:3b`). |
| **MCP tools** (optional) | e.g. [local-confluence-RAG](https://github.com/tealtadpole/local-confluence-RAG) installed and synced. |

## Installation

```bash
git clone https://github.com/tealtadpole/agent-harness.git
cd agent-harness

# 1. Database
cp .env.example .env                 # set POSTGRES_PASSWORD
docker compose up -d                 # PostgreSQL on 127.0.0.1:5432

# 2. Backend
python3 -m venv .venv
.venv/bin/pip install -e .           # add ".[dev]" for tests

# 3. Web UI
cd frontend && npm install && npm run build && cd ..

# 4. Configure
cp config.example.toml config.toml   # set database.url password + MCP server paths
export ANTHROPIC_API_KEY='sk-ant-...'
export CONFLUENCE_PAT='...'          # passed through to the Confluence MCP server

.venv/bin/agent-harness check        # database, MCP tools, provider status
.venv/bin/agent-harness serve        # → http://127.0.0.1:8000
```

`check` prints one line per component, for example:

```
Database: connected
MCP ok  confluence: search_confluence, get_confluence_page
ok  Claude (Anthropic API): ready models: claude-opus-5-5, claude-sonnet-5-5, claude-haiku-4-5
--  GitHub Copilot: Not signed in to GitHub. Run `gh auth login` (GitHub CLI) or set $COPILOT_GITHUB_TOKEN.
ok  Local LLM (Ollama): ready models: llama3.2:3b
```

The tables are created automatically on first start.

### UI development

```bash
.venv/bin/agent-harness serve                  # API on :8000
cd frontend && npm run dev                     # UI with hot reload on http://localhost:5173 (proxies /api)
```

## Using it

- **New chat:** pick a provider and model in the sidebar, then type. The first message becomes the
  chat title. Each chat keeps the provider and model it started with.
- **History:** every chat is listed in the sidebar (most recent first) and stored in Postgres.
  Rename with ✎, delete with × (this also deletes the agent's memory for that chat).
- **Tool calls** show as collapsible cards with their input and output.
- Replies run **in the background on the server**: closing or reloading the tab doesn't lose them.
  When you reopen the chat, the finished answer is there.
- Links like `http://127.0.0.1:8000/#/s/<chat-id>` open a specific chat.

## Providers

### Claude (Anthropic API)

A standard LangChain agent (`langchain.agents.create_agent`) with `ChatAnthropic`, the MCP tools,
and the LangGraph Postgres checkpointer. Settings in `[claude]`:

- `effort`: how much thinking to spend (`low`…`max`). Claude Opus 5.5 always thinks; effort is
  the control. Not sent for Haiku 4.5, which doesn't support it.
- `refusal_fallback`: on Opus 5.5 / Sonnet 5.5, if a safety classifier declines a request, the
  API retries it on a fallback model in the same call (server-side fallback beta).
- `base_url`: send requests through an Anthropic-compatible gateway instead.

### GitHub Copilot (Copilot SDK)

The Copilot SDK runs Copilot's **own agent**, so this provider is a one-node LangGraph graph that
forwards the message to a Copilot session and streams its tokens and tool calls back.
LangGraph still checkpoints the conversation.

**Locked down to MCP tools only.** Copilot normally has shell, file-editing and web tools. Here
they're disabled twice over: an `available_tools` allow-list containing only your MCP tools,
and a permission handler that rejects every other request. The SDK is also told not to load
your personal Copilot config, skills or custom instructions.

Sign-in: by default the SDK reuses your existing GitHub CLI login (`gh auth login`). To use a
specific token, set `COPILOT_GITHUB_TOKEN`. If `check` says *"Copilot refused the request (403)"*,
the account is signed in but its Copilot plan doesn't include this kind of access.

> GitHub Models (the old free OpenAI-compatible endpoint) was
> [retired on 2026-07-30](https://github.blog/changelog/2026-07-01-github-models-is-being-fully-retired-on-july-30-2026/),
> so the Copilot SDK is the official way to use Copilot from your own app. Unofficial
> "Copilot API proxies" are deliberately not supported.

### Local LLM (Ollama)

A standard LangChain agent (`langchain.agents.create_agent`), identical in shape to the Claude
provider, but with `ChatOllama` talking to a local [Ollama](https://ollama.com) server instead of
the Anthropic API. Settings in `[llama]`:

- `base_url`: where Ollama is listening (default `http://127.0.0.1:11434`).
- `models` / `default_model`: model tags you've already pulled, e.g. `ollama pull llama3.2:3b`.
- `num_ctx`: context window override; `0` leaves Ollama's per-model default.

No API key and nothing billed. `check` calls Ollama's `/api/tags` endpoint to confirm the server
is reachable and the configured models are actually pulled; if not, it tells you which
`ollama pull` to run.

## MCP tools

Any stdio MCP server can be added under `[mcp.servers.<name>]`:

```toml
[mcp.servers.confluence]
command = "/abs/path/local-confluence-RAG/.venv/bin/confluence-rag"
args = ["--config", "/abs/path/local-confluence-RAG/config.toml", "serve"]
env = { CONFLUENCE_PAT = "${CONFLUENCE_PAT}" }   # ${VAR} expands from the harness environment
tools = ["search_confluence", "get_confluence_page"]   # optional allow-list
```

Each server is started **once** and kept running, rather than once per tool call, because a RAG
server would otherwise reload its embedding model on every search. Copilot starts its own copy
of each server. A server that fails to start is reported in `check` and the UI, and the rest
keeps working.

## Database

| Table | Contents |
|---|---|
| `harness_sessions` | One row per chat: title, provider, model, timestamps. |
| `harness_messages` | What the UI shows: user messages, assistant replies, tool calls (input + output, capped at 20 KB), errors. |
| `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` | LangGraph's checkpointer: the full agent state per chat (`thread_id` = chat id), so the model remembers earlier turns. |

## Security notes

- **No login.** Anyone who can reach the server can use your API key, your Copilot quota and
  your MCP tools. It listens on `127.0.0.1` only by default; keep it that way, or put it behind
  an authenticating proxy.
- **Browser-attack protection:** requests whose `Host` header (DNS rebinding) or `Origin`
  header (cross-site requests) isn't in `server.allowed_hosts` are rejected, so a malicious
  web page can't drive your local agent through your browser.
- **Prompt injection:** tool results (e.g. wiki pages) can contain instructions. The system
  prompt tells the model to treat them as data, the tool allow-lists keep the agent to read-only
  tools, and Copilot has no shell or file access here.
- **Secrets** come only from environment variables. `config.toml` and `.env` are gitignored.
- Docker's Postgres port is bound to `127.0.0.1`, and Compose refuses to start until
  `POSTGRES_PASSWORD` is set in `.env`.

## Limitations

- Single-user: no accounts or per-user permissions.
- No "stop generating" button yet.
- The Copilot SDK is in public preview; its API may change.
- Chats keep the model they started with.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                      # starts a throwaway PostgreSQL (pgserver); no Docker or API keys needed
cd frontend && npm run typecheck
```

The tests run the real FastAPI app, LangGraph graphs, Postgres checkpointer and a real stdio MCP
server, with a scripted model in place of Claude and a fake Copilot runtime.

```
src/agent_harness/
  config.py            config.toml loading/validation
  db.py                Postgres pool, schema, sessions/messages
  mcp_tools.py         long-lived MCP server sessions → LangChain tools
  providers/claude.py  ChatAnthropic + create_agent
  providers/copilot.py Copilot SDK runtime, lockdown, LangGraph node
  providers/llama.py   ChatOllama + create_agent (local LLM via Ollama)
  runner.py            one chat turn: stream events, save history (background task)
  api.py               FastAPI routes, SSE, static UI
  cli.py               serve / check
frontend/src/          React UI (App, Sidebar, MessageList, Composer)
```
