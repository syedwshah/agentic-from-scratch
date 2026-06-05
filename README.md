# agentic-from-scratch
---

A from-scratch OpenAI tool-calling agent built around a **reducer-style
state machine**. No `langchain`, no `gradio`, no `streamlit` — the kind
of thing you'd build in a live AI Engineer interview.

## Branches

| Branch | Purpose |
| ------ | ------- |
| **`master`** | Starter project — work through the checkpoints below. |
| **`solution`** | Completed reference implementation (agent loop, server, streaming). |

```bash
git checkout master     # back to the exercise (TODOs)
git checkout solution   # this branch — finished code
```

## The problem

Build an agent that satisfies user prompts by **chaining tool calls**.
The "agent" is just three pieces:

| Piece                      | Role                                         |
| -------------------------- | -------------------------------------------- |
| **System prompt**          | The *spec* of the state machine.             |
| **`TOOL_REGISTRY` dict**   | The *reducer table* — a map of name → fn.    |
| **The agent loop**         | The *dispatcher* — runs one step at a time.  |

Each tool is a small reducer function: given some arguments, it returns
structured JSON that gets appended back to the conversation as new
context. The LLM looks at the updated state and decides the next action
(call another tool, or stop and answer).

### Motivating query

> *"What is the word for today's day of the week, in Spanish?"*

A single round trip can't answer this — the model has no clock and no
dictionary. With our reducer table it becomes a 3-step state machine:

```
user prompt
   │
   ▼
[ACTION] get_nanotime()                       ─► {nanotime: 1_748_887_xxx_xxxxxxxxx}
   │
   ▼
[ACTION] day_of_week_from_nanotime(nanotime)  ─► {day_of_week: "Tuesday"}
   │
   ▼
[ACTION] translate(text="Tuesday",
                   target_language="spanish") ─► {translated: "martes"}
   │
   ▼
"martes."
```

Other queries route through the same machine:

- *"What time is it in nanoseconds?"* → 1 call (`get_nanotime`).
- *"How do you say goodbye in French?"* → 1 call (`translate`).
- *"What day is it today, in German?"* → 3 calls (time → day → translate).
- *"What is unobtainium in French?"* → 1 call, tool returns an `error`
  payload, the model gracefully says it doesn't know.

The "agentic" magic is just `while tool_calls: dispatch; append; ask
again`. Everything else is glue.

---

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  main.py    │────────▶│  agent.py   │────────▶│ OpenAI API  │
│  (CLI loop) │         │ (agent loop)│◀────────│             │
└─────────────┘         └──────┬──────┘         └─────────────┘
                               │ httpx (HTTP)
                               ▼
                        ┌─────────────┐
                        │  server.py  │  ← FastAPI + uvicorn
                        │  /tools/*   │
                        └──────┬──────┘
                               ▼
                        ┌─────────────┐
                        │  tools.py   │  ← reducer table + JSON schemas
                        └─────────────┘
```

The LLM **never** runs code. It returns a `tool_calls` payload that
*describes* the call it wants. Our agent POSTs to our own FastAPI server,
gets the result, appends it to the conversation, and loops.

---

## 1. Prerequisites

- macOS / Linux
- `uv` (`brew install uv` on macOS)
- Python 3.13 (uv will install it if missing)
- An OpenAI API key

## 2. Setup

```bash
cd ~/Documents/projects/agentic-from-scratch

# uv already created .venv and installed deps when the project was scaffolded.
# If you ever pull this fresh or wipe .venv, run:
uv sync

# Make your .env from the example and fill in your key:
cp .env.example .env
# then edit .env and put your real OPENAI_API_KEY in
```

> **Why `uv`?** It replaces `python -m venv`, `pip`, `pip-tools`, and
> `pip install -r requirements.txt` with a single fast tool. `uv sync`
> reads `pyproject.toml` + `uv.lock` and gives you a reproducible
> `.venv/`. `uv run <cmd>` runs `<cmd>` inside that venv without you
> having to activate it.

If you'd rather see what's happening:

```bash
source .venv/bin/activate     # classic venv activation
python -V                     # should print 3.13.x
which python                  # should point inside .venv
deactivate                    # exit the venv
```

## 3. Running the project

You'll need **two terminals**:

**Terminal 1 — the tool server (FastAPI + uvicorn):**
```bash
uv run uvicorn server:app --reload --port 8000
```
- `server:app` means *"import `app` from `server.py`"*.
- `--reload` watches files and hot-restarts uvicorn on changes.
- Visit http://localhost:8000/docs for an auto-generated Swagger UI.

**Terminal 2 — the chat REPL:**
```bash
uv run python main.py
```

Until you implement the loop you'll see `NotImplementedError` — that's
expected. Use it as your TODO list.

## 4. Tests

```bash
uv run pytest -q
```

---

## 5. Live-coding checkpoints

Work through these in order. Each step is small, runs in seconds, and
leaves you with a thing you can actually exercise.

### ✅ Checkpoint 0 — sanity
- `uv run uvicorn server:app --port 8000` boots without error.
- `curl localhost:8000/healthz` returns `{"status":"ok"}`.
- `curl localhost:8000/tools` returns the registry list with the one
  worked schema (`get_nanotime`).
- `uv run pytest -q` shows 2 passing tests.

### 🟡 Checkpoint 1 — fill in the reducer table (`tools.py`)
1. Implement `day_of_week_from_nanotime(nanotime)` — convert ns → seconds,
   build a UTC datetime, return `{"day_of_week": "<name>"}`.
2. Implement `translate(text, target_language)` — normalize both inputs,
   look up `_TRANSLATIONS[lang][word]`, return errors **as data** (not
   exceptions) so the model can recover.
3. Add the two missing schemas to `TOOL_SCHEMAS` so the LLM can call them.
4. Uncomment the TODO tests in `tests/test_tools.py` and make them pass.

### 🟡 Checkpoint 2 — implement the server endpoint (`server.py`)
1. Finish `POST /tools/{name}` per the docstring (map exceptions to HTTP
   status codes).
2. Verify each transition with curl:
   ```bash
   curl -s -XPOST localhost:8000/tools/get_nanotime              -H 'content-type: application/json' -d '{}'
   curl -s -XPOST localhost:8000/tools/day_of_week_from_nanotime -H 'content-type: application/json' -d '{"nanotime": 1748887200000000000}'
   curl -s -XPOST localhost:8000/tools/translate                 -H 'content-type: application/json' -d '{"text":"Tuesday","target_language":"spanish"}'
   ```

### 🔴 Checkpoint 3 — implement the agent loop (`agent.py`)
The big one. Implement in this order:
1. `_call_model()` — one OpenAI call, normalize the assistant message
   into a re-sendable dict (including `tool_calls` if present).
2. `_execute_tool_call()` — JSON-decode `arguments`, POST to the server,
   return a `role="tool"` message with `tool_call_id` set.
3. `chat()` — the loop itself. Append user message, then repeatedly:
   call the model, append its reply, execute every tool call, append
   each result, repeat — until the model returns a message with no
   `tool_calls`. Cap with `self.max_steps`.

Then run `uv run python main.py` and ask:
- *"What time is it in nanoseconds?"*                       → 1 call
- *"How do you say `thanks` in German?"*                    → 1 call
- *"What is today's day of the week, in Spanish?"*          → 3 calls
- *"Same question but in French and German, please."*       → up to 5 calls

Watch the server logs in Terminal 1 — you should see one POST per tool
call, in the order the agent decided to make them.

### 🟢 Stretch goals (pick any)
- **More tools**: `format_date(nanotime, fmt)`, `add_phrase(language,
  english, translation)` (mutates `_TRANSLATIONS` at runtime so the
  agent can teach itself a word and reuse it next turn).
- **Streaming**: switch to `stream=True` and print tokens as they arrive.
- **Parallel tools**: execute multiple `tool_calls` in one assistant
  turn concurrently with `concurrent.futures.ThreadPoolExecutor`.
- **Trace log**: print every model call + tool call as JSON so you can
  replay/debug a conversation.
- **Async end-to-end**: `AsyncOpenAI` + `httpx.AsyncClient`, expose the
  agent itself as a FastAPI route at `POST /chat`.
- **Real translation tool**: replace the dictionary with a call to an
  actual translation API (kept behind the same `translate` schema —
  the rest of the system doesn't change).

---

## 6. Why this design (interview talking points)

- **Tools = reducer functions, registry = reducer table, loop =
  dispatcher.** You're really just running a tiny event-sourced
  state machine where the LLM is the policy.
- **The model is a planner, not an executor.** Tool calls are
  *requests* the model makes; the trust boundary is in our code. The
  HTTP hop makes that boundary explicit and lets us reuse the same
  tools from multiple agents or processes.
- **Schemas live next to implementations.** `tools.py` owns the
  Python callable *and* the JSON schema the model sees, so they
  can't drift apart.
- **`max_steps` guards against runaway loops.** Cheap, essential,
  and the first thing a reviewer will look for.
- **Errors are data.** When a tool returns `{"error": ...}` instead of
  raising, the model can read it on its next turn and either retry or
  admit a limitation — the conversation never crashes.
