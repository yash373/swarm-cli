
# Swarm CLI — Conversational Worker Orchestrator

A three-tier AI orchestration system that combines a powerful **Manager** model with a pool of lightweight **Worker** models to execute tasks in parallel, automatically selecting the optimal strategy for speed, accuracy, and efficiency.

Every conversation is automatically persisted to a project folder on your Desktop (or a custom location), creating a complete, searchable archive of your AI sessions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Files](#files)
  - [worker.py](#workerpy)
  - [manager.py](#managerpy)
  - [interface.py](#interfacepy)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Persistence](#project-persistence)
- [Usage Examples](#usage-examples)
  - [Layer 1: Worker](#layer-1-worker)
  - [Layer 2: Manager](#layer-2-manager)
  - [Layer 3: Interface](#layer-3-interface)
- [Strategies](#strategies)
- [CLI Commands](#cli-commands)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Best Practices](#best-practices)

---

## Overview

Swarm CLI is a conversational interface for local LLM orchestration via [Ollama](https://ollama.com/). It features:

- **Persistent REPL** with multi-turn conversation history
- **Auto-strategy selection** — the Manager intelligently routes tasks
- **Dynamic worker scaling** — spawn or prune workers at runtime
- **Parallel execution** — independent tasks run simultaneously
- **Ensemble reasoning** — multiple workers vote on hard problems
- **Map-Reduce** — large datasets processed in chunks
- **Split-and-Conquer** — complex tasks auto-decomposed into parallel sub-tasks
- **Tool-augmented workers** — each worker can execute Python, search the web, fetch URLs, and install packages
- **Automatic project archiving** — every conversation is saved to disk in real time

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (You)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  INTERFACE (interface.py)                                   │
│  ├─ Conversational REPL                                     │
│  ├─ History management                                      │
│  ├─ Auto-strategy router                                    │
│  ├─ Dynamic worker pool scaling                             │
│  ├─ Slash commands (/help, /workers, /strategy, etc.)       │
│  └─ ProjectStore — real-time disk persistence               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  MANAGER (manager.py)  ←  Gemma 4 (12B/27B)                │
│  ├─ Inherits all Worker capabilities                        │
│  ├─ parallel_respond()   — distribute independent queries   │
│  ├─ ensemble_respond()   — vote/synthesize for accuracy     │
│  ├─ map_reduce()         — chunk → process → aggregate      │
│  ├─ split_and_conquer()  — auto-decompose complex tasks     │
│  ├─ delegate()           — round-robin single task          │
│  ├─ broadcast()          — seed all workers with context    │
│  └─ ThreadPoolExecutor for concurrent execution             │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│  WORKER POOL (worker.py)  ←  Qwen 3 (4B/8B) x N            │
│  ├─ execute_python()     — real Python REPL (persistent)    │
│  ├─ install_package()    — PyPI package installation        │
│  ├─ search_web()         — web search (DuckDuckGo, etc.)    │
│  ├─ fetch_url()          — page scraping & text extraction  │
│  └─ Each worker maintains its own persistent Python globals │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

### worker.py

The foundational execution unit. Each `Worker` instance wraps an Ollama model and exposes four tool functions that the model can call autonomously:

| Tool | Purpose |
|------|---------|
| `execute_python` | Runs Python code in a persistent namespace (REPL-like state across calls) |
| `install_package` | Installs any PyPI package via `uv pip install` |
| `search_web` | Searches the web using multiple backends (DuckDuckGo, Brave, Bing, etc.) |
| `fetch_url` | Fetches and strips HTML to plain text (~8000 char limit) |

**Key features:**
- Configurable timeouts (default 30s, max 1800s)
- Persistent `__globals__` across `execute_python` calls within a conversation
- Graceful timeout handling via `SIGALRM`
- Streaming responses with `think=True` support

### manager.py

The orchestration layer. `Manager` inherits from `Worker`, so it retains all tool capabilities while adding parallel execution strategies via `ThreadPoolExecutor`.

**Key features:**
- **Thread-based concurrency** — optimal for I/O-bound LLM inference
- **Round-robin load balancing** across the worker pool
- **Thread-safe worker counter** via `threading.Lock`
- Dynamic worker pool resizing (spawn/prune)
- Result aggregation with `WorkerResult` dataclass

**Public API:**

| Method | Description |
|--------|-------------|
| `parallel_respond(queries)` | Distribute independent queries; results returned in input order |
| `ensemble_respond(query, n_workers, strategy)` | Same query to N workers; aggregate via synthesize / vote / first / all |
| `map_reduce(items, map_prompt, reduce_prompt, chunk_size)` | Classic map-reduce over a list of items |
| `split_and_conquer(complex_query, n_subtasks)` | Manager auto-decomposes task, workers execute in parallel, Manager synthesizes |
| `delegate(query, worker_index)` | Single task to one worker (round-robin or targeted) |
| `broadcast(query)` | Send same context/persona to every worker |
| `shutdown()` | Gracefully terminate the thread pool |

### interface.py

The user-facing conversational CLI. Wraps `Manager` with persistent history, auto-strategy routing, runtime controls, and automatic project archiving via `ProjectStore`.

**Key features:**
- **Auto-strategy mode** — Manager classifies each user message and picks the best execution pattern
- **Manual override** — force any strategy via `/strategy <name>`
- **Dynamic scaling** — resize worker pool mid-conversation via `/workers <n>`
- **Conversation history** — full transcript with strategy and worker metadata
- **Model hot-swapping** — change manager or worker models without restarting
- **Real-time persistence** — every turn is immediately written to disk
- **Project folders** — each session gets its own folder on your Desktop
- **Artifact saving** — save generated files directly into the project folder

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- `uv` (optional, for package installation tool)

### Install Dependencies

```bash
pip install ollama requests beautifulsoup4 duckduckgo_search
# or
uv pip install ollama requests beautifulsoup4 duckduckgo_search
```

### Pull Models

```bash
# Manager (reasoning / synthesis)
ollama pull gemma4:12b

# Workers (fast parallel execution)
ollama pull qwen3:4b

# Optional alternatives
ollama pull gemma4:27b   # stronger manager
ollama pull qwen3:8b     # stronger workers
```

---

## Quick Start

```bash
# Auto-strategy mode with a named project
python interface.py --manager gemma4:12b --worker qwen3:4b --workers 3 --project "PhysicsNotes"

# Heavy-duty configuration with custom save location
python interface.py \\
    --manager gemma4:27b \\
    --worker qwen3:8b \\
    --workers 6 \\
    --max-workers 12 \\
    --ctx 65536 \\
    --project "CodeReview" \\
    --projects-dir ~/Documents/AIChats
```

---

## Project Persistence

Every conversation is automatically saved to a project folder. This happens **in real time** — after every single turn, the full state is written to disk.

### Folder Structure

```
~/Desktop/SwarmProjects/                    # or your custom --projects-dir
├── 2026-08-17_013045_PhysicsNotes/
│   ├── manifest.json          # creation time, project name, platform info
│   ├── history.json           # structured turn-by-turn data (machine-readable)
│   ├── chat.md                # human-readable markdown transcript
│   ├── strategies.json        # usage analytics per strategy
│   ├── interface_state.json   # full runtime snapshot for potential resume
│   └── artifacts/             # files saved via /artifact command
│       └── quicksort.py
│       └── notes.txt
├── 2026-08-17_015030_Default/
│   ├── manifest.json
│   ├── history.json
│   ├── chat.md
│   └── ...
└── ...
```

### File Descriptions

| File | Purpose |
|------|---------|
| `manifest.json` | Session metadata: creation time, project name, OS platform, working directory |
| `history.json` | Structured array of every turn with role, content, strategy, worker count, and timestamp |
| `chat.md` | Clean Markdown transcript you can open in VS Code, Obsidian, or any text editor — updates live |
| `strategies.json` | Analytics object tracking how often each strategy was used |
| `interface_state.json` | Snapshot of runtime config (models, worker count, strategy mode) |
| `artifacts/` | Folder for files you explicitly save with the `/artifact` command |

### Cross-Platform Desktop Detection

The `ProjectStore` automatically finds your Desktop folder:

- **Windows**: `C:\\Users\\<You>\\Desktop\\SwarmProjects`
- **macOS**: `~/Desktop/SwarmProjects`
- **Linux**: `~/Desktop/SwarmProjects` (falls back to `~` if Desktop doesn't exist)

Override with `--projects-dir <path>`.

---

## Usage Examples

### Layer 1: Worker

Use `Worker` directly for simple, single-turn tool-augmented tasks:

```python
from worker import Worker

w = Worker(model="qwen3:4b", num_ctx=32768)

# The model can autonomously call tools
response = w.respond(
    "Search the web for the latest Mars rover news, then summarize it."
)
print(response)
```

### Layer 2: Manager

Use `Manager` for parallel execution and accuracy enhancement:

```python
from manager import Manager

with Manager(
    model="gemma4:12b",
    num_workers=4,
    num_ctx=32768,
) as mgr:
    # Override workers to use Qwen
    for worker in mgr.workers:
        worker.model = "qwen3:4b"

    # 1. Parallel independent tasks
    results = mgr.parallel_respond([
        "Explain quantum tunneling in 3 sentences.",
        "Write a Python one-liner to reverse a dictionary.",
        "What are the differences between TCP and UDP?",
    ])
    for r in results:
        print(f"Worker {r.worker_id}: {r.response[:200]}")

    # 2. Ensemble for accuracy (3 workers, Gemma synthesizes)
    answer = mgr.ensemble_respond(
        query="Solve: A train leaves A at 60 mph...",
        n_workers=3,
        strategy="synthesize",
    )
    print(answer)

    # 3. Map-Reduce over a large dataset
    docs = [f"Document {i} text..." for i in range(20)]
    summary = mgr.map_reduce(
        items=docs,
        map_prompt_template="Summarize these:\\n{item}",
        reduce_prompt="Combine into one executive summary.",
        chunk_size=5,
    )
    print(summary)

    # 4. Split-and-Conquer for complex design tasks
    design = mgr.split_and_conquer(
        complex_query="Design a REST API for a task app...",
        n_subtasks=4,
    )
    print(design)
```

### Layer 3: Interface

The full conversational experience with automatic project archiving:

```bash
$ python interface.py --manager gemma4:12b --worker qwen3:4b --workers 3 --project "ResearchSession"

[Project] Saved to: /Users/you/Desktop/SwarmProjects/2026-08-17_013045_ResearchSession
[System] Spawning Manager (gemma4:12b) with 3 workers (qwen3:4b)...

╔══════════════════════════════════════════════════════════════╗
║  SWARM CLI  —  Conversational Worker Orchestrator            ║
║  Manager: gemma4:12b                                         ║
║  Workers: qwen3:4b          x 3                              ║
║  Mode:    AUTO (manager decides strategy)                    ║
║  Project: /Users/you/Desktop/SwarmProjects/...               ║
╚══════════════════════════════════════════════════════════════╝
Type /help for commands. Start chatting below.

> What is the capital of France?
[Strategy] SINGLE | Workers: 1
Paris is the capital and most populous city of France...

> Calculate 128*512, explain quantum tunneling, and list 3 Python tricks.
[Strategy] PARALLEL | Workers: 3
[Part 1] 65,536
[Part 2] Quantum tunneling is a phenomenon where...
[Part 3] 1. Walrus operator := ...

> Solve this logic puzzle: [hard problem]
[Strategy] ENSEMBLE | Workers: 3
Synthesized Answer: The bird flies 360 miles...

> Design a REST API for a task app with endpoints, schema, errors, examples.
[Strategy] SPLIT | Workers: 4
Final Synthesized Answer:
## Endpoints
GET /tasks ...
POST /tasks ...
...

> /artifact api_spec.md "## REST API Spec\\n\\n### GET /tasks..."
Artifact saved:
  /Users/you/Desktop/SwarmProjects/.../artifacts/api_spec.md

> /workers 6
[System] Spawning 3 new worker(s)...
Worker pool resized to 6.

> /strategy ensemble
Strategy forced to: ENSEMBLE

> /strategy auto
Strategy set to AUTO (manager decides).

> /project
Current project folder:
  /Users/you/Desktop/SwarmProjects/2026-08-17_013045_ResearchSession

> /projects
Saved Projects:
----------------------------------------
  • 2026-08-17_013045_ResearchSession
    Name: ResearchSession
    Created: 2026-08-17T01:30:45
    Path: /Users/you/Desktop/SwarmProjects/...
  • 2026-08-16_223012_Default
    Name: Default
    Created: 2026-08-16T22:30:12
    Path: /Users/you/Desktop/SwarmProjects/...

> /quit
[System] Saving final state...
```

---

## Strategies

The Interface automatically selects a strategy based on query characteristics, or you can force one via `/strategy <name>`.

| Strategy | When Auto-Selected | Description |
|----------|-------------------|-------------|
| **single** | Simple Q&A, greetings, one-liners | One worker, one shot. Fastest. |
| **parallel** | Multiple distinct questions in one message | Splits into independent queries, runs simultaneously, synthesizes. |
| **ensemble** | Math, logic, code correctness, "best answer" requests | Same query to N workers; Gemma resolves contradictions or votes. |
| **split** | Complex multi-step tasks (design, planning, writing) | Gemma decomposes into sub-tasks; workers execute in parallel; Gemma assembles. |
| **mapreduce** | Large datasets/lists needing aggregation | Chunks data, maps processing across workers, reduces to final output. |
| **broadcast** | Context setting, persona changes, system instructions | Sends message to all workers simultaneously. |

### Strategy Override

```
> /strategy ensemble
Strategy forced to: ENSEMBLE

> What is 2^64?
[Strategy] ENSEMBLE | Workers: 3
Synthesized Answer: 18,446,744,073,709,551,616

> /strategy auto
Strategy set to AUTO (manager decides).
```

---

## CLI Commands

Type any command at the `>` prompt:

| Command | Description |
|---------|-------------|
| `/quit`, `/q` | Exit the interface (saves final state before exiting) |
| `/clear` | Clear in-memory history (disk files remain) |
| `/history` | Show conversation transcript |
| `/status` | Show current configuration |
| `/workers <n>` | Resize worker pool (respects min/max bounds) |
| `/strategy <s>` | Force strategy: `single`, `parallel`, `ensemble`, `split`, `mapreduce`, `broadcast`, or `auto` |
| `/model <name>` | Change manager model (re-initializes manager) |
| `/worker_model <name>` | Change model for all workers instantly |
| `/project` | Show the full path to the current session folder |
| `/projects` | List all saved conversation projects with metadata |
| `/save` | Force an immediate state snapshot to disk |
| `/artifact <filename> <content>` | Save a text file into the project's `artifacts/` folder |
| `/help`, `/h` | Show command reference |

---

## Configuration

### Command-Line Arguments

```bash
python interface.py [OPTIONS]

Options:
  --manager TEXT        Manager model name (default: gemma4:12b)
  --worker TEXT         Worker model name (default: qwen3:4b)
  --workers INTEGER     Initial worker count (default: 3)
  --max-workers INTEGER Max worker pool size (default: 8)
  --ctx INTEGER         Context window size (default: 32768)
  --manual              Disable auto-strategy (manual mode)
  --project TEXT        Project name for this session (default: auto-generated)
  --projects-dir PATH   Custom directory for project folders (default: Desktop/SwarmProjects)
```

### Programmatic Configuration

```python
from interface import Interface

iface = Interface(
    manager_model="gemma4:27b",
    worker_model="qwen3:8b",
    initial_workers=4,
    max_workers=12,
    min_workers=1,
    num_ctx=65536,
    auto_mode=True,
    project_name="MyResearch",
    projects_dir=Path.home() / "Documents" / "AIChats",
)

# Single turn
response = iface.chat("What is the meaning of life?")
print(response)

# Or run the REPL
iface.run()
```

---

## How It Works

### Auto-Strategy Routing

1. **Classification**: The Manager (Gemma) analyzes the user query against a system prompt that defines when to use each strategy.
2. **JSON Decision**: The Manager returns a JSON object: `{"strategy": "...", "reason": "...", "workers": N, "subtasks": []}`
3. **Worker Scaling**: The Interface resizes the worker pool if the recommended count differs from the current pool.
4. **Execution**: The appropriate method (`parallel_respond`, `ensemble_respond`, etc.) is called.
5. **Persistence**: The turn is immediately saved to `history.json`, `chat.md`, and `strategies.json`.
6. **History**: The turn is logged in memory with strategy and worker metadata for transparency.

### Why Threading?

The `Manager` uses `ThreadPoolExecutor` because LLM inference is **I/O-bound** (waiting on the Ollama API). Threads are ideal because:

- They share memory (cheap worker objects)
- The Python GIL is released during network I/O
- No process overhead or serialization costs
- Dynamic pool resizing is instantaneous

For CPU-bound tasks inside `execute_python`, the tool itself runs in the calling thread but is protected by timeouts.

### Persistent Python State

Each `Worker` maintains a `_python_globals` dict. Variables, imports, and functions defined in one `execute_python` call are available in the next call **within that same worker**. This makes it behave like a real REPL, not a fresh interpreter per call.

### ProjectStore Persistence

The `ProjectStore` class handles all disk I/O:

- **Auto-folder creation**: On initialization, it creates a timestamped folder on your Desktop (or custom path)
- **Real-time writes**: Every user and assistant turn triggers an immediate filesystem write — no batching, no delay
- **Dual formats**: `history.json` for structured data, `chat.md` for human readability
- **Artifact storage**: The `/artifact` command saves files into a dedicated `artifacts/` subfolder
- **Project listing**: The `/projects` command scans the base directory and reads each project's `manifest.json`

---

## Best Practices

1. **Manager vs. Worker Model Selection**
   - Use a **stronger, slower model** for the Manager (Gemma 4 12B/27B) — it plans, synthesizes, and routes.
   - Use a **faster, smaller model** for Workers (Qwen 3 4B/8B) — they execute in parallel.

2. **Worker Pool Sizing**
   - Start with 3 workers for general use.
   - Scale to 4-6 for heavy parallel or ensemble workloads.
   - Scale down to 1-2 for simple chat to save VRAM.

3. **Context Window**
   - Use `--ctx 65536` when processing large documents or long conversations.
   - Default 32768 is sufficient for most Q&A.

4. **Strategy Selection**
   - Trust auto-mode for 90% of queries.
   - Force `ensemble` for math, code, and factual verification.
   - Force `split` for creative writing, design, and research tasks.
   - Use `broadcast` to set a persona before a long session.

5. **Project Organization**
   - Always use `--project "Name"` so sessions are easy to find later.
   - Use `--projects-dir ~/Documents/AIChats` if you prefer a non-Desktop location.
   - Open `chat.md` in VS Code or Obsidian while you chat — it updates live.
   - Use `/artifact` to save generated code, notes, or data extracts into the project folder.

6. **Tool Usage**
   - Workers auto-install packages when needed — no manual intervention required.
   - Web search is automatically retried across multiple backends if one fails.
   - Always use `print()` in `execute_python` — return values are not auto-captured.

---

## License

MIT — use, modify, and distribute freely.
"""

with open("/mnt/agents/output/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"README.md updated successfully.")
print(f"Total characters: {len(readme_content)}")
