
# Swarm CLI — Conversational Worker Orchestrator

A three-tier AI orchestration system that combines a powerful **Manager** model with a pool of lightweight **Worker** models to execute tasks in parallel, automatically selecting the optimal strategy for speed, accuracy, and efficiency.

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
│  └─ Slash commands (/help, /workers, /strategy, etc.)       │
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

The user-facing conversational CLI. Wraps `Manager` with persistent history, auto-strategy routing, and runtime controls.

**Key features:**
- **Auto-strategy mode** — Manager classifies each user message and picks the best execution pattern
- **Manual override** — force any strategy via `/strategy <name>`
- **Dynamic scaling** — resize worker pool mid-conversation via `/workers <n>`
- **Conversation history** — full transcript with strategy and worker metadata
- **Model hot-swapping** — change manager or worker models without restarting

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
# Auto-strategy mode (recommended)
python interface.py --manager gemma4:12b --worker qwen3:4b --workers 3

# Heavy-duty configuration
python interface.py \\
    --manager gemma4:27b \\
    --worker qwen3:8b \\
    --workers 6 \\
    --max-workers 12 \\
    --ctx 65536
```

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

The full conversational experience:

```bash
$ python interface.py --manager gemma4:12b --worker qwen3:4b --workers 3

╔══════════════════════════════════════════════════════════════╗
║  SWARM CLI  —  Conversational Worker Orchestrator            ║
║  Manager: gemma4:12b                                         ║
║  Workers: qwen3:4b          x 3                              ║
║  Mode:    AUTO (manager decides strategy)                    ║
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

> /workers 6
[System] Spawning 3 new worker(s)...
Worker pool resized to 6.

> /strategy ensemble
Strategy forced to: ENSEMBLE

> /strategy auto
Strategy set to AUTO (manager decides).

> /quit
[System] Shutting down...
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
| `/quit`, `/q` | Exit the interface |
| `/clear` | Clear conversation history |
| `/history` | Show full transcript with strategy metadata |
| `/status` | Show current configuration (models, workers, strategy) |
| `/workers <n>` | Resize worker pool (respects min/max bounds) |
| `/strategy <s>` | Force strategy: `single`, `parallel`, `ensemble`, `split`, `mapreduce`, `broadcast`, or `auto` |
| `/model <name>` | Change manager model (re-initializes manager) |
| `/worker_model <name>` | Change model for all workers instantly |
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
5. **History**: The turn is logged with strategy and worker metadata for transparency.

### Why Threading?

The `Manager` uses `ThreadPoolExecutor` because LLM inference is **I/O-bound** (waiting on the Ollama API). Threads are ideal because:

- They share memory (cheap worker objects)
- The Python GIL is released during network I/O
- No process overhead or serialization costs
- Dynamic pool resizing is instantaneous

For CPU-bound tasks inside `execute_python`, the tool itself runs in the calling thread but is protected by timeouts.

### Persistent Python State

Each `Worker` maintains a `_python_globals` dict. Variables, imports, and functions defined in one `execute_python` call are available in the next call **within that same worker**. This makes it behave like a real REPL, not a fresh interpreter per call.

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

5. **Tool Usage**
   - Workers auto-install packages when needed — no manual intervention required.
   - Web search is automatically retried across multiple backends if one fails.
   - Always use `print()` in `execute_python` — return values are not auto-captured.

---

## License

MIT — use, modify, and distribute freely.
"""

with open("/mnt/agents/output/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("README.md written successfully.")
print(f"Total characters: {len(readme_content)}")
