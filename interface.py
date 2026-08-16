# interface.py
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from manager import Manager, WorkerResult


# ------------------------------------------------------------------------- #
# Project Persistence Layer
# ------------------------------------------------------------------------- #

class ProjectStore:
    """
    Handles filesystem persistence for every conversation.
    Structure:
        ~/Desktop/SwarmProjects/
        ├── 2026-08-17_013045_Untitled/
        │   ├── manifest.json
        │   ├── history.json
        │   ├── chat.md
        │   ├── strategies.json
        │   └── artifacts/
        └── ...
    """

    def __init__(self, project_name: Optional[str] = None, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or self._default_projects_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.project_name = project_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Sanitize folder name
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", self.project_name).strip() or "Untitled"
        self.project_dir = self.base_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_name}"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Subdirs
        self.artifacts_dir = self.project_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.manifest_path = self.project_dir / "manifest.json"
        self.history_path = self.project_dir / "history.json"
        self.chat_md_path = self.project_dir / "chat.md"
        self.strategies_path = self.project_dir / "strategies.json"

        # Initialize files
        self._init_manifest()
        self._init_chat_md()

    @staticmethod
    def _default_projects_dir() -> Path:
        """Cross-platform Desktop detection."""
        home = Path.home()
        # Windows
        if sys.platform == "win32":
            desktop = home / "Desktop"
        # macOS
        elif sys.platform == "darwin":
            desktop = home / "Desktop"
        # Linux & others
        else:
            desktop = home / "Desktop"
            if not desktop.exists():
                desktop = home  # fallback
        return desktop / "SwarmProjects"

    def _init_manifest(self):
        manifest = {
            "created_at": datetime.now().isoformat(),
            "project_name": self.project_name,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _init_chat_md(self):
        header = textwrap.dedent(f"""\
        # {self.project_name}

        **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Location:** `{self.project_dir}`

        ---
        """)
        self.chat_md_path.write_text(header, encoding="utf-8")

    def save_turn(self, turn: "Turn", strategy: str, workers_used: int):
        """Append a single turn to all persistence files."""

        # 1. history.json — structured, machine-readable
        history = []
        if self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        history.append({
            "role": turn.role,
            "content": turn.content,
            "strategy": strategy if turn.role == "assistant" else "",
            "workers_used": workers_used if turn.role == "assistant" else 0,
            "timestamp": turn.timestamp,
        })
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        # 2. chat.md — human-readable transcript
        md_line = f"\n## Turn {len(history)}\n\n"
        md_line += f"**Role:** {turn.role}  \n"
        if turn.role == "assistant":
            md_line += f"**Strategy:** `{strategy}` | **Workers:** {workers_used}  \n"
        md_line += f"**Time:** {turn.timestamp}\n\n"
        md_line += f"{turn.content}\n\n---\n"
        with self.chat_md_path.open("a", encoding="utf-8") as f:
            f.write(md_line)

        # 3. strategies.json — analytics
        strategies = {}
        if self.strategies_path.exists():
            try:
                strategies = json.loads(self.strategies_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                strategies = {}
        strategies[strategy] = strategies.get(strategy, 0) + 1
        self.strategies_path.write_text(json.dumps(strategies, indent=2), encoding="utf-8")

    def save_artifact(self, filename: str, content: str) -> Path:
        """Save a generated artifact (code, data, etc.) into the project."""
        path = self.artifacts_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def save_state(self, interface_state: dict):
        """Save full interface state for resume capability."""
        state_path = self.project_dir / "interface_state.json"
        state_path.write_text(json.dumps(interface_state, indent=2, default=str), encoding="utf-8")

    @classmethod
    def list_projects(cls, base_dir: Optional[Path] = None) -> List[dict]:
        """List all saved projects with metadata."""
        base = base_dir or cls._default_projects_dir()
        if not base.exists():
            return []
        projects = []
        for folder in sorted(base.iterdir()):
            if folder.is_dir():
                manifest_file = folder / "manifest.json"
                manifest = {}
                if manifest_file.exists():
                    try:
                        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                projects.append({
                    "folder": folder.name,
                    "path": str(folder),
                    "name": manifest.get("project_name", folder.name),
                    "created": manifest.get("created_at", "Unknown"),
                })
        return projects

    @classmethod
    def load_history(cls, project_path: Path) -> List[dict]:
        """Load history.json from a project folder."""
        hist_file = project_path / "history.json"
        if hist_file.exists():
            return json.loads(hist_file.read_text(encoding="utf-8"))
        return []

    def __repr__(self):
        return f"ProjectStore({self.project_dir})"


# ------------------------------------------------------------------------- #
# Data models
# ------------------------------------------------------------------------- #

@dataclass
class Turn:
    role: str          # "user" | "assistant" | "system"
    content: str
    strategy: str = "single"
    workers_used: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ------------------------------------------------------------------------- #
# Interface
# ------------------------------------------------------------------------- #

class Interface:
    STRATEGIES = {
        "single": "One worker, one shot. Fastest for simple Q&A.",
        "parallel": "Multiple independent queries answered simultaneously.",
        "ensemble": "Same query to N workers, synthesized for accuracy.",
        "split": "Auto-decompose a complex task into parallel sub-tasks.",
        "mapreduce": "Process large lists in chunks then reduce.",
        "broadcast": "Send context to all workers.",
    }

    AUTO_PROMPT = textwrap.dedent("""\
        You are the strategy router for an AI worker swarm. Given the user's
        message and conversation context, choose exactly ONE strategy from:
        single, parallel, ensemble, split, mapreduce, broadcast.
        
        Rules:
        - "single": Casual chat, simple facts, greetings, one-liner questions.
        - "parallel": The user asks multiple distinct things at once, or gives a list of independent items to process.
        - "ensemble": Math, logic, code correctness, factual claims needing verification, or when the user explicitly asks for "best" / "most accurate" answer.
        - "split": Complex multi-step tasks (design, research, planning, writing long documents) that can be broken into independent pieces.
        - "mapreduce": The user provides a large dataset/list and wants aggregation/summary.
        - "broadcast": The user is setting context, persona, or system instructions for future turns.
        
        Respond with ONLY a JSON object: {"strategy": "...", "reason": "...", "workers": N, "subtasks": []}
        - "workers": recommended number of workers (1 for single/broadcast, 3-5 for ensemble, 2-4 for split, equal to chunks for mapreduce)
        - "subtasks": only for split/mapreduce — list of sub-task strings or chunk descriptions. Empty for others.
        
        User message: {query}
    """)

    def __init__(
        self,
        manager_model: str = "gemma4:12b",
        worker_model: str = "qwen3:4b",
        initial_workers: int = 3,
        max_workers: int = 8,
        min_workers: int = 1,
        num_ctx: int = 32768,
        auto_mode: bool = True,
        project_name: Optional[str] = None,
        projects_dir: Optional[Path] = None,
    ):
        self.manager_model = manager_model
        self.worker_model = worker_model
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.auto_mode = auto_mode
        self._forced_strategy: Optional[str] = None

        self.history: List[Turn] = []
        self.turn_count = 0

        # Initialize persistence
        self.store = ProjectStore(project_name=project_name, base_dir=projects_dir)
        print(f"[Project] Saved to: {self.store.project_dir}")

        # Initialize Manager
        self._init_manager(initial_workers, num_ctx)

        # Welcome banner
        self._print_banner()

    def _init_manager(self, num_workers: int, num_ctx: int):
        print(f"[System] Spawning Manager ({self.manager_model}) with {num_workers} workers ({self.worker_model})...")
        self.mgr = Manager(
            model=self.manager_model,
            num_workers=num_workers,
            num_ctx=num_ctx,
            max_parallel=num_workers,
        )
        for w in self.mgr.workers:
            w.model = self.worker_model
        self.current_workers = num_workers

    def _resize_workers(self, n: int) -> int:
        n = max(self.min_workers, min(n, self.max_workers))
        if n == self.current_workers:
            return n

        if n > self.current_workers:
            needed = n - self.current_workers
            print(f"[System] Spawning {needed} new worker(s)...")
            for _ in range(needed):
                from worker import Worker
                w = Worker(
                    model=self.worker_model,
                    default_timeout_seconds=self.mgr.default_timeout_seconds,
                    max_timeout_seconds=self.mgr.max_timeout_seconds,
                    num_ctx=self.mgr.num_ctx,
                )
                self.mgr.workers.append(w)
            self.mgr.num_workers = n
            self.mgr.max_parallel = n
            self.mgr._executor.shutdown(wait=False)
            from concurrent.futures import ThreadPoolExecutor
            self.mgr._executor = ThreadPoolExecutor(max_workers=n)
        else:
            prune = self.current_workers - n
            print(f"[System] Pruning {prune} worker(s)...")
            self.mgr.workers = self.mgr.workers[:n]
            self.mgr.num_workers = n
            self.mgr.max_parallel = n
            self.mgr._executor.shutdown(wait=False)
            from concurrent.futures import ThreadPoolExecutor
            self.mgr._executor = ThreadPoolExecutor(max_workers=n)

        self.current_workers = n
        return n

    def _choose_strategy(self, query: str) -> tuple[str, int, List[str]]:
        if self._forced_strategy:
            return self._forced_strategy, self.current_workers, []

        if not self.auto_mode:
            return "single", self.current_workers, []

        prompt = self.AUTO_PROMPT.format(query=query)
        raw = self.mgr.respond(prompt)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                decision = json.loads(match.group())
            else:
                raise ValueError("No JSON found")
        except Exception:
            return self._heuristic_strategy(query)

        strategy = decision.get("strategy", "single")
        workers = decision.get("workers", self.current_workers)
        subtasks = decision.get("subtasks", [])

        if strategy not in self.STRATEGIES:
            strategy = "single"

        if strategy in ("ensemble", "split", "mapreduce", "parallel"):
            self._resize_workers(workers)
        else:
            self._resize_workers(max(1, min(workers, 2)))

        return strategy, self.current_workers, subtasks

    def _heuristic_strategy(self, query: str) -> tuple[str, int, List[str]]:
        q = query.lower()
        if any(c in q for c in ["/workers", "set context", "persona:", "you are"]):
            return "broadcast", self.current_workers, []
        if len(query) > 800 and ("list" in q or "summarize" in q or "each" in q):
            return "mapreduce", min(4, self.max_workers), []
        if any(k in q for k in ["calculate", "solve", "what is", "logic", "math", "prove"]):
            return "ensemble", min(3, self.max_workers), []
        if len(query) > 500 and ("design" in q or "plan" in q or "write" in q or "create" in q):
            return "split", min(4, self.max_workers), []
        if "and" in q and len(query) > 200:
            parts = [p.strip() for p in re.split(r'[;•]|\n', query) if len(p.strip()) > 20]
            if len(parts) > 1:
                return "parallel", min(len(parts), self.max_workers), []
        return "single", 1, []

    def _execute_single(self, query: str) -> str:
        return self.mgr.delegate(query)

    def _execute_parallel(self, query: str, subtasks: List[str]) -> str:
        if not subtasks:
            parts = [p.strip() for p in re.split(r'[;•]|\n', query) if len(p.strip()) > 10]
            if len(parts) < 2:
                parts = [query]
            subtasks = parts

        results = self.mgr.parallel_respond(subtasks)
        combined = "\n\n---\n\n".join(
            f"[Part {i+1}]\n{r.response}" for i, r in enumerate(results) if not r.error
        )
        if not combined:
            return "Error: All parallel workers failed."
        synthesis = (
            f"The user asked: {query}\n\n"
            f"Here are parallel partial answers:\n\n{combined}\n\n"
            f"Provide one coherent final answer."
        )
        return self.mgr.respond(synthesis)

    def _execute_ensemble(self, query: str) -> str:
        return self.mgr.ensemble_respond(
            query=query,
            n_workers=self.current_workers,
            strategy="synthesize",
        )

    def _execute_split(self, query: str, subtasks: List[str]) -> str:
        if subtasks:
            results = self.mgr.parallel_respond(subtasks)
            combined = "\n\n---\n\n".join(
                f"[Sub-task {i+1}: {t}]\n{r.response if not r.error else f'Error: {r.error}'}"
                for i, (t, r) in enumerate(zip(subtasks, results))
            )
            synthesis = (
                f"Original task: {query}\n\nParallel sub-results:\n\n{combined}\n\n"
                f"Synthesize into one final answer."
            )
            return self.mgr.respond(synthesis)
        return self.mgr.split_and_conquer(query, n_subtasks=self.current_workers)

    def _execute_mapreduce(self, query: str) -> str:
        lines = [l.strip() for l in query.splitlines() if l.strip()]
        items = []
        for line in lines:
            m = re.match(r'^\s*[-•*\d.)\]]+\s*(.+)', line)
            if m:
                items.append(m.group(1))
            elif len(line) > 40 and not line.startswith("Here"):
                items.append(line)

        if len(items) < 2:
            return self._execute_single(query)

        return self.mgr.map_reduce(
            items=items,
            map_prompt_template=(
                "Process these items and extract key insights:\n{item}\n\n"
                "Output concise bullet points only."
            ),
            reduce_prompt=f"Synthesize these mapped results into a final answer for: {query}",
            chunk_size=max(1, len(items) // self.current_workers),
        )

    def _execute_broadcast(self, query: str) -> str:
        results = self.mgr.broadcast(query)
        ok = sum(1 for r in results if not r.error)
        return f"Context broadcast to {ok}/{len(results)} workers. Ready for next query."

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #

    def _persist_turn(self, turn: Turn, strategy: str, workers_used: int):
        """Save turn to disk immediately."""
        self.store.save_turn(turn, strategy, workers_used)
        # Also save full interface state snapshot
        self.store.save_state({
            "manager_model": self.manager_model,
            "worker_model": self.worker_model,
            "current_workers": self.current_workers,
            "auto_mode": self.auto_mode,
            "forced_strategy": self._forced_strategy,
            "turn_count": self.turn_count,
        })

    def _save_artifact(self, filename: str, content: str) -> str:
        """Save a file artifact and return its path."""
        path = self.store.save_artifact(filename, content)
        return str(path)

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def chat(self, query: str) -> str:
        self.turn_count += 1

        if query.startswith("/"):
            return self._handle_command(query)

        # Save user turn
        user_turn = Turn(role="user", content=query)
        self.history.append(user_turn)
        self._persist_turn(user_turn, "", 0)

        # Strategy selection
        strategy, n_workers, subtasks = self._choose_strategy(query)
        print(f"\n[Strategy] {strategy.upper()} | Workers: {n_workers}")

        # Execute
        try:
            if strategy == "single":
                answer = self._execute_single(query)
            elif strategy == "parallel":
                answer = self._execute_parallel(query, subtasks)
            elif strategy == "ensemble":
                answer = self._execute_ensemble(query)
            elif strategy == "split":
                answer = self._execute_split(query, subtasks)
            elif strategy == "mapreduce":
                answer = self._execute_mapreduce(query)
            elif strategy == "broadcast":
                answer = self._execute_broadcast(query)
            else:
                answer = self._execute_single(query)
        except Exception as e:
            answer = f"Error during execution: {type(e).__name__}: {e}"

        # Save assistant turn
        assistant_turn = Turn(
            role="assistant",
            content=answer,
            strategy=strategy,
            workers_used=n_workers,
        )
        self.history.append(assistant_turn)
        self._persist_turn(assistant_turn, strategy, n_workers)

        return answer

    # --------------------------------------------------------------------- #
    # Commands
    # --------------------------------------------------------------------- #

    def _handle_command(self, cmd: str) -> str:
        parts = cmd.strip().split()
        if not parts:
            return ""

        action = parts[0].lower()

        if action == "/quit" or action == "/q":
            print("[System] Saving final state...")
            self._persist_turn(Turn(role="system", content="Session ended."), "system", 0)
            self.mgr.shutdown()
            sys.exit(0)

        elif action == "/help" or action == "/h":
            return self._help_text()

        elif action == "/clear":
            self.history.clear()
            self.turn_count = 0
            return "Conversation history cleared. (Disk files remain.)"

        elif action == "/history":
            return self._format_history()

        elif action == "/workers":
            if len(parts) > 1:
                try:
                    n = int(parts[1])
                    self._resize_workers(n)
                    return f"Worker pool resized to {self.current_workers}."
                except ValueError:
                    return "Usage: /workers <number>"
            return f"Current workers: {self.current_workers} (min: {self.min_workers}, max: {self.max_workers})"

        elif action == "/strategy":
            if len(parts) > 1:
                s = parts[1].lower()
                if s == "auto":
                    self._forced_strategy = None
                    self.auto_mode = True
                    return "Strategy set to AUTO (manager decides)."
                elif s in self.STRATEGIES:
                    self._forced_strategy = s
                    self.auto_mode = False
                    return f"Strategy forced to: {s.upper()}"
                else:
                    return f"Unknown strategy. Available: {', '.join(self.STRATEGIES)}"
            current = self._forced_strategy or "AUTO"
            return f"Current strategy: {current.upper()}"

        elif action == "/model":
            if len(parts) > 1:
                self.manager_model = parts[1]
                old_workers = self.current_workers
                self.mgr.shutdown()
                self._init_manager(old_workers, self.mgr.num_ctx)
                return f"Manager model changed to {self.manager_model}"
            return f"Manager model: {self.manager_model}"

        elif action == "/worker_model":
            if len(parts) > 1:
                self.worker_model = parts[1]
                for w in self.mgr.workers:
                    w.model = self.worker_model
                return f"Worker model changed to {self.worker_model}"
            return f"Worker model: {self.worker_model}"

        elif action == "/status":
            return (
                f"Manager: {self.manager_model}\n"
                f"Workers: {self.worker_model} x {self.current_workers}\n"
                f"Strategy: {self._forced_strategy or 'AUTO'}\n"
                f"Turns: {self.turn_count}\n"
                f"History length: {len(self.history)} messages\n"
                f"Project: {self.store.project_dir}"
            )

        elif action == "/project":
            return f"Current project folder:\n{self.store.project_dir}"

        elif action == "/projects":
            projects = ProjectStore.list_projects(self.store.base_dir)
            if not projects:
                return "No saved projects found."
            lines = ["Saved Projects:", "-" * 40]
            for p in projects:
                lines.append(f"  • {p['folder']}")
                lines.append(f"    Name: {p['name']}")
                lines.append(f"    Created: {p['created']}")
                lines.append(f"    Path: {p['path']}")
                lines.append("")
            return "\n".join(lines)

        elif action == "/save":
            self._persist_turn(Turn(role="system", content="Manual save triggered."), "system", 0)
            return f"State saved to:\n  {self.store.project_dir}"

        elif action == "/artifact":
            # /artifact filename content...
            if len(parts) >= 3:
                filename = parts[1]
                content = " ".join(parts[2:])
                path = self._save_artifact(filename, content)
                return f"Artifact saved:\n  {path}"
            return "Usage: /artifact <filename> <content>"

        else:
            return f"Unknown command: {action}. Type /help for available commands."

    def _help_text(self) -> str:
        return textwrap.dedent(f"""\
        Available commands:
          /quit, /q          Exit the interface
          /clear             Clear in-memory history (disk files remain)
          /history           Show conversation transcript
          /status            Show current configuration
          /workers <n>       Resize worker pool (1-{self.max_workers})
          /strategy <s>      Force strategy or auto
          /model <name>      Change manager model
          /worker_model <n>  Change worker model for all workers
          /project           Show current project folder path
          /projects          List all saved conversation projects
          /save              Force save current state to disk
          /artifact <f> <c>  Save a text artifact to the project folder
          /help, /h          Show this message
        """)

    def _format_history(self) -> str:
        if not self.history:
            return "No history yet."
        lines = []
        for t in self.history:
            meta = f" [{t.strategy}, w={t.workers_used}]" if t.role == "assistant" else ""
            lines.append(f"[{t.role}]{meta}\n{t.content[:500]}{'...' if len(t.content) > 500 else ''}")
        return "\n\n".join(lines)

    def _print_banner(self):
        print(textwrap.dedent(f"""\
        ╔══════════════════════════════════════════════════════════════╗
        ║  SWARM CLI  —  Conversational Worker Orchestrator            ║
        ║  Manager: {self.manager_model:20}                         ║
        ║  Workers: {self.worker_model:20} x {self.current_workers:<3}                    ║
        ║  Mode:    {'AUTO (manager decides strategy)':20}                         ║
        ║  Project: {str(self.store.project_dir):20}                         ║
        ╚══════════════════════════════════════════════════════════════╝
        Type /help for commands. Start chatting below.
        """))

    def run(self):
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[System] Interrupted. Use /quit to exit.")
                continue

            if not user_input:
                continue

            response = self.chat(user_input)
            print(f"\n{response}")


# ------------------------------------------------------------------------- #
# Entry point
# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Swarm CLI Interface")
    parser.add_argument("--manager", default="gemma4:12b", help="Manager model")
    parser.add_argument("--worker", default="qwen3:4b", help="Worker model")
    parser.add_argument("--workers", type=int, default=3, help="Initial worker count")
    parser.add_argument("--max-workers", type=int, default=8, help="Max worker pool size")
    parser.add_argument("--ctx", type=int, default=32768, help="Context window")
    parser.add_argument("--manual", action="store_true", help="Disable auto-strategy")
    parser.add_argument("--project", default=None, help="Project name for this session")
    parser.add_argument("--projects-dir", default=None, help="Custom projects directory (default: Desktop/SwarmProjects)")

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir) if args.projects_dir else None

    iface = Interface(
        manager_model=args.manager,
        worker_model=args.worker,
        initial_workers=args.workers,
        max_workers=args.max_workers,
        num_ctx=args.ctx,
        auto_mode=not args.manual,
        project_name=args.project,
        projects_dir=projects_dir,
    )
    iface.run()