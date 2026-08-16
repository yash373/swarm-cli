# interface.py
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from manager import Manager, WorkerResult


@dataclass
class Turn:
    role: str          # "user" | "assistant" | "system"
    content: str
    strategy: str = "single"
    workers_used: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Interface:
    """
    Conversational CLI that wraps Manager.
    
    Features:
      - Persistent multi-turn conversation history
      - Auto-strategy selection (single / parallel / ensemble / split / mapreduce)
      - Dynamic worker spawn/prune
      - Rich slash commands for runtime control
      - Streaming-like progress reporting
    """

    # --------------------------------------------------------------------- #
    # Strategy definitions
    # --------------------------------------------------------------------- #
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
    ):
        self.manager_model = manager_model
        self.worker_model = worker_model
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.auto_mode = auto_mode
        self._forced_strategy: Optional[str] = None

        self.history: List[Turn] = []
        self.turn_count = 0

        # Initialize Manager
        self._init_manager(initial_workers, num_ctx)

        # Welcome banner
        self._print_banner()

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

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
        """Spawn or prune workers to reach target count."""
        n = max(self.min_workers, min(n, self.max_workers))
        if n == self.current_workers:
            return n

        if n > self.current_workers:
            # Spawn additional workers
            needed = n - self.current_workers
            print(f"[System] Spawning {needed} new worker(s)...")
            for _ in range(needed):
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
            # Prune excess workers
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

    # --------------------------------------------------------------------- #
    # Strategy routing
    # --------------------------------------------------------------------- #

    def _choose_strategy(self, query: str) -> tuple[str, int, List[str]]:
        """
        Returns (strategy, num_workers, subtasks).
        If auto_mode is off, uses forced strategy.
        """
        if self._forced_strategy:
            return self._forced_strategy, self.current_workers, []

        if not self.auto_mode:
            return "single", self.current_workers, []

        # Use Manager's own brain to classify
        prompt = self.AUTO_PROMPT.format(query=query)
        raw = self.mgr.respond(prompt)

        # Extract JSON
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                decision = json.loads(match.group())
            else:
                raise ValueError("No JSON found")
        except Exception:
            # Fallback to heuristic
            return self._heuristic_strategy(query)

        strategy = decision.get("strategy", "single")
        workers = decision.get("workers", self.current_workers)
        subtasks = decision.get("subtasks", [])

        if strategy not in self.STRATEGIES:
            strategy = "single"

        # Resize if needed
        if strategy in ("ensemble", "split", "mapreduce", "parallel"):
            self._resize_workers(workers)
        else:
            self._resize_workers(max(1, min(workers, 2)))

        return strategy, self.current_workers, subtasks

    def _heuristic_strategy(self, query: str) -> tuple[str, int, List[str]]:
        """Fast fallback when JSON parsing fails."""
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

    # --------------------------------------------------------------------- #
    # Execution engines
    # --------------------------------------------------------------------- #

    def _execute_single(self, query: str) -> str:
        return self.mgr.delegate(query)

    def _execute_parallel(self, query: str, subtasks: List[str]) -> str:
        if not subtasks:
            # Try to split on newlines / bullets / "and"
            parts = [p.strip() for p in re.split(r'[;•]|\n', query) if len(p.strip()) > 10]
            if len(parts) < 2:
                parts = [query]  # Fallback
            subtasks = parts

        results = self.mgr.parallel_respond(subtasks)
        # Synthesize through manager
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
            # Override auto-subtasks with provided ones
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
        # Extract list items from query heuristically
        lines = [l.strip() for l in query.splitlines() if l.strip()]
        items = []
        for line in lines:
            # Match bullet points, numbers, or quoted blocks
            m = re.match(r'^\s*[-•*\d.)\]]+\s*(.+)', line)
            if m:
                items.append(m.group(1))
            elif len(line) > 40 and not line.startswith("Here"):
                items.append(line)

        if len(items) < 2:
            # Not enough items — treat as single
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
    # Main loop
    # --------------------------------------------------------------------- #

    def chat(self, query: str) -> str:
        """
        Process one user turn. Auto-selects strategy, executes, stores history.
        Returns the assistant's response string.
        """
        self.turn_count += 1

        # Slash commands
        if query.startswith("/"):
            return self._handle_command(query)

        # Add user turn
        self.history.append(Turn(role="user", content=query))

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

        # Store assistant turn
        self.history.append(Turn(
            role="assistant",
            content=answer,
            strategy=strategy,
            workers_used=n_workers,
        ))

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
            print("[System] Shutting down...")
            self.mgr.shutdown()
            sys.exit(0)

        elif action == "/help" or action == "/h":
            return self._help_text()

        elif action == "/clear":
            self.history.clear()
            self.turn_count = 0
            return "Conversation history cleared."

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
                # Re-init manager with new model
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
                f"History length: {len(self.history)} messages"
            )

        else:
            return f"Unknown command: {action}. Type /help for available commands."

    def _help_text(self) -> str:
        return textwrap.dedent(f"""\
        Available commands:
          /quit, /q          Exit the interface
          /clear             Clear conversation history
          /history           Show conversation transcript
          /status            Show current configuration
          /workers <n>       Resize worker pool (1-{self.max_workers})
          /strategy <s>      Force strategy: {', '.join(self.STRATEGIES)} or auto
          /model <name>      Change manager model (requires restart)
          /worker_model <n>  Change worker model for all workers
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
        ╚══════════════════════════════════════════════════════════════╝
        Type /help for commands. Start chatting below.
        """))

    # --------------------------------------------------------------------- #
    # Interactive runner
    # --------------------------------------------------------------------- #

    def run(self):
        """Blocking REPL loop."""
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

    args = parser.parse_args()

    iface = Interface(
        manager_model=args.manager,
        worker_model=args.worker,
        initial_workers=args.workers,
        max_workers=args.max_workers,
        num_ctx=args.ctx,
        auto_mode=not args.manual,
    )
    iface.run()