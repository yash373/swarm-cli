# interface.py
import argparse
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from manager import Manager
from ui import UI, Style
from store import ProjectStore, CodeExtractor
from wizard import StartupWizard


@dataclass
class Turn:
    role: str
    content: str
    strategy: str = "single"
    workers_used: int = 1


class Interface:
    STRATEGIES = {
        "single": "One worker, one shot.",
        "parallel": "Multiple independent queries in parallel.",
        "ensemble": "N workers vote, synthesized for accuracy.",
        "split": "Auto-decomposed into parallel sub-tasks.",
        "mapreduce": "Chunked processing then aggregation.",
        "broadcast": "Context broadcast to all workers.",
    }

    AUTO_PROMPT = textwrap.dedent("""\
        You are the strategy router for an AI worker swarm. Given the user's
        message, choose exactly ONE strategy from: single, parallel, ensemble,
        split, mapreduce, broadcast.
        
        Rules:
        - "single": Simple Q&A, greetings, one-liners.
        - "parallel": Multiple distinct questions or independent items.
        - "ensemble": Math, logic, code, verification, "best" answer requests.
        - "split": Complex multi-step tasks (design, planning, writing).
        - "mapreduce": Large datasets/lists needing aggregation.
        - "broadcast": Context setting, persona, system instructions.
        
        Respond with ONLY JSON: {"strategy":"...","reason":"...","workers":N,"subtasks":[]}
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

        self.store = ProjectStore(project_name=project_name, base_dir=projects_dir)
        self._print_opening()
        self._init_manager(initial_workers, num_ctx)

    def _init_manager(self, num_workers: int, num_ctx: int):
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
            print(UI.status("SPAWN", f"{needed} worker(s) → pool: {n}", Style.GREEN))
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
            print(UI.status("PRUNE", f"{prune} worker(s) → pool: {n}", Style.YELLOW))
            self.mgr.workers = self.mgr.workers[:n]
            self.mgr.num_workers = n
            self.mgr.max_parallel = n
            self.mgr._executor.shutdown(wait=False)
            from concurrent.futures import ThreadPoolExecutor
            self.mgr._executor = ThreadPoolExecutor(max_workers=n)
        self.current_workers = n
        return n

    def _choose_strategy(self, query: str):
        if self._forced_strategy:
            return self._forced_strategy, self.current_workers, []
        if not self.auto_mode:
            return "single", self.current_workers, []

        print(UI.status("ROUTING", "Manager analyzing task...", Style.MAGENTA), end="", flush=True)
        prompt = self.AUTO_PROMPT.format(query=query)
        raw = self.mgr.respond(prompt)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                decision = json.loads(match.group())
            else:
                raise ValueError("No JSON found")
        except Exception:
            print(f"\r{UI.status('ROUTING', 'Fallback heuristic', Style.YELLOW)}")
            return self._heuristic_strategy(query)

        strategy = decision.get("strategy", "single")
        workers = decision.get("workers", self.current_workers)
        subtasks = decision.get("subtasks", [])
        if strategy not in self.STRATEGIES:
            strategy = "single"

        reason = decision.get("reason", "auto-selected")
        print(f"\r{UI.status('ROUTING', f'{strategy.upper()} — {reason}', Style.GREEN)}")

        if strategy in ("ensemble", "split", "mapreduce", "parallel"):
            self._resize_workers(workers)
        else:
            self._resize_workers(max(1, min(workers, 2)))
        return strategy, self.current_workers, subtasks

    def _heuristic_strategy(self, query: str):
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
        print(UI.status("WORKING", "Delegating to worker...", Style.CYAN))
        return self.mgr.delegate(query)

    def _execute_parallel(self, query: str, subtasks: List[str]) -> str:
        if not subtasks:
            parts = [p.strip() for p in re.split(r'[;•]|\n', query) if len(p.strip()) > 10]
            if len(parts) < 2:
                parts = [query]
            subtasks = parts
        print(UI.status("PARALLEL", f"Distributing across {len(subtasks)} task(s)...", Style.BLUE))
        results = self.mgr.parallel_respond(subtasks)
        for i, r in enumerate(results):
            status = f"{Style.GREEN}✓{Style.RESET}" if not r.error else f"{Style.RED}✗{Style.RESET}"
            print(f"  {status}  Task {i+1}/{len(subtasks)}  Worker {r.worker_id}")
        combined = "\n\n---\n\n".join(
            f"[Part {i+1}]\n{r.response}" for i, r in enumerate(results) if not r.error
        )
        if not combined:
            return "Error: All parallel workers failed."
        print(UI.status("SYNTH", "Manager synthesizing final answer...", Style.MAGENTA))
        synthesis = (
            f"The user asked: {query}\n\n"
            f"Here are parallel partial answers:\n\n{combined}\n\n"
            f"Provide one coherent final answer."
        )
        return self.mgr.respond(synthesis)

    def _execute_ensemble(self, query: str) -> str:
        n = self.current_workers
        print(UI.status("ENSEMBLE", f"Consulting {n} workers for consensus...", Style.BLUE))
        return self.mgr.ensemble_respond(query=query, n_workers=n, strategy="synthesize")

    def _execute_split(self, query: str, subtasks: List[str]) -> str:
        if subtasks:
            print(UI.status("SPLIT", f"Executing {len(subtasks)} sub-tasks in parallel...", Style.BLUE))
            results = self.mgr.parallel_respond(subtasks)
            for i, r in enumerate(results):
                status = f"{Style.GREEN}✓{Style.RESET}" if not r.error else f"{Style.RED}✗{Style.RESET}"
                print(f"  {status}  Sub-task {i+1}/{len(subtasks)}")
            combined = "\n\n---\n\n".join(
                f"[Sub-task {i+1}: {t}]\n{r.response if not r.error else f'Error: {r.error}'}"
                for i, (t, r) in enumerate(zip(subtasks, results))
            )
            print(UI.status("SYNTH", "Assembling sub-results...", Style.MAGENTA))
            synthesis = (
                f"Original task: {query}\n\nParallel sub-results:\n\n{combined}\n\n"
                f"Synthesize into one final answer."
            )
            return self.mgr.respond(synthesis)
        print(UI.status("SPLIT", "Manager decomposing task...", Style.BLUE))
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
        chunk_size = max(1, len(items) // self.current_workers)
        print(UI.status("MAP", f"Chunking {len(items)} items into {len(items)//chunk_size} batches...", Style.BLUE))
        return self.mgr.map_reduce(
            items=items,
            map_prompt_template=(
                "Process these items and extract key insights:\n{item}\n\n"
                "Output concise bullet points only."
            ),
            reduce_prompt=f"Synthesize these mapped results into a final answer for: {query}",
            chunk_size=chunk_size,
        )

    def _execute_broadcast(self, query: str) -> str:
        print(UI.status("BROADCAST", f"Seeding {self.current_workers} workers...", Style.YELLOW))
        results = self.mgr.broadcast(query)
        ok = sum(1 for r in results if not r.error)
        return f"Context broadcast to {ok}/{len(results)} workers."

    def _auto_save_artifacts(self, response: str) -> List[Path]:
        blocks = CodeExtractor.extract(response)
        saved: List[Path] = []
        for block in blocks:
            path = self.store.save_code(block["filename"], block["content"], block["language"])
            saved.append(path)
        return saved

    def chat(self, query: str) -> str:
        self.turn_count += 1
        if query.startswith("/"):
            return self._handle_command(query)

        user_turn = Turn(role="user", content=query)
        self.history.append(user_turn)
        self.store.save_turn("user", query)

        strategy, n_workers, subtasks = self._choose_strategy(query)

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
            answer = f"Error: {type(e).__name__}: {e}"

        saved_files = self._auto_save_artifacts(answer)
        assistant_turn = Turn(role="assistant", content=answer, strategy=strategy, workers_used=n_workers)
        self.history.append(assistant_turn)
        self.store.save_turn("assistant", answer, strategy, n_workers)
        self.store.save_state({
            "manager_model": self.manager_model,
            "worker_model": self.worker_model,
            "current_workers": self.current_workers,
            "auto_mode": self.auto_mode,
            "forced_strategy": self._forced_strategy,
            "turn_count": self.turn_count,
        })

        if saved_files:
            print()
            for path in saved_files:
                print(UI.artifact_saved(path))
        return answer

    def _handle_command(self, cmd: str) -> str:
        parts = cmd.strip().split()
        if not parts:
            return ""
        action = parts[0].lower()

        if action in ("/quit", "/q"):
            print(UI.status("EXIT", "Saving state and shutting down...", Style.YELLOW))
            self.store.save_turn("system", "Session ended.", "system", 0)
            self.mgr.shutdown()
            sys.exit(0)

        elif action in ("/help", "/h"):
            return self._help_text()
        elif action == "/clear":
            self.history.clear()
            self.turn_count = 0
            return "In-memory history cleared. Disk archive remains."
        elif action == "/history":
            return self._format_history()
        elif action == "/workers":
            if len(parts) > 1:
                try:
                    n = int(parts[1])
                    self._resize_workers(n)
                    return f"Pool size: {self.current_workers}"
                except ValueError:
                    return "Usage: /workers <number>"
            return f"Pool: {self.current_workers} (max {self.max_workers})"
        elif action == "/strategy":
            if len(parts) > 1:
                s = parts[1].lower()
                if s == "auto":
                    self._forced_strategy = None
                    self.auto_mode = True
                    return "Strategy: AUTO"
                elif s in self.STRATEGIES:
                    self._forced_strategy = s
                    self.auto_mode = False
                    return f"Strategy: {s.upper()}"
                else:
                    return f"Unknown. Available: {', '.join(self.STRATEGIES)}"
            current = self._forced_strategy or "AUTO"
            return f"Strategy: {current.upper()}"
        elif action == "/model":
            if len(parts) > 1:
                self.manager_model = parts[1]
                old = self.current_workers
                self.mgr.shutdown()
                self._init_manager(old, self.mgr.num_ctx)
                return f"Manager: {self.manager_model}"
            return f"Manager: {self.manager_model}"
        elif action == "/worker_model":
            if len(parts) > 1:
                self.worker_model = parts[1]
                for w in self.mgr.workers:
                    w.model = self.worker_model
                return f"Workers: {self.worker_model}"
            return f"Workers: {self.worker_model}"
        elif action == "/status":
            return (
                f"Manager:     {self.manager_model}\n"
                f"Workers:     {self.worker_model} × {self.current_workers}\n"
                f"Strategy:    {self._forced_strategy or 'AUTO'}\n"
                f"Turns:       {self.turn_count}\n"
                f"Project:     {self.store.project_dir}"
            )
        elif action == "/project":
            return str(self.store.project_dir)
        elif action == "/projects":
            projects = ProjectStore.list_projects(self.store.base_dir)
            if not projects:
                return "No saved projects."
            lines = [f"{Style.BOLD}Saved Projects{Style.RESET}", UI.divider()]
            for p in projects:
                lines.append(f"  {Style.CYAN}●{Style.RESET} {p['folder']}")
                lines.append(f"     {Style.DIM}Created:{Style.RESET} {p['created']}  {Style.DIM}Files:{Style.RESET} {p['files']}")
            return "\n".join(lines)
        elif action == "/save":
            self.store.save_state({
                "manager_model": self.manager_model,
                "worker_model": self.worker_model,
                "current_workers": self.current_workers,
                "turn_count": self.turn_count,
            })
            return "Snapshot saved."
        elif action == "/files":
            src = self.store.src_dir
            if not src.exists() or not any(src.iterdir()):
                return "No generated files yet."
            lines = [f"{Style.BOLD}Generated Files{Style.RESET}", UI.divider()]
            for f in sorted(src.iterdir()):
                size = f.stat().st_size
                unit = "B" if size < 1024 else "KB"
                size_str = f"{size}" if size < 1024 else f"{size/1024:.1f}"
                lines.append(f"  {Style.GREEN}●{Style.RESET} {f.name:<30} {Style.DIM}{size_str} {unit}{Style.RESET}")
            return "\n".join(lines)
        else:
            return f"Unknown command. Type /help."

    def _help_text(self) -> str:
        return textwrap.dedent(f"""\
        {Style.BOLD}Commands{Style.RESET}
          /quit, /q          Exit
          /clear             Clear memory (disk kept)
          /history           Show transcript
          /status            Configuration
          /workers <n>       Resize pool
          /strategy <s>      Force: single, parallel, ensemble, split, mapreduce, broadcast, auto
          /model <name>      Change manager model
          /worker_model <n>  Change worker model
          /project           Current project path
          /projects          List all projects
          /files             List generated code files
          /save              Force snapshot
          /help, /h          This message
        """)

    def _format_history(self) -> str:
        if not self.history:
            return "No history."
        lines = []
        for t in self.history:
            meta = f"  {Style.DIM}[{t.strategy}, ×{t.workers_used}]{Style.RESET}" if t.role == "assistant" else ""
            lines.append(f"{Style.BOLD}{t.role.upper()}{Style.RESET}{meta}\n{t.content[:400]}{'…' if len(t.content) > 400 else ''}")
        return "\n\n".join(lines)

    def _print_opening(self):
        print(UI.banner(
            "SWARM CLI",
            f"{self.manager_model}  →  {self.worker_model} × {self.current_workers}"
        ))
        print(UI.status("PROJECT", f"{self.store.project_dir.name}", Style.GREEN))
        print(f"  {Style.DIM}Location:{Style.RESET} {self.store.project_dir}")
        print()

    def run(self):
        while True:
            try:
                prompt = f"\n{Style.BOLD}{Style.CYAN}⟩{Style.RESET}  "
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{UI.status('INFO', 'Use /quit to exit.', Style.YELLOW)}")
                continue
            if not user_input:
                continue
            response = self.chat(user_input)
            print(f"\n{response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm CLI")
    parser.add_argument("--manager", default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--ctx", type=int, default=None)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--project", default=None)
    parser.add_argument("--projects-dir", default=None)
    args = parser.parse_args()

    power_flags_set = any([
        args.manager, args.worker, args.workers, args.max_workers, args.ctx,
        args.manual, args.project, args.projects_dir,
    ])

    if power_flags_set:
        config = {
            "manager_model": args.manager or "gemma4:12b",
            "worker_model": args.worker or "qwen3:4b",
            "initial_workers": args.workers or 3,
            "max_workers": args.max_workers or 8,
            "num_ctx": args.ctx or 32768,
            "auto_mode": not args.manual,
            "project_name": args.project,
            "projects_dir": Path(args.projects_dir) if args.projects_dir else None,
        }
    else:
        wizard = StartupWizard()
        config = wizard.run()
        if config.get("skip_wizard"):
            print(UI.banner("Power User Mode", "Use flags to configure directly"))
            print(f"\n  {Style.DIM}python interface.py --manager gemma4:12b --worker qwen3:4b --workers 4 --project MyApp{Style.RESET}\n")
            sys.exit(0)

    iface = Interface(**config)
    iface.run()