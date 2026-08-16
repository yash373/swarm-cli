# manager.py
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

from worker import Worker


@dataclass
class WorkerResult:
    """Result container for a single worker execution."""
    worker_id: int
    query: str
    response: str
    error: Optional[str] = None


class Manager(Worker):
    """
    A Manager that inherits all Worker capabilities and orchestrates a pool
    of Worker instances in parallel to increase throughput and accuracy.
    
    Usage patterns:
        - parallel_respond():   Distribute independent queries across workers
        - ensemble_respond():   Same query to N workers, synthesize for accuracy
        - map_reduce():         Split data, process chunks in parallel, aggregate
        - split_and_conquer():  Auto-decompose a complex task, parallelize, synthesize
        - delegate():           Round-robin or targeted single-task assignment
    """

    def __init__(
        self,
        model: str,
        num_workers: int = 3,
        default_timeout_seconds: int = 30,
        max_timeout_seconds: int = 1800,
        num_ctx: int = 32768,
        max_parallel: Optional[int] = None,
    ):
        """
        Initialize the Manager and its worker pool.
        
        Args:
            model: Ollama model name (e.g. "llama3.1", "qwen2.5")
            num_workers: How many Worker instances to spawn
            default_timeout_seconds: Per-tool default timeout passed to workers
            max_timeout_seconds: Hard cap on tool execution time
            num_ctx: Context window size for the model
            max_parallel: Max concurrent executions (defaults to num_workers)
        """
        # Initialize the base Worker — the Manager retains its own Worker
        # identity for coordination, planning, and synthesis tasks.
        super().__init__(
            model=model,
            default_timeout_seconds=default_timeout_seconds,
            max_timeout_seconds=max_timeout_seconds,
            num_ctx=num_ctx,
        )

        self.num_workers = num_workers
        self.max_parallel = max_parallel or num_workers

        # Round-robin counter
        self._worker_counter = 0
        self._worker_lock = threading.Lock()

        # Build the worker pool
        self.workers: List[Worker] = []
        for _ in range(num_workers):
            self.workers.append(
                Worker(
                    model=model,
                    default_timeout_seconds=default_timeout_seconds,
                    max_timeout_seconds=max_timeout_seconds,
                    num_ctx=num_ctx,
                )
            )

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=self.max_parallel)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _next_worker_index(self) -> int:
        """Round-robin worker selection."""
        with self._worker_lock:
            idx = self._worker_counter % self.num_workers
            self._worker_counter += 1
            return idx

    def _run_on_worker(self, worker_id: int, query: str) -> WorkerResult:
        """Execute a query on a specific worker and capture any exception."""
        worker = self.workers[worker_id]
        try:
            response = worker.respond(query)
            return WorkerResult(
                worker_id=worker_id, query=query, response=response
            )
        except Exception as exc:
            return WorkerResult(
                worker_id=worker_id,
                query=query,
                response="",
                error=f"{type(exc).__name__}: {exc}",
            )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def parallel_respond(self, queries: List[str]) -> List[WorkerResult]:
        """
        Distribute a list of **independent** queries across the worker pool
        and execute them concurrently. Results are returned in input order.
        
        Args:
            queries: One query string per task you want processed in parallel.
            
        Returns:
            List of WorkerResult, index-aligned with the input `queries`.
        """
        if not queries:
            return []

        # Assign workers round-robin so load is balanced
        assignments = [
            (idx, idx % self.num_workers, query)
            for idx, query in enumerate(queries)
        ]

        futures = {
            self._executor.submit(self._run_on_worker, wid, query): idx
            for idx, wid, query in assignments
        }

        results: List[Optional[WorkerResult]] = [None] * len(queries)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

        return results  # type: ignore[return-value]

    def ensemble_respond(
        self,
        query: str,
        n_workers: Optional[int] = None,
        strategy: str = "synthesize",
    ) -> str:
        """
        Send the **same** query to multiple workers in parallel and aggregate
        their answers to improve accuracy and robustness.
        
        Args:
            query: The question / task to execute.
            n_workers: How many workers to consult (default = all).
            strategy: Aggregation method:
                - "synthesize" (default): Manager reasons over all answers and
                  produces a single best response, resolving contradictions.
                - "vote": Returns the most common exact answer.
                - "first": Returns the first answer that finishes.
                - "all": Returns every answer, labelled by worker.
                
        Returns:
            A single aggregated string.
        """
        n = min(n_workers or self.num_workers, self.num_workers)

        futures = [
            self._executor.submit(self._run_on_worker, i, query)
            for i in range(n)
        ]

        responses: List[str] = []
        for future in futures:
            result = future.result()
            if not result.error:
                responses.append(result.response)

        if not responses:
            return "Error: All workers failed to produce a response."

        if strategy == "first":
            return responses[0]

        if strategy == "all":
            return "\n\n---\n\n".join(
                f"[Worker {i + 1}]\n{r}" for i, r in enumerate(responses)
            )

        if strategy == "vote":
            from collections import Counter

            # Normalize whitespace for fair comparison
            normalized = [re.sub(r"\s+", " ", r.strip()) for r in responses]
            counts = Counter(normalized)
            winner, _ = counts.most_common(1)[0]
            # Return the first original response that matches the winner
            for raw, norm in zip(responses, normalized):
                if norm == winner:
                    return raw
            return winner  # fallback

        # ---- synthesize (default) ----
        synthesis_prompt = (
            f"You are a critical synthesis engine. {len(responses)} independent "
            f"workers answered the same question. Produce a single, accurate, "
            f"and comprehensive answer. Identify consensus, resolve contradictions "
            f"with reasoning, and do not simply concatenate responses.\n\n"
            f"Question: {query}\n\n"
        )
        for i, resp in enumerate(responses, 1):
            synthesis_prompt += f"--- Worker {i} ---\n{resp}\n\n"
        synthesis_prompt += "Synthesized Answer:"

        # Use the Manager's own Worker brain for the final synthesis
        return self.respond(synthesis_prompt)

    def map_reduce(
        self,
        items: List[str],
        map_prompt_template: str,
        reduce_prompt: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        """
        Classic Map-Reduce: split `items` into chunks, map each chunk across
        workers in parallel, then reduce the mapped outputs into one result.
        
        Args:
            items: The data list to process (e.g. URLs, records, questions).
            map_prompt_template: A prompt string containing the literal token
                `{item}` which will be replaced with the chunk content.
            reduce_prompt: Optional custom prompt for the reduce step.
                If omitted, a generic synthesis prompt is used.
            chunk_size: Items per chunk. If None, auto-balanced across workers.
            
        Returns:
            The final reduced string.
        """
        if not items:
            return "Error: No items provided for map-reduce."

        # Balance chunks
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.num_workers)
        chunks = [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

        # ---- MAP ----
        map_queries = []
        for chunk in chunks:
            chunk_text = "\n".join(f"- {it}" for it in chunk)
            # Safer than .format() — avoids accidental brace expansion
            map_queries.append(map_prompt_template.replace("{item}", chunk_text))

        map_results = self.parallel_respond(map_queries)
        mapped = [r.response for r in map_results if not r.error]

        if not mapped:
            return "Error: All map workers failed."

        # ---- REDUCE ----
        combined = "\n\n---\n\n".join(mapped)
        if reduce_prompt:
            final_query = f"{reduce_prompt}\n\n{combined}"
        else:
            final_query = (
                "Synthesize the following parallel mapped results into a single "
                f"coherent output:\n\n{combined}"
            )

        return self.respond(final_query)

    def split_and_conquer(
        self,
        complex_query: str,
        n_subtasks: Optional[int] = None,
    ) -> str:
        """
        Automatically decompose a complex query into independent sub-tasks,
        execute them in parallel, and synthesize the final answer.
        
        Args:
            complex_query: The high-level task to break apart.
            n_subtasks: Target number of sub-tasks (default = num_workers).
            
        Returns:
            The synthesized final answer.
        """
        n = n_subtasks or self.num_workers

        # Use the Manager's own Worker to plan the split
        plan_prompt = (
            f"You are a task-decomposition expert. Break the following complex "
            f"task into exactly {n} independent sub-tasks that can be executed "
            f"in parallel by separate workers. Each sub-task must be self-contained "
            f"and advance the overall solution.\n\n"
            f"Complex Task: {complex_query}\n\n"
            f"Respond with ONLY a JSON array of strings. No markdown, no prose. "
            f'Example: ["Sub-task 1", "Sub-task 2", "Sub-task 3"]'
        )

        plan_response = self.respond(plan_prompt)

        # Extract JSON array robustly
        try:
            match = re.search(r"\[.*\]", plan_response, re.DOTALL)
            if match:
                subtasks = json.loads(match.group())
            else:
                subtasks = [plan_response]
        except json.JSONDecodeError:
            subtasks = [plan_response]

        if not isinstance(subtasks, list):
            subtasks = [str(subtasks)]

        # Execute sub-tasks in parallel
        results = self.parallel_respond(subtasks)

        # Synthesize
        synthesis_prompt = (
            f"You are a synthesis expert. The following sub-tasks were executed "
            f"in parallel. Combine them into one coherent, accurate answer.\n\n"
            f"Original Problem: {complex_query}\n\n"
        )
        for i, (task, res) in enumerate(zip(subtasks, results), 1):
            synthesis_prompt += f"--- Sub-task {i}: {task} ---\n"
            synthesis_prompt += f"{res.response if not res.error else f'Error: {res.error}'}\n\n"
        synthesis_prompt += "Final Synthesized Answer:"

        return self.respond(synthesis_prompt)

    def delegate(self, query: str, worker_index: Optional[int] = None) -> str:
        """
        Delegate a single query to one worker (round-robin by default).
        
        Args:
            query: The task to run.
            worker_index: Explicit worker to use, or None for round-robin.
            
        Returns:
            The worker's response string.
        """
        wid = worker_index if worker_index is not None else self._next_worker_index()
        result = self._run_on_worker(wid, query)
        if result.error:
            return f"Error from worker {wid}: {result.error}"
        return result.response

    def broadcast(self, query: str) -> List[WorkerResult]:
        """
        Send the **same** query to **every** worker in the pool.
        Useful for seeding all workers with shared context or system state.
        
        Args:
            query: The message / instruction to broadcast.
            
        Returns:
            List of WorkerResult from every worker.
        """
        futures = [
            self._executor.submit(self._run_on_worker, i, query)
            for i in range(self.num_workers)
        ]
        return [f.result() for f in futures]

    def shutdown(self):
        """Gracefully shut down the thread pool."""
        self._executor.shutdown(wait=True)

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False