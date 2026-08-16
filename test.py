# example_usage.py
from manager import Manager


def main():
    # =====================================================================
    # Setup: Gemma 4 (12B) acts as the "brain" — it plans, synthesizes,
    # and decomposes tasks. Qwen 3 (4B) instances do the parallel grunt work.
    # =====================================================================
    with Manager(
        model="gemma4:12b",          # Manager's own model (coordination & synthesis)
        num_workers=4,               # Spawn 4 Qwen workers
        default_timeout_seconds=60,
        max_timeout_seconds=1800,
        num_ctx=32768,
        max_parallel=4,
    ) as mgr:
        
        # Override each worker's model to Qwen 3 (4B) — lightweight & fast
        for worker in mgr.workers:
            worker.model = "qwen3:4b"

        print("=" * 60)
        print("MANAGER: gemma4:12b  |  WORKERS: qwen3:4b x4")
        print("=" * 60)

        # -----------------------------------------------------------------
        # 1. PARALLEL RESPOND — Independent queries, 4 workers at once
        # -----------------------------------------------------------------
        print("\n--- 1. PARALLEL RESPOND ---")
        queries = [
            "Explain quantum tunneling in 3 sentences.",
            "Write a Python one-liner to reverse a dictionary.",
            "What are the primary differences between TCP and UDP?",
            "Calculate the factorial of 12 and explain the steps.",
        ]
        results = mgr.parallel_respond(queries)
        for r in results:
            status = "OK" if not r.error else f"ERR: {r.error}"
            print(f"\n[Worker {r.worker_id}] {status}")
            print(r.response[:300] + "..." if len(r.response) > 300 else r.response)

        # -----------------------------------------------------------------
        # 2. ENSEMBLE RESPOND — Same hard problem to 3 workers, Gemma decides
        # -----------------------------------------------------------------
        print("\n--- 2. ENSEMBLE RESPOND (synthesize strategy) ---")
        hard_problem = (
            "A train leaves Station A at 60 mph. Another leaves Station B "
            "at 80 mph toward A on the same track, 420 miles apart. "
            "A bird flies back and forth between them at 120 mph until they "
            "collide. How far does the bird fly?"
        )
        answer = mgr.ensemble_respond(
            query=hard_problem,
            n_workers=3,
            strategy="synthesize",   # Gemma 4 (12B) resolves contradictions
        )
        print(f"Synthesized Answer:\n{answer}")

        # -----------------------------------------------------------------
        # 3. ENSEMBLE VOTE — Deterministic/categorical task
        # -----------------------------------------------------------------
        print("\n--- 3. ENSEMBLE RESPOND (vote strategy) ---")
        classification = (
            "Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Respond with ONLY the label.\n\n"
            "Review: 'The movie was visually stunning but the plot made no sense.'"
        )
        vote_result = mgr.ensemble_respond(
            query=classification,
            n_workers=3,
            strategy="vote",
        )
        print(f"Majority Vote Result: {vote_result.strip()}")

        # -----------------------------------------------------------------
        # 4. MAP-REDUCE — Large dataset processed in parallel chunks
        # -----------------------------------------------------------------
        print("\n--- 4. MAP-REDUCE ---")
        documents = [
            "Doc 1: The Mars rover Perseverance landed in Jezero Crater in 2021.",
            "Doc 2: Ingenuity was the first helicopter to fly on another planet.",
            "Doc 3: NASA's Artemis program aims to return humans to the Moon by 2026.",
            "Doc 4: The James Webb Space Telescope observes infrared light from distant galaxies.",
            "Doc 5: SpaceX Starship is a fully reusable super-heavy launch vehicle.",
            "Doc 6: The Voyager probes have left the solar system and entered interstellar space.",
        ]
        summary = mgr.map_reduce(
            items=documents,
            map_prompt_template=(
                "Read these space-exploration documents and extract key facts:\n{item}\n\n"
                "Output a concise bullet list of facts only."
            ),
            reduce_prompt=(
                "You are Gemma 4 (12B). Synthesize the following mapped bullet lists "
                "into one coherent executive summary of space exploration milestones."
            ),
            chunk_size=2,  # 3 chunks of 2 docs each → 3 parallel workers
        )
        print(f"Map-Reduce Summary:\n{summary}")

        # -----------------------------------------------------------------
        # 5. SPLIT-AND-CONQUER — Gemma plans, Qwen workers execute in parallel
        # -----------------------------------------------------------------
        print("\n--- 5. SPLIT-AND-CONQUER ---")
        complex_task = (
            "Design a simple REST API for a task-management app. Include: "
            "1) Endpoint definitions with HTTP methods, "
            "2) A JSON schema for the Task model, "
            "3) Basic error handling strategy, "
            "4) A short example request/response for each endpoint."
        )
        design_doc = mgr.split_and_conquer(
            complex_query=complex_task,
            n_subtasks=4,  # Gemma breaks this into 4 parallel sub-tasks
        )
        print(f"Split-and-Conquer Result:\n{design_doc}")

        # -----------------------------------------------------------------
        # 6. DELEGATE — Round-robin single task dispatch
        # -----------------------------------------------------------------
        print("\n--- 6. DELEGATE (Round-Robin) ---")
        for i in range(3):
            result = mgr.delegate(f"Quick math: What is {i+7} * {i+11}?")
            print(f"  Task {i+1} → {result.strip()}")

        # -----------------------------------------------------------------
        # 7. BROADCAST — Seed all workers with shared context
        # -----------------------------------------------------------------
        print("\n--- 7. BROADCAST ---")
        context = (
            "You are now operating in 'medical assistant' mode. "
            "Always add a disclaimer that you are not a real doctor."
        )
        acks = mgr.broadcast(context)
        for ack in acks:
            print(f"  Worker {ack.worker_id} acknowledged: {ack.response[:80]}...")

        # Now any subsequent delegate/parallel call has that context in each worker
        medical_answer = mgr.delegate("What are common symptoms of dehydration?")
        print(f"\n  Context-aware answer: {medical_answer[:200]}...")


if __name__ == "__main__":
    main()