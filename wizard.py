# wizard.py
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ui import UI, Style
from store import ProjectStore


class StartupWizard:
    """Interactive startup menu with selectors instead of flags."""

    PRESET_MANAGERS = [
        ("gemma4:12b", "Gemma 4 — 12B (balanced, recommended)"),
        ("gemma4:27b", "Gemma 4 — 27B (strongest, slower)"),
        ("qwen3:8b", "Qwen 3 — 8B (fast, capable)"),
        ("llama3.1:8b", "Llama 3.1 — 8B (general purpose)"),
        ("mistral:7b", "Mistral — 7B (efficient)"),
        ("custom", "Other (type manually)"),
    ]

    PRESET_WORKERS = [
        ("qwen3:4b", "Qwen 3 — 4B (fast, efficient, recommended)"),
        ("qwen3:8b", "Qwen 3 — 8B (stronger workers)"),
        ("gemma4:4b", "Gemma 4 — 4B (if available)"),
        ("phi4:4b", "Phi 4 — 4B (Microsoft)"),
        ("custom", "Other (type manually)"),
    ]

    PRESET_CTX = [
        (32768, "32K — Standard, ~4-6 GB VRAM (default)"),
        (65536, "64K — Long documents, ~8 GB VRAM"),
        (131072, "128K — Very large context, ~16 GB VRAM"),
        (262144, "256K — Massive context, ~24+ GB VRAM (A100/4090)"),
    ]

    def __init__(self):
        self.base_dir = ProjectStore._default_projects_dir()
        self.existing_projects = ProjectStore.list_projects(self.base_dir)

    def _ask(self, text: str, default: str = "") -> str:
        prompt = UI.prompt(text)
        if default:
            prompt += f"{Style.DIM}[{default}]{Style.RESET} "
        return input(prompt).strip() or default

    def _ask_int(self, text: str, default: int, min_val: int, max_val: int) -> int:
        while True:
            val = self._ask(text, str(default))
            try:
                n = int(val)
                if min_val <= n <= max_val:
                    return n
                print(UI.info(f"Please enter a number between {min_val} and {max_val}."))
            except ValueError:
                print(UI.info("Please enter a valid number."))

    def _selector(self, title: str, options: List[tuple]) -> str:
        print(UI.section(title))
        for i, (value, desc) in enumerate(options, 1):
            print(UI.menu_item(i, str(value), desc))
        print()
        choice = self._ask_int("Select option", 1, 1, len(options))
        selected = options[choice - 1][0]
        if selected == "custom":
            return self._ask("Enter custom value")
        return selected

    def run(self) -> Dict[str, Any]:
        print(UI.banner("SWARM CLI", "Conversational Worker Orchestrator"))

        print(UI.section("Welcome"))
        print(UI.menu_item(1, "New Project", "Start a fresh session with configuration"))
        has_projects = len(self.existing_projects) > 0
        print(UI.menu_item(2, "Resume Project", "Continue a previous session" if has_projects else "No saved projects", disabled=not has_projects))
        print(UI.menu_item(3, "Quick Start", "Default settings, skip configuration"))
        print(UI.menu_item(4, "Power User", "Skip wizard, use command-line flags"))
        print()

        choice = self._ask_int("Select option", 1, 1, 4)

        if choice == 4:
            return {"skip_wizard": True}

        if choice == 3:
            return self._quick_start()

        if choice == 2:
            return self._resume_project()

        return self._new_project()

    def _quick_start(self) -> Dict[str, Any]:
        print(UI.status("QUICK", "Using defaults: gemma4:12b → qwen3:4b × 3", Style.GREEN))
        return {
            "manager_model": "gemma4:12b",
            "worker_model": "qwen3:4b",
            "initial_workers": 3,
            "max_workers": 8,
            "num_ctx": 32768,
            "auto_mode": True,
            "project_name": None,
            "projects_dir": None,
        }

    def _resume_project(self) -> Dict[str, Any]:
        print(UI.section("Resume Project"))
        print(UI.info(f"Found {len(self.existing_projects)} project(s)\n"))

        for i, p in enumerate(self.existing_projects[:10], 1):
            print(UI.menu_item(i, p["folder"][:40], f"Created: {p['created']} | Files: {p['files']}"))
        print(UI.menu_item(0, "Back", "Return to main menu"))
        print()

        choice = self._ask_int("Select project", 1, 0, min(len(self.existing_projects), 10))
        if choice == 0:
            return self.run()

        selected = self.existing_projects[choice - 1]
        project_name = selected["folder"].split("_", 2)[-1] if "_" in selected["folder"] else selected["folder"]

        print(UI.status("RESUME", f"Resuming '{project_name}'", Style.GREEN))
        return {
            "manager_model": "gemma4:12b",
            "worker_model": "qwen3:4b",
            "initial_workers": 3,
            "max_workers": 8,
            "num_ctx": 32768,
            "auto_mode": True,
            "project_name": project_name,
            "projects_dir": self.base_dir,
        }

    def _new_project(self) -> Dict[str, Any]:
        print(UI.section("New Project"))
        project_name = self._ask("Project name", f"Swarm_{datetime.now().strftime('%H%M')}")

        manager_model = self._selector("Select Manager Model (reasoning & synthesis)", self.PRESET_MANAGERS)
        worker_model = self._selector("Select Worker Model (parallel execution)", self.PRESET_WORKERS)

        print(UI.section("Worker Pool"))
        workers = self._ask_int("Initial workers (1-8)", 3, 1, 8)
        max_workers = self._ask_int("Max workers (1-16)", max(workers, 8), 1, 16)

        ctx = int(self._selector("Context Window (VRAM estimate)", self.PRESET_CTX))

        print(UI.section("Strategy Mode"))
        print(UI.menu_item(1, "Auto", "Manager intelligently picks strategies (recommended)"))
        print(UI.menu_item(2, "Manual", "You pick strategies with /strategy during chat"))
        print()
        mode_choice = self._ask_int("Select mode", 1, 1, 2)
        auto_mode = mode_choice == 1

        print()
        print(UI.banner("Configuration Summary"))
        print(f"  {Style.BOLD}Project:{Style.RESET}     {project_name}")
        print(f"  {Style.BOLD}Manager:{Style.RESET}     {manager_model}")
        print(f"  {Style.BOLD}Workers:{Style.RESET}     {worker_model} × {workers} (max {max_workers})")
        print(f"  {Style.BOLD}Context:{Style.RESET}     {ctx:,} tokens")
        print(f"  {Style.BOLD}Mode:{Style.RESET}        {'Auto' if auto_mode else 'Manual'}")
        print(f"  {Style.BOLD}Save to:{Style.RESET}     {self.base_dir}")
        print()

        confirm = self._ask("Press Enter to start, or type 'back' to reconfigure", "")
        if confirm.lower() == "back":
            return self._new_project()

        return {
            "manager_model": manager_model,
            "worker_model": worker_model,
            "initial_workers": workers,
            "max_workers": max_workers,
            "num_ctx": ctx,
            "auto_mode": auto_mode,
            "project_name": project_name,
            "projects_dir": None,
        }
