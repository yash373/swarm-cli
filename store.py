# store.py
import json
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


class ProjectStore:
    """
    Organizes every session like a real software project:
        ~/Desktop/SwarmProjects/
        └── YYYY-MM-DD_HHMMSS_ProjectName/
            ├── README.md
            ├── src/           ← generated code files
            ├── data/          ← JSON/CSV outputs
            ├── docs/          ← markdown docs & transcripts
            └── logs/          ← session logs & history
    """

    def __init__(self, project_name: Optional[str] = None, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or self._default_projects_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.project_name = project_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe_name = re.sub(r'[<>"/\\|?*]', "_", self.project_name).strip() or "Untitled"
        self.project_dir = self.base_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe_name}"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.src_dir = self.project_dir / "src"
        self.data_dir = self.project_dir / "data"
        self.docs_dir = self.project_dir / "docs"
        self.logs_dir = self.project_dir / "logs"
        for d in (self.src_dir, self.data_dir, self.docs_dir, self.logs_dir):
            d.mkdir(exist_ok=True)

        self.readme_path = self.project_dir / "README.md"
        self.history_path = self.logs_dir / "history.json"
        self.chat_path = self.docs_dir / "chat.md"
        self.state_path = self.logs_dir / "interface_state.json"

        self._init_readme()
        self._init_chat()

    @staticmethod
    def _default_projects_dir() -> Path:
        home = Path.home()
        desktop = home / "Desktop"
        if sys.platform == "linux" and not desktop.exists():
            desktop = home
        return desktop / "SwarmProjects"

    def _init_readme(self):
        readme = textwrap.dedent(f"""\
        # {self.project_name}

        **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Location:** `{self.project_dir}`

        ## Structure
        - `src/` — Generated source code
        - `data/` — JSON/CSV/data outputs
        - `docs/` — Documentation & transcripts
        - `logs/` — Session logs & state snapshots

        ---
        """)
        self.readme_path.write_text(readme, encoding="utf-8")

    def _init_chat(self):
        header = textwrap.dedent(f"""\
        # Chat Transcript — {self.project_name}

        **Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        ---
        """)
        self.chat_path.write_text(header, encoding="utf-8")

    def save_turn(self, role: str, content: str, strategy: str = "", workers: int = 0):
        history = []
        if self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        history.append({
            "role": role,
            "content": content,
            "strategy": strategy,
            "workers": workers,
            "timestamp": datetime.now().isoformat(),
        })
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        md = f"\n## {role.upper()}\n\n"
        if role == "assistant" and strategy:
            md += f"*Strategy: `{strategy}` | Workers: {workers}*  \n"
        md += f"{content}\n\n---\n"
        with self.chat_path.open("a", encoding="utf-8") as f:
            f.write(md)

    def save_code(self, filename: str, content: str, language: str = "python") -> Path:
        if not filename.endswith((".py", ".js", ".ts", ".json", ".md", ".csv", ".sh", ".sql", ".html", ".css")):
            ext_map = {
                "python": ".py", "javascript": ".js", "typescript": ".ts",
                "json": ".json", "markdown": ".md", "csv": ".csv",
                "bash": ".sh", "sql": ".sql", "html": ".html", "css": ".css",
            }
            filename += ext_map.get(language.lower(), ".txt")
        path = self.src_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def save_data(self, filename: str, content: str) -> Path:
        path = self.data_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def save_state(self, state: dict):
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    @classmethod
    def list_projects(cls, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        base = base_dir or cls._default_projects_dir()
        if not base.exists():
            return []
        projects = []
        for folder in sorted(base.iterdir(), reverse=True):
            if folder.is_dir():
                readme = folder / "README.md"
                name = folder.name
                created = "Unknown"
                if readme.exists():
                    try:
                        for line in readme.read_text(encoding="utf-8").splitlines():
                            if line.startswith("**Created:**"):
                                created = line.replace("**Created:**", "").strip()
                                break
                    except Exception:
                        pass
                src_count = len(list((folder / "src").glob("*"))) if (folder / "src").exists() else 0
                projects.append({
                    "folder": folder.name,
                    "name": name,
                    "created": created,
                    "path": str(folder),
                    "files": src_count,
                })
        return projects


class CodeExtractor:
    """Extracts code blocks from LLM responses and auto-saves them."""

    LANG_MAP = {
        "python": "py", "javascript": "js", "typescript": "ts",
        "bash": "sh", "shell": "sh", "json": "json", "sql": "sql",
        "html": "html", "css": "css", "markdown": "md", "csv": "csv",
    }

    @classmethod
    def extract(cls, text: str) -> List[Dict[str, str]]:
        blocks = []
        pattern = r"```(\w+)?\n(.*?)```"
        for match in re.finditer(pattern, text, re.DOTALL):
            lang = (match.group(1) or "text").lower()
            content = match.group(2).strip()
            if len(content) < 20:
                continue
            filename = cls._guess_filename(content, lang)
            blocks.append({"language": lang, "filename": filename, "content": content})
        return blocks

    @classmethod
    def _guess_filename(cls, content: str, lang: str) -> str:
        m = re.search(r'[#//]\s*filename:\s*([^\s]+)', content, re.IGNORECASE)
        if m:
            return m.group(1)
        if lang == "python":
            m = re.search(r'^(?:class|def)\s+(\w+)', content, re.MULTILINE)
            if m:
                return f"{m.group(1)}.py"
            return "script.py"
        if lang in ("javascript", "typescript"):
            m = re.search(r'(?:function|class|const|let)\s+(\w+)', content)
            if m:
                return f"{m.group(1)}.{'ts' if lang == 'typescript' else 'js'}"
            return "script.js"
        if lang == "json":
            return "data.json"
        if lang == "sql":
            return "query.sql"
        ext = cls.LANG_MAP.get(lang, "txt")
        return f"generated.{ext}"