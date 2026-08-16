# ui.py
import re


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"

    @classmethod
    def strip(cls, text: str) -> str:
        return re.sub(r"\033\[[0-9;]*m", "", text)


class UI:
    WIDTH = 64

    @staticmethod
    def banner(title: str, subtitle: str = "") -> str:
        lines = [
            "",
            f"{Style.BOLD}{Style.CYAN}╔{'═' * UI.WIDTH}╗{Style.RESET}",
            f"{Style.BOLD}{Style.CYAN}║{Style.RESET}  {Style.BOLD}{title:<{UI.WIDTH - 2}}{Style.RESET}{Style.CYAN}  ║{Style.RESET}",
        ]
        if subtitle:
            lines.append(
                f"{Style.CYAN}║{Style.RESET}  {Style.DIM}{subtitle:<{UI.WIDTH - 2}}{Style.RESET}{Style.CYAN}  ║{Style.RESET}"
            )
        lines.append(f"{Style.BOLD}{Style.CYAN}╚{'═' * UI.WIDTH}╝{Style.RESET}")
        return "\n".join(lines)

    @staticmethod
    def status(label: str, message: str, color: str = Style.CYAN) -> str:
        return f"  {Style.DIM}[{Style.RESET}{color}{label:>10}{Style.RESET}{Style.DIM}]{Style.RESET}  {message}"

    @staticmethod
    def section(title: str) -> str:
        return f"\n{Style.BOLD}{Style.BLUE}▸ {title}{Style.RESET}\n"

    @staticmethod
    def artifact_saved(path) -> str:
        from pathlib import Path
        rel = path.relative_to(Path.home()) if path.is_relative_to(Path.home()) else path
        return f"  {Style.GREEN}✓{Style.RESET}  Saved {Style.UNDERLINE}{rel}{Style.RESET}"

    @staticmethod
    def divider() -> str:
        return f"{Style.DIM}{'─' * (UI.WIDTH + 2)}{Style.RESET}"

    @staticmethod
    def menu_item(index: int, label: str, desc: str = "", disabled: bool = False) -> str:
        if disabled:
            return f"  {Style.DIM}[{index}] {label:<20} {desc}{Style.RESET}"
        return f"  {Style.BOLD}{Style.CYAN}[{index}]{Style.RESET} {Style.BOLD}{label:<20}{Style.RESET} {Style.DIM}{desc}{Style.RESET}"

    @staticmethod
    def prompt(text: str) -> str:
        return f"\n{Style.BOLD}{Style.CYAN}?{Style.RESET}  {Style.BOLD}{text}{Style.RESET}  "

    @staticmethod
    def info(text: str) -> str:
        return f"  {Style.DIM}ℹ {text}{Style.RESET}"