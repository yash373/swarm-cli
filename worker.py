import contextlib
import io
import signal
import subprocess
import sys
import traceback

from ollama import chat, ChatResponse


class _ExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ExecutionTimeout("Execution timed out")


class Worker:
    def __init__(
        self,
        model,
        default_timeout_seconds: int = 30,
        max_timeout_seconds: int = 1800,
        num_ctx: int = 32768,
    ):
        self.model = model
        self.num_ctx = num_ctx

        # Per-call timeout is conditional: the model picks how long to
        # allow (default_timeout_seconds if it doesn't specify), but it
        # can never exceed max_timeout_seconds. This lets quick arithmetic
        # stay fast while genuinely long jobs (training a model, scraping
        # many pages) can ask for more time without removing the safety
        # net entirely.
        self.default_timeout_seconds = default_timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds

        # Persistent namespace so variables/imports/functions defined
        # in one execute_python call are still available in the next
        # one - this makes it behave like a real REPL across the
        # whole conversation, not a fresh interpreter every call.
        self._python_globals = {"__builtins__": __builtins__}

        def execute_python(code: str, timeout_seconds: int = 0) -> str:
            """
            Execute a snippet of Python code and return what it printed.
            Has access to the full standard library. Variables, imports,
            and functions persist between calls within this conversation,
            so you can build up state incrementally like a REPL.
            Args:
                code: The Python source code to run. Use print() for
                    any values you want to see in the result - the
                    return value of the last expression is NOT
                    captured automatically.
                timeout_seconds: How long to allow this code to run.
                    Leave as 0 to use the default (30s), which is
                    plenty for calculations, searches, or file work.
                    Raise this explicitly for genuinely long tasks -
                    e.g. training an ML model, scraping many pages, or
                    a large simulation. Capped at 1800s (30 minutes)
                    regardless of what you request.
            Returns:
                Combined stdout output from running the code, or the
                error traceback if the code raised an exception, or a
                timeout message if it ran too long.
            """
            effective_timeout = timeout_seconds if timeout_seconds > 0 else self.default_timeout_seconds
            effective_timeout = min(effective_timeout, self.max_timeout_seconds)

            output = io.StringIO()
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(effective_timeout)
            try:
                with contextlib.redirect_stdout(output):
                    exec(code, self._python_globals)
            except _ExecutionTimeout:
                output.write(
                    f"\nError: execution timed out after {effective_timeout}s. "
                    f"If this task genuinely needs longer, retry with a "
                    f"higher timeout_seconds (up to {self.max_timeout_seconds})."
                )
            except Exception:
                output.write("\n" + traceback.format_exc())
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            result = output.getvalue()
            return result if result.strip() else "(no output - use print() to see results)"

        def install_package(package_name: str) -> str:
            """
            Install a Python package with uv so it becomes importable
            in subsequent execute_python calls. Use this immediately
            whenever a task needs a capability the standard library or
            pre-installed packages don't cover - do not wait to fail
            first, and do not ask for permission.
            Args:
                package_name: The pip package name to install, e.g.
                    "torch" or "xgboost". Do not include a version pin
                    unless one is specifically required.
            Returns:
                A success message with install output, or an error
                message if the install failed or timed out.
            """
            try:
                result = subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--system",
                        "--break-system-packages",
                        package_name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except FileNotFoundError:
                return (
                    "Error: 'uv' was not found on PATH. Install it "
                    "(https://docs.astral.sh/uv/) or adjust install_package "
                    "to use pip instead."
                )
            except subprocess.TimeoutExpired:
                return f"Error: installing '{package_name}' timed out after 300s"
            except Exception as e:
                return f"Error installing '{package_name}': {e}"

            output = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                return f"Error installing '{package_name}':\n{output}"
            return f"Successfully installed '{package_name}'.\n{output}".strip()

        def fetch_url(url: str) -> str:
            """
            Fetch a web page and return its visible text content, with
            HTML tags, scripts, and styling stripped out. Use this to
            pull data from a specific URL - e.g. a page you found via
            search_web, or a known API endpoint.
            Args:
                url: The full URL to fetch, including scheme
                    (http:// or https://).
            Returns:
                The extracted visible text (truncated to ~8000 characters
                to avoid overwhelming context), or an error message.
            """
            try:
                import requests
                from bs4 import BeautifulSoup
            except ImportError as e:
                return (
                    f"Error: missing dependency ({e}). Call install_package "
                    f"for 'requests' and/or 'beautifulsoup4', then retry."
                )

            try:
                resp = requests.get(
                    url,
                    timeout=15,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ARCHE-agent/1.0)"},
                )
                resp.raise_for_status()
            except Exception as e:
                return f"Error fetching '{url}': {e}"

            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type or "text/plain" in content_type:
                text = resp.text
            else:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
                text = "\n".join(lines)

            if len(text) > 8000:
                text = text[:8000] + "\n...[truncated]"
            return text if text.strip() else "(page fetched but no text content found)"

        def search_web(query: str, max_results: int = 5) -> str:
            """
            Search the web and return a list of result titles, URLs,
            and short snippets. Tries several underlying search engines
            in sequence and returns results from the first one that
            works, since any single engine can rate-limit or block a
            given network. Use this to find pages worth fetching with
            fetch_url, or to answer questions about current
            events/facts beyond your training data.
            Args:
                query: The search query.
                max_results: How many results to return (default 5).
            Returns:
                A formatted list of results, or a diagnostic error
                message if every backend failed.
            """
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS  # deprecated fallback
                except ImportError as e:
                    return (
                        f"Error: missing dependency ({e}). Call install_package "
                        f"for 'ddgs', then retry."
                    )

            backends_to_try = ["duckduckgo", "brave", "mojeek", "bing", "startpage", "yahoo"]
            errors = []
            for backend in backends_to_try:
                try:
                    results = DDGS().text(query, max_results=max_results, backend=backend)
                except Exception as e:
                    errors.append(f"{backend}: {type(e).__name__}: {e}")
                    continue

                if results:
                    formatted = []
                    for i, r in enumerate(results, 1):
                        formatted.append(
                            f"{i}. {r.get('title', '(no title)')}\n"
                            f"   URL: {r.get('href', '')}\n"
                            f"   {r.get('body', '')}"
                        )
                    return "\n\n".join(formatted) + f"\n\n(results via {backend})"

            # Every backend failed or returned nothing - give a diagnostic
            # error, not a silent empty result, and steer the model away
            # from just retrying the identical query forever.
            return (
                f"Error: search failed on all backends ({', '.join(backends_to_try)}) "
                f"for query '{query}'.\nDetails: {'; '.join(errors) if errors else 'no results from any backend'}\n"
                f"Do not retry this exact same query - either try a "
                f"meaningfully different phrasing once, or if you already "
                f"know a likely URL (e.g. a specific site's standings page), "
                f"use fetch_url on it directly instead."
            )

        # Four tools: run code, install what that code needs, and two
        # dedicated shortcuts for the most common internet task (search
        # then fetch) so the model doesn't have to hand-write requests/
        # bs4 boilerplate inside execute_python every single time.
        self.available_functions = {
            "execute_python": execute_python,
            "install_package": install_package,
            "search_web": search_web,
            "fetch_url": fetch_url,
        }

    SYSTEM_PROMPT = (
        "You have access to an execute_python tool that runs real Python "
        "code with the full standard library. For any arithmetic beyond "
        "simple single-digit operations - this includes multiplication, "
        "division, roots, trig, statistics, or anything with decimals - "
        "you must use execute_python to compute it. Do not calculate by "
        "hand in your reasoning, even approximately. Hand calculation is "
        "unreliable and not acceptable for this task. If a problem has "
        "multiple steps, use the tool for each step, and store "
        "intermediate results in variables you can reuse in later calls, "
        "since state persists across calls.\n\n"
        "You also have an install_package tool that installs any package "
        "from PyPI. You are NOT limited to the standard library. If a task "
        "needs web search, PDF creation or editing, HTTP requests, data "
        "analysis, ML model training, image processing, or any other "
        "capability a Python package provides, install that package "
        "immediately and use it. Do not say a task is not possible, out of "
        "scope, or something you cannot do because a package is not "
        "pre-installed - it is always one install_package call away. Do "
        "not ask the user for permission first and do not explain that you "
        "are about to install something - just call install_package, then "
        "proceed with execute_python.\n\n"
        "The following packages are already installed and importable right "
        "now - do NOT call install_package for these, just import and use "
        "them directly: requests, ddgs, bs4 (BeautifulSoup4), "
        "pypdf, fitz (PyMuPDF), reportlab, pandas, numpy, matplotlib, "
        "sympy, sklearn (scikit-learn).\n\n"
        "For training ML models: sklearn covers most classic tasks "
        "(regression, classification, clustering) and is already "
        "installed - use it by default. For deep learning specifically, "
        "install_package('torch') first. For gradient-boosted trees, "
        "install_package('xgboost') first. Training can legitimately take "
        "longer than a normal calculation - pass a higher timeout_seconds "
        "to execute_python for these instead of letting it time out and "
        "retrying blindly.\n\n"
        "You have two dedicated tools for internet tasks: search_web to "
        "find relevant pages, and fetch_url to pull the text content of a "
        "specific page. Prefer these over hand-writing requests/bs4 code "
        "inside execute_python - they are faster and less error-prone. "
        "Only fall back to writing your own scraping code inside "
        "execute_python for something these two tools genuinely can't do "
        "(e.g. a JSON API with a specific query format, or a page needing "
        "JS rendering). search_web already tries several search engines "
        "internally before giving up, so if it returns an error, retrying "
        "the exact same query will not help - either reword it once, or "
        "switch to fetch_url on a specific URL you already suspect is "
        "relevant. Never call search_web more than twice in a row with "
        "near-identical wording.\n\n"
        "Work efficiently. Do not over-deliberate or repeat the same "
        "verification step multiple times - decide on an approach, call "
        "the right tool, and move forward. Prefer a small number of "
        "purposeful tool calls over lengthy step-by-step reasoning in "
        "text. Your job is to finish the task correctly, not to narrate "
        "your thought process at length.\n\n"
        "The current year is 2026, which is after your training data ends. "
        "Do not rely on memory for anything time-sensitive - current "
        "events, prices, current holders of a position, latest versions of "
        "software, or similar. Use search_web and fetch_url for these, and "
        "treat your own memory as a fallback only for stable, unchanging "
        "facts."
    )

    def respond(self, query: str):
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            }
        ]

        final_content = ""

        while True:
            stream = chat(
                model=self.model,
                messages=messages,
                tools=list(self.available_functions.values()),
                think=True,
                stream=True,
                options={"num_ctx": self.num_ctx},
            )

            thinking_buffer = ""
            content_buffer = ""
            tool_calls = None
            printed_thinking_header = False
            printed_content_header = False

            for chunk in stream:
                delta = chunk.message

                if delta.thinking:
                    if not printed_thinking_header:
                        print("\nThinking: ", end="", flush=True)
                        printed_thinking_header = True
                    print(delta.thinking, end="", flush=True)
                    thinking_buffer += delta.thinking

                if delta.content:
                    if not printed_content_header:
                        print("\nContent: ", end="", flush=True)
                        printed_content_header = True
                    print(delta.content, end="", flush=True)
                    content_buffer += delta.content

                if delta.tool_calls:
                    tool_calls = delta.tool_calls

            print()

            assistant_message = {
                "role": "assistant",
                "content": content_buffer,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    for tc in tool_calls
                ]

            messages.append(assistant_message)
            final_content = content_buffer

            if not tool_calls:
                break

            for tc in tool_calls:
                function_name = tc.function.name
                arguments = tc.function.arguments

                if function_name not in self.available_functions:
                    result = f"Error: unknown function '{function_name}'"
                    print(result)
                else:
                    print(f"\nCalling {function_name} with arguments {arguments}")

                    function = self.available_functions[function_name]
                    try:
                        result = function(**arguments)
                    except Exception as e:
                        result = f"Error: {e}"

                    print(f"Result: {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": function_name,
                        "content": str(result),
                    }
                )

        return final_content