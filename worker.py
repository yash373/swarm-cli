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
    def __init__(self, model, exec_timeout_seconds: int = 10):
        self.model = model
        self.exec_timeout_seconds = exec_timeout_seconds

        # Persistent namespace so variables/imports/functions defined
        # in one execute_python call are still available in the next
        # one - this makes it behave like a real REPL across the
        # whole conversation, not a fresh interpreter every call.
        self._python_globals = {"__builtins__": __builtins__}

        def execute_python(code: str) -> str:
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
            Returns:
                Combined stdout output from running the code, or the
                error traceback if the code raised an exception, or a
                timeout message if it ran too long.
            """
            output = io.StringIO()
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.exec_timeout_seconds)
            try:
                with contextlib.redirect_stdout(output):
                    exec(code, self._python_globals)
            except _ExecutionTimeout:
                output.write(
                    f"\nError: execution timed out after {self.exec_timeout_seconds}s"
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
            Install a Python package with pip so it becomes importable
            in subsequent execute_python calls. Use this whenever code
            in execute_python fails with ModuleNotFoundError - install
            the missing package, then retry the code.
            Args:
                package_name: The pip package name to install, e.g.
                    "pymupdf" or "duckduckgo-search". Do not include a
                    version pin unless one is specifically required.
            Returns:
                A success message with pip's output, or an error
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
                    timeout=120,
                )
            except FileNotFoundError:
                return (
                    "Error: 'uv' was not found on PATH. Install it "
                    "(https://docs.astral.sh/uv/) or adjust install_package "
                    "to use pip instead."
                )
            except subprocess.TimeoutExpired:
                return f"Error: installing '{package_name}' timed out after 120s"
            except Exception as e:
                return f"Error installing '{package_name}': {e}"

            output = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                return f"Error installing '{package_name}':\n{output}"
            return f"Successfully installed '{package_name}'.\n{output}".strip()

        # Two tools: one to run code, one to install what that code needs.
        self.available_functions = {
            "execute_python": execute_python,
            "install_package": install_package,
        }

    SYSTEM_PROMPT = (
        "You have access to an execute_python tool that runs real Python code"
        "with the full standard library. For any arithmetic beyond simple "
        "single-digit operations - this includes multiplication, division, "
        "roots, trig, statistics, or anything with decimals - you must use "
        "execute_python to compute it. Do not calculate by hand in your "
        "reasoning, even approximately. Hand calculation is unreliable and "
        "not acceptable for this task. If a problem has multiple steps, use "
        "the tool for each step, and store intermediate results in variables "
        "you can reuse in later calls, since state persists across calls.\n\n"
        "You also have an install_package tool. If execute_python fails with "
        "ModuleNotFoundError (or ImportError) for a package that is not part "
        "of the standard library, call install_package with that package's "
        "pip name, then retry the exact same code in execute_python. Only "
        "install a package after seeing this specific error - do not "
        "preemptively install things you have not confirmed are missing."
        "Some packages are already installed, so only install what is actually needed."
        "Installed packages include: requests duckduckgo-search beautifulsoup4 pypdf pymupdf reportlab pandas numpy matplotlib sympy"
        "Note: THE CURRENT YEAR IS 2026 but your memory is only updated up till 2023 but you can use the internet to get more recent information."
        "In cicumstances when you cannot find information on a problem publically, attempt to find a pytohn package or an API that can help you get the information you need."
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
                options={"num_ctx": 16384},
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