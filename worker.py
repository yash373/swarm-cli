import contextlib
import io
import signal
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

        # Only one tool - no more scanning a module for 71 functions.
        self.available_functions = {"execute_python": execute_python}

    def respond(self, query: str):
        messages = [
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