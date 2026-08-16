from ollama import chat, ChatResponse
from tools.temperature import get_temperature
from tools import arithmetic

class Worker:
    def __init__(self, model):
        self.model = model

        # List of available functions that the model can call
        self.available_functions = {
            "add": arithmetic.add,
            "multiply": arithmetic.multiply,
            "get_temperature": get_temperature,
        }

    def add(self, a: int, b: int) -> int:
        """
        Add two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The sum of the two numbers.
        """
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """
        Multiply two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The product of the two numbers.
        """
        return a * b

    def respond(self, query: str):
        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]

        while True:
            response: ChatResponse = chat(
                model=self.model,
                messages=messages,
                tools=self.available_functions.values(),
                think=True,
            )

            # Add the assistant's response to the conversation
            messages.append(response.message)

            print("Thinking:", response.message.thinking)
            print("Content:", response.message.content)

            # No tool calls -> model has finished
            if not response.message.tool_calls:
                break

            # Execute requested tools
            for tc in response.message.tool_calls:
                function_name = tc.function.name
                arguments = tc.function.arguments

                if function_name not in self.available_functions:
                    print(f"Unknown function: {function_name}")
                    continue

                print(
                    f"Calling {function_name} "
                    f"with arguments {arguments}"
                )

                function = self.available_functions[function_name]
                result = function(**arguments)

                print(f"Result: {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": function_name,
                        "content": str(result),
                    }
                )

        return response.message.content