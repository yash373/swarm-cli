from ollama import chat


def respond(model, query):
    stream = chat(
    model=model,
    messages=[{'role': 'user', 'content': query}],
    think=True,
    stream=True,
    )

    in_thinking = False

    for chunk in stream:
        if chunk.message.thinking and not in_thinking:
            in_thinking = True
            print('Thinking:\n', end='')

        if chunk.message.thinking:
            print(chunk.message.thinking, end='')
        elif chunk.message.content:
            if in_thinking:
                print('\n\nAnswer:\n', end='')
                in_thinking = False
            print(chunk.message.content, end='')
            
respond('gpt-oss:20b', 'caluclate 6+5')