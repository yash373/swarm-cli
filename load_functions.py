import importlib
import inspect


def load_functions(module_name):
    """
    Load all functions actually defined in a module (as opposed to
    anything merely callable, which would also sweep in imported
    modules, typing aliases, classes, etc.).

    Args:
        module_name: Dotted path of the module to load functions from,
            e.g. "tools.arithmetic".

    Returns:
        A dict mapping function name -> function object.
    """
    module = importlib.import_module(module_name)
    functions = {}

    for func_name in dir(module):
        if func_name.startswith("__"):
            continue

        func = getattr(module, func_name)

        # Only real functions defined *in this module* — excludes
        # imported modules (e.g. `math`), typing constructs (e.g.
        # `List`, `Union`), classes, and anything imported from
        # elsewhere that happens to be callable.
        if inspect.isfunction(func) and func.__module__ == module.__name__:
            functions[func_name] = func

    return functions