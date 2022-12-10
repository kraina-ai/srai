from types import ModuleType
from typing import Any, Callable, List, Optional


# Inspired by:
# https://github.com/pandas-dev/pandas/blob/main/pandas/compat/_optional.py
# https://stackoverflow.com/questions/563022/whats-python-good-practice-for-importing-and-offering-optional-features
def import_optional_dependency(
    dependency_group: str, module: str, name: Optional[str] = None, error: str = "raise"
) -> Optional[ModuleType]:
    """
    Import a module or a element from the module.

    Args:
        dependency_group (str): Name of dependency group where dependency is defined.
            Helps communicate a proper error message to the end user.
        module (str): Name of a module.
        name (str, optional): Name of element from the module. If none, returns whole module.
            Otherwise returns a submodule found using a given name. Defaults to None.
        error (str, {'raise', 'warn', 'ignore'}): Information what to do when module hasn't
            been found. Can `raise` an error, write a warning (`warn`) or `ignore` missing module.

    Raises:
        ImportError: When required dependency is not installed.

    Returns:
        Optional[ModuleType]: Module or submodule imported using a name. None if not found.

    """
    assert error in {"raise", "warn", "ignore"}
    import importlib

    try:
        imported_module = importlib.import_module(module)
        return imported_module if name is None else getattr(imported_module, name)
    except ImportError as e:
        error_msg = (
            f'Missing optional dependency "{e.name}". Please install required packages using '
            f"`pip install srai[{dependency_group}]`."
        )
        if error == "raise":
            raise ImportError(error_msg)
        if error == "warn":
            import warnings

            warnings.warn(f"{error_msg} Skipping import.")
    return None


def check_for_dependencies(
    dependency_group: str, modules: List[str]
) -> Callable[..., Callable[..., Any]]:
    """
    Decorator allowing import of required dependencies.

    Args:
        dependency_group (str): Name of optional group that contains dependencies.
        modules (List[str]): List of module names that are expected to be imported.

    Returns:
        Callable[..., Callable[..., Any]]: Decorator for a function.

    """

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for module in modules:
                import_optional_dependency(dependency_group=dependency_group, module=module)
            func(*args, **kwargs)

        return wrapper

    return decorator
