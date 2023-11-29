from enum import Enum
from types import ModuleType
from typing import Optional


class ImportErrorHandle(Enum):
    """List of values defining how to handle import errors."""

    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


# Inspired by:
# https://github.com/pandas-dev/pandas/blob/main/pandas/compat/_optional.py
# https://stackoverflow.com/questions/563022/whats-python-good-practice-for-importing-and-offering-optional-features
def import_optional_dependency(
    dependency_group: str,
    module: str,
    name: Optional[str] = None,
    error: ImportErrorHandle = ImportErrorHandle.RAISE,
) -> Optional[ModuleType]:
    """
    Import a module or a element from the module.

    Args:
        dependency_group (str): Name of dependency group where dependency is defined.
            Helps communicate a proper error message to the end user.
        module (str): Name of a module.
        name (str, optional): Name of element from the module. If none, returns whole module.
            Otherwise returns a submodule found using a given name. Defaults to None.
        error (ErrorHandle, {'raise', 'warn', 'ignore'}): Information what to do when module hasn't
            been found. Can `raise` an error, write a warning (`warn`) or `ignore` missing module.
            Defaults to ErrorHandle.RAISE.

    Raises:
        ImportError: When required dependency is not installed.

    Returns:
        Optional[ModuleType]: Module or submodule imported using a name. None if not found.
    """
    assert error in ImportErrorHandle.__members__.values()
    import importlib

    try:
        imported_module = importlib.import_module(module)
        return imported_module if name is None else getattr(imported_module, name)
    except ImportError as exc:
        error_msg = (
            f'Missing optional dependency "{module}". Please install required packages using '
            f"`pip install srai[{dependency_group}]`."
        )
        if error == ImportErrorHandle.RAISE:
            raise ImportError(error_msg) from exc
        if error == ImportErrorHandle.WARN:
            import warnings

            warnings.warn(f"{error_msg} Skipping import.", ImportWarning, stacklevel=2)
    return None


def import_optional_dependencies(
    dependency_group: str, modules: list[str], error: ImportErrorHandle = ImportErrorHandle.RAISE
) -> None:
    """
    Import list of optional dependencies.

    Args:
        dependency_group (str): Name of optional group that contains dependencies.
        modules (List[str]): List of module names that are expected to be imported.
        error (ErrorHandle, {'raise', 'warn', 'ignore'}): Information what to do when any of
            the modules hasn't been found. Defaults to ErrorHandle.RAISE.
    """
    for module in modules:
        import_optional_dependency(dependency_group=dependency_group, module=module, error=error)
