"""Utility function for automatic api documentation generation."""
from pathlib import Path

import mkdocs_gen_files

MODULE_DIRECTORY_PATH = Path("srai")
API_DIRECTORY_PATH = Path("api")


def write_file(file_path: Path) -> None:
    """
    Writes dummy file with reference to a module.

    Args:
        file_path: Current file path.
    """
    root_path = i.relative_to(MODULE_DIRECTORY_PATH)
    dst_path = (API_DIRECTORY_PATH / i.parts[-2]).with_suffix(".md")
    print(f"Adding {root_path} to API")
    with mkdocs_gen_files.open(dst_path, "w") as dst:
        dst.write(f"::: {'.'.join(list(i.parts)[:-1])}")


for i in MODULE_DIRECTORY_PATH.glob("*/__init__.py"):
    if i.is_file() and len(i.relative_to(".").parents) <= 3:
        write_file(i)
