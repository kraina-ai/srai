"""Examples copying utility function."""

from pathlib import Path

import mkdocs_gen_files

EXAMPLES_DIRECTORY_PATH = Path("examples")


def write_file(file_path: Path) -> None:
    """
    Copies file from examples directory into mkdocs scope.

    Args:
        file_path (Path): Current file path.
    """
    root_path = file_path.relative_to(".")
    print(f"Copying {root_path} file to {root_path}")
    with root_path.open("rb") as src, mkdocs_gen_files.open(root_path, "wb") as dst:
        dst.write(src.read())


for i in EXAMPLES_DIRECTORY_PATH.glob("**/*"):
    if i.is_file() and "cache" not in i.parts:
        write_file(i)
