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
    with open(root_path, "r") as src, mkdocs_gen_files.open(root_path, "w") as dst:
        dst.writelines(src.readlines())


for i in EXAMPLES_DIRECTORY_PATH.glob("**/*"):
    if i.is_file():
        write_file(i)
