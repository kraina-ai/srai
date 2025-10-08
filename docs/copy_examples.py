"""Examples copying utility function."""

from pathlib import Path

import mkdocs_gen_files
import nbformat

EXAMPLES_DIRECTORY_PATH = Path("examples")
DATASETS_DIRECTORY_NAME = "datasets"

RUN_IN_COLAB_CELL_MARKDOWN = """
Run this notebook in Google Colab:

<a target="_blank" href="https://colab.research.google.com/github/kraina-ai/srai/blob/main/{relative_file_path}">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Remember to install the `srai` library before running the notebook:
```python
%pip install srai[all]
```
"""


def write_file(file_path: Path) -> None:
    """
    Copies file from examples directory into mkdocs scope.

    Args:
        file_path (Path): Current file path.
    """
    root_path = file_path.relative_to(".")
    destination_path = root_path

    if file_path.parts[1] == DATASETS_DIRECTORY_NAME:
        destination_path = Path(DATASETS_DIRECTORY_NAME) / Path(*file_path.parts[2:])

    print(f"Copying {file_path} file to {destination_path}")
    with root_path.open("r") as src, mkdocs_gen_files.open(destination_path, "w") as dst:
        if root_path.suffix != ".ipynb":
            dst.write(src.read())
            return

        nb = nbformat.read(src, as_version=4)
        has_google_colab_link = any(
            "colab.research.google.com" in cell.get("source", "") for cell in nb["cells"]
        )
        if not has_google_colab_link:
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    nb["cells"][i:] = (
                        nbformat.v4.new_markdown_cell(
                            RUN_IN_COLAB_CELL_MARKDOWN.format(relative_file_path=root_path)
                        ),
                        *nb["cells"][i:],
                    )
                    break
        else:
            print("Google Colab link already present in the notebook.")

        nbformat.write(nb, dst)


banned_directories = [
    "cache",
    "files",
    "example_files",
    "__pycache__",
    "lightning_logs",
    ".ruff_cache",
    # exclude long running notebooks
    "benchmark",
]
banned_extensions = [
    ".pbf",
    ".parquet",
    ".json",
    ".geojson",
    ".pt",
    ".toml",
    ".pkl",
    ".duckdb",
    ".png",
]
banned_filenames = [".DS_Store"]
for i in EXAMPLES_DIRECTORY_PATH.glob("**/*"):
    if i.is_file():
        should_copy = True

        if i.name in banned_filenames:
            should_copy = False

        for banned_directory in banned_directories:
            if banned_directory in i.parts:
                should_copy = False
                break

        for banned_extension in banned_extensions:
            if banned_extension in i.suffixes:
                should_copy = False
                break

        if should_copy:
            try:
                write_file(i)
            except:
                print(i)
                raise
