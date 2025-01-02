"""Readme copying utility function."""

from pathlib import Path

import mkdocs_gen_files

with Path("README.md").open("rb") as src, mkdocs_gen_files.open("README.md", "wb") as dst:
    dst.write(src.read())
