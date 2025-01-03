"""Readme copying utility function."""

from pathlib import Path

import mkdocs_gen_files

with Path("README.md").open("rb") as src, mkdocs_gen_files.open("README.md", "wb") as dst:
    dst.write(src.read())

with (
    Path("CHANGELOG.md").open("rb") as src,
    mkdocs_gen_files.open("releases/CHANGELOG.md", "wb") as dst,
):
    dst.write(src.read())

with (
    Path("CONTRIBUTING.md").open("rb") as src,
    mkdocs_gen_files.open("CONTRIBUTING.md", "wb") as dst,
):
    dst.write(src.read())
