"""README copying utility function."""
from pathlib import Path

import mkdocs_gen_files

readme_path = Path("README.md")

with readme_path.open("rb") as src, mkdocs_gen_files.open(readme_path, "wb") as dst:
    dst.write(src.read())
