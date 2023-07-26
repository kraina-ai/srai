"""Generate the code reference pages."""

import os
from pathlib import Path

import mkdocs_gen_files

FULL_API_DIRECTORY_PATH = Path("full_api")

nav = mkdocs_gen_files.Nav()

is_dev = os.getenv("MKDOCS_DEV", "true").lower() == "true"

if is_dev:
    for path in sorted(Path("srai").rglob("*.py")):
        module_path = path.relative_to("srai").with_suffix("")
        doc_path = path.relative_to("srai").with_suffix(".md")
        full_doc_path = Path(FULL_API_DIRECTORY_PATH, doc_path)

        parts = list(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        if not parts:
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            identifier = ".".join(parts)
            print("::: " + identifier, file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    with mkdocs_gen_files.open(f"{FULL_API_DIRECTORY_PATH}/README.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())
