"""Utility function for automatic api documentation generation."""
import ast
from pathlib import Path
from typing import List, Tuple

import mkdocs_gen_files

MODULE_DIRECTORY_PATH = Path("srai")
API_DIRECTORY_PATH = Path("api")

nav = mkdocs_gen_files.Nav()


def write_file(file_path: Path) -> None:
    """
    Writes dummy file with reference to a module.

    Args:
        file_path (Path): Current file path.
    """
    root_path = file_path.relative_to(MODULE_DIRECTORY_PATH)
    print(f"Loading imports from {root_path}")
    classes, functions = _read_imports_from_file(file_path)

    if classes:
        module_nav = mkdocs_gen_files.Nav()

        for imported_class in classes:
            dst_path = API_DIRECTORY_PATH / file_path.parts[-2] / imported_class
            parts = [*list(file_path.parts)[:-1], imported_class]
            identifier = ".".join(parts)
            print(f"Adding {identifier} to API")
            with mkdocs_gen_files.open(dst_path.with_suffix(".md"), "w") as dst:
                dst.write(f"::: {identifier}")

            nav[parts[1:]] = Path(*dst_path.parts[1:]).as_posix()
            module_nav[imported_class] = Path(*dst_path.parts[2:]).as_posix()

        dst_path = API_DIRECTORY_PATH / file_path.parts[-2]
        with mkdocs_gen_files.open((dst_path / "index").with_suffix(".md"), "a") as dst:
            print(dst_path.parts[-1])
            dst.write(f"::: {dst_path.parts[-1]}\n")
            dst.write("    options:\n")
            dst.write("      members: false\n")
            dst.writelines(module_nav.build_literate_nav())

    if functions:
        dst_path = API_DIRECTORY_PATH / file_path.parts[-2]
        with mkdocs_gen_files.open(dst_path.with_suffix(".md"), "a") as dst:
            for imported_function in functions:
                parts = [*list(file_path.parts)[:-1], imported_function]
                identifier = ".".join(parts)
                print(f"Adding {identifier} to API")
                dst.write(f"## `{imported_function}`\n")
                dst.write(f"::: {identifier}\n")

                nav[parts[1:]] = Path(*dst_path.parts[1:], f"#{imported_function}").as_posix()

    if classes or functions:
        nav[list(file_path.parts)[1:-1]] = Path(file_path.parts[-2]).as_posix()


def _read_imports_from_file(file_path: Path) -> Tuple[List[str], List[str]]:
    st = ast.parse(file_path.read_text())

    modules_imports = [stmt for stmt in st.body if isinstance(stmt, ast.ImportFrom)]
    imports = [alias.name for stmt in modules_imports for alias in stmt.names]

    classes = [i for i in imports if _is_camel_case(i)]
    functions = [i for i in imports if not _is_camel_case(i)]
    return classes, functions


def _is_camel_case(s: str) -> bool:
    return s != s.lower() and s != s.upper() and "_" not in s


for i in MODULE_DIRECTORY_PATH.glob("*/__init__.py"):
    if i.is_file() and len(i.relative_to(".").parents) <= 3:
        write_file(i)

with mkdocs_gen_files.open("api/README.md", "a") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
