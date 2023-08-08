"""Utility function for automatic api documentation generation."""
import ast
from pathlib import Path
from typing import Any, List, Tuple, cast

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
    classes, functions, module_docstring = _read_imports_from_file(file_path)

    is_module = len(root_path.parts) == 1

    operational_path = file_path
    if is_module:
        operational_path = file_path.parent / file_path.stem / "__init__.py"

    if classes or functions:
        dst_path = API_DIRECTORY_PATH / operational_path.parts[-2]
        dst_full_path = (dst_path / "README").with_suffix(".md")
        with mkdocs_gen_files.open(dst_full_path, "a") as dst:
            dst.write(f"# {dst_path.parts[-1].capitalize()}\n")

    if classes:
        module_nav = mkdocs_gen_files.Nav()

        for imported_class in classes:
            dst_path = API_DIRECTORY_PATH / operational_path.parts[-2] / imported_class
            parts = [*list(operational_path.parts)[:-1], imported_class]
            identifier = ".".join(parts)
            print(f"[Class] Adding {identifier} to API")
            with mkdocs_gen_files.open(dst_path.with_suffix(".md"), "w") as dst:
                dst.write(f"::: {identifier}")

            nav[parts[1:]] = Path(*dst_path.parts[1:]).as_posix()
            module_nav[imported_class] = Path(*dst_path.parts[2:]).as_posix()

        dst_path = API_DIRECTORY_PATH / operational_path.parts[-2]
        dst_full_path = (dst_path / "README").with_suffix(".md")
        with mkdocs_gen_files.open(dst_full_path, "a") as dst:
            dst.write(f"::: {dst_path.parts[-1]}\n")
            dst.write("    options:\n")
            dst.write("      members: false\n")
            dst.write("## Classes\n")
            dst.writelines(module_nav.build_literate_nav())

    if functions:
        dst_path = API_DIRECTORY_PATH / operational_path.parts[-2]
        dst_full_path = (dst_path / "README").with_suffix(".md")
        with mkdocs_gen_files.open(dst_full_path, "a") as dst:
            if not classes:
                dst.write(f"{module_docstring}\n")

            dst.write("## Functions\n")
            for imported_function in functions:
                parts = [*list(operational_path.parts)[:-1], imported_function]
                identifier = ".".join(parts)
                print(f"[Function] Adding {identifier} to API")
                dst.write(f"### `{imported_function}`\n")
                dst.write(f"::: {identifier}\n")

                nav[parts[1:]] = Path(*dst_path.parts[1:], f"#{imported_function}").as_posix()

    if classes or functions:
        key = list(operational_path.parts)[1:-1]
        nav[key] = Path(operational_path.parts[-2]).as_posix()


def _read_imports_from_file(file_path: Path) -> Tuple[List[str], List[str], str]:
    st = ast.parse(file_path.read_text())

    module_docstring = ""
    st_expression = [stmt for stmt in st.body if isinstance(stmt, ast.Expr)]
    if st_expression:
        module_docstring = cast(Any, st_expression[0]).value.value

    st_all_definition = [
        stmt
        for stmt in st.body
        if isinstance(stmt, ast.Assign) and cast(Any, stmt.targets[0]).id == "__all__"
    ]
    if not st_all_definition:
        return [], [], ""

    module_all_definition = [
        definition.value for definition in cast(Any, st_all_definition[0]).value.elts
    ]

    classes: List[str] = []
    functions: List[str] = []

    # Content

    classes.extend(
        set(
            stmt.name
            for stmt in st.body
            if isinstance(stmt, ast.ClassDef) and stmt.name in module_all_definition
        )
    )
    functions.extend(
        set(
            stmt.name
            for stmt in st.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name in module_all_definition
        )
    )

    # Imports

    modules_imports = [stmt for stmt in st.body if isinstance(stmt, ast.ImportFrom)]
    imports = [alias.name for stmt in modules_imports for alias in stmt.names]

    classes.extend(i for i in imports if _is_camel_case(i) and i in module_all_definition)
    functions.extend(i for i in imports if not _is_camel_case(i) and i in module_all_definition)

    return classes, functions, module_docstring


def _is_camel_case(s: str) -> bool:
    return s != s.lower() and s != s.upper() and "_" not in s


for i in MODULE_DIRECTORY_PATH.glob("*/__init__.py"):
    if i.is_file() and len(i.relative_to(".").parents) <= 3:
        write_file(i)

for i in MODULE_DIRECTORY_PATH.glob("*.py"):
    if i.is_file() and len(i.relative_to(".").parents) <= 3:
        write_file(i)

with mkdocs_gen_files.open(f"{API_DIRECTORY_PATH}/README.md", "a") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
