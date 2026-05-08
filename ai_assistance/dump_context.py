"""
dump_context.py
---------------
Run this script from the root of your Renku project to generate two context files:

    python dump_context.py

Output files (written next to this script):
    context_arborescence.txt  — file tree with module-level docstring for each .py file
    context_code.txt          — all .py files concatenated in full
"""

import ast
import pathlib

# ---------------------------------------------------------------------------
# Config — adjust ROOT if you run from a subdirectory
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent.resolve()

# Files and directories to ignore entirely
IGNORE_DIRS  = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".renku", "results", "tempodata"}
IGNORE_FILES = {"dump_context.py"}

# Output filenames
OUT_ARBO = pathlib.Path(__file__).parent / "context_arborescence.txt"
OUT_CODE = pathlib.Path(__file__).parent / "context_code.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_py_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return all .py files under root, sorted, excluding ignored paths."""
    files = []
    for p in sorted(root.rglob("*.py")):
        # Skip ignored directories anywhere in the path
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.name in IGNORE_FILES:
            continue
        files.append(p)
    return files


def get_module_docstring(path: pathlib.Path) -> str:
    """Extract the module-level docstring from a .py file, or '' if none."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        return docstring.strip() if docstring else ""
    except SyntaxError:
        return "(syntax error — could not parse)"
    except Exception:
        return ""


def build_tree_lines(root: pathlib.Path, py_files: list[pathlib.Path]) -> list[str]:
    """
    Build a human-readable tree of the project.
    For each .py file, append its module docstring (first line only).
    Non-py files are listed without docstring.
    """
    lines = []
    py_set = set(py_files)

    def _recurse(directory: pathlib.Path, prefix: str):
        children = sorted(
            [c for c in directory.iterdir()
             if c.name not in IGNORE_DIRS and not c.name.startswith(".")],
            key=lambda p: (p.is_file(), p.name)   # dirs first, then files
        )
        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            extension = "    " if i == len(children) - 1 else "│   "

            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                _recurse(child, prefix + extension)
            else:
                if child in py_set:
                    doc = get_module_docstring(child)
                    first_line = doc.splitlines()[0] if doc else ""
                    annotation = f"  # {first_line}" if first_line else ""
                    lines.append(f"{prefix}{connector}{child.name}{annotation}")
                else:
                    lines.append(f"{prefix}{connector}{child.name}")

    lines.append(f"{root.name}/")
    _recurse(root, "")
    return lines


def write_arborescence(root: pathlib.Path, py_files: list[pathlib.Path], out: pathlib.Path):
    """Write context_arborescence.txt."""
    sep = "=" * 70
    header = [
        sep,
        "PROJECT ARBORESCENCE WITH MODULE DESCRIPTIONS",
        f"Root : {root}",
        sep,
        "",
    ]

    tree_lines = build_tree_lines(root, py_files)

    detail_lines = [
        "",
        sep,
        "DETAILED DOCSTRINGS PER FILE",
        sep,
    ]
    for p in py_files:
        rel = p.relative_to(root)
        doc = get_module_docstring(p)
        detail_lines.append(f"\n# {rel}")
        detail_lines.append('"""')
        detail_lines.append(doc if doc else "(no module docstring)")
        detail_lines.append('"""')

    content = "\n".join(header + tree_lines + detail_lines)
    out.write_text(content, encoding="utf-8")
    print(f"✅  Written: {out}  ({out.stat().st_size // 1024} KB)")


def write_full_code(root: pathlib.Path, py_files: list[pathlib.Path], out: pathlib.Path):
    """Write context_code.txt — all .py files concatenated."""
    sep = "=" * 70
    blocks = [
        sep,
        "FULL PROJECT CODE — ALL .py FILES",
        f"Root : {root}",
        f"Files: {len(py_files)}",
        sep,
    ]

    for p in py_files:
        rel = p.relative_to(root)
        blocks.append(f"\n\n{'#' * 70}")
        blocks.append(f"# FILE: {rel}")
        blocks.append(f"{'#' * 70}\n")
        try:
            blocks.append(p.read_text(encoding="utf-8"))
        except Exception as e:
            blocks.append(f"# ERROR reading file: {e}")

    content = "\n".join(blocks)
    out.write_text(content, encoding="utf-8")
    print(f"✅  Written: {out}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    py_files = collect_py_files(ROOT)
    print(f"Found {len(py_files)} Python files under {ROOT}\n")

    write_arborescence(ROOT, py_files, OUT_ARBO)
    write_full_code(ROOT, py_files, OUT_CODE)

    print("\nDone. Upload these two files to your Claude project:")
    print(f"  • {OUT_ARBO.name}")
    print(f"  • {OUT_CODE.name}")