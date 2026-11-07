import ast
from pathlib import Path

from rscm import names
from rscm._lib.core import names as core_names

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "python" / "rscm"
STUB = PACKAGE / "_lib" / "core" / "names.pyi"

# Single words such as `Emissions` also appear in prose, so only compound names count.
VALUES = {
    value: key
    for key, value in vars(names).items()
    if key.isupper() and isinstance(value, str) and (" " in value or "|" in value)
}

ALLOWED = (
    # Self-contained example that shows the literal names a component declares.
    "example_components.py",
    # MAGICC7 to RSCM rewrite rules, which match partial names.
    "names.py",
)


def _docstrings(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for statement in body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                found.add(id(statement.value))
    return found


def _constant_for(literal: str) -> str | None:
    literal = literal.removeprefix("Prescribed ")
    if literal in VALUES:
        return VALUES[literal]
    for value, key in VALUES.items():
        if literal.startswith(f"{value}|"):
            return key
    return None


def test_python_uses_name_constants():
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in ALLOWED:
            continue
        tree = ast.parse(path.read_text())
        docstrings = _docstrings(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and (key := _constant_for(node.value))
            ):
                hits.append(f"{relative}:{node.lineno} {node.value!r} -> names.{key}")
    assert not hits


def test_names_stub_values_match_runtime():
    tree = ast.parse(STUB.read_text())
    stub = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    runtime = {
        key: value
        for key, value in vars(core_names).items()
        if key.isupper() and isinstance(value, str)
    }
    assert stub == runtime
