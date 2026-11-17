"""
Report retired identifiers listed in `naming-denylist.txt`.

Checks the files given as arguments, or every tracked file without arguments.
A line containing `naming: allow` is exempt.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
DENYLIST = ROOT / "naming-denylist.txt"
ALLOW_MARKER = "naming: allow"

# Changelogs, proposals and the conventions page quote retired names on purpose.
EXCLUDED_PREFIXES = ("changelog/", "openspec/", "docs/changelog.md")
EXCLUDED_FILES = {
    "naming-denylist.txt",
    "scripts/check_naming.py",
    "docs/developers/naming_conventions.md",
}
# Synonym tables, their tests and MAGICC7 data keep MAGICC7 species tokens.
SYNONYM_PATHS = (
    "crates/rscm-core/src/units/",
    "python/rscm/names.py",
    "tests/test_units.py",
    "docs/notebooks/units_tutorial.py",
    "docs/modules/",
    "tests/regression/data/",
    "python/rscm/magicc/data/magicc7/",
)
SYNONYMS = {r"HFC4310\b"}
# MAGICC7 parameter keys keep MAGICC7's spelling.
MAGICC7_KEY = re.compile(r"""p\[\s*["'][^"']*["']\s*\]""")


def load_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Read the denylist, one regular expression per line."""
    patterns = []
    for raw in DENYLIST.read_text().splitlines():
        source = raw.strip()
        if source and not source.startswith("#"):
            patterns.append((source, re.compile(rf"(?<!\w)(?:{source})(?!\w)")))
    return patterns


def check(path: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    """Return a report line for each denied identifier in `path`."""
    if path.startswith(EXCLUDED_PREFIXES) or path in EXCLUDED_FILES:
        return []
    try:
        text = (ROOT / path).read_text()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return []
    hits = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in raw or (path == "GLOSSARY.md" and "_Avoid_" in raw):
            continue
        line = MAGICC7_KEY.sub("", raw)
        for source, pattern in patterns:
            if path.startswith(SYNONYM_PATHS) and source in SYNONYMS:
                continue
            for match in pattern.finditer(line):
                hits.append(f"{path}:{number}: {match.group()} ({source})")
    return hits


def main(paths: list[str]) -> int:
    """Check `paths`, or every tracked file, and return the exit code."""
    if not paths:
        tracked = subprocess.run(
            ["git", "ls-files"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        paths = tracked.stdout.splitlines()
    patterns = load_patterns()
    hits = [hit for path in paths for hit in check(path, patterns)]
    for hit in hits:
        print(hit)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
