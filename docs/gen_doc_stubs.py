"""
Generate virtual doc files for the mkdocs site.

This script can also be run directly to actually write out those files,
as a preview.

All credit to the creators of:
https://oprypin.github.io/mkdocs-gen-files/
and the docs at:
https://mkdocstrings.github.io/crystal/quickstart/migrate.html
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mkdocs_gen_files
from attrs import define

ROOT_DIR = Path("api")
nav = mkdocs_gen_files.Nav()

# Packages documented by their public re-exports, so their plain modules get no page.
FLATTENED_PACKAGES = {"rscm.config"}

# Per-module mkdocstrings options, merged over the defaults in mkdocs.yml.
DIRECTIVE_OPTIONS: dict[str, dict[str, Any]] = {
    "rscm": {"members": False},
    # run_magicc7 is documented once, on rscm.magicc.full.
    "rscm.magicc": {"filters": ["!^_[^_]", "!^run_magicc7$"]},
    "rscm.magicc.full": {
        "group_by_category": False,
        "members": [
            "run_magicc7",
            "default_parameters",
            "load_inputs",
            "natural_emissions",
            "run_components",
            "default_initial_concentrations",
            "climate_parameters",
            "forcing_agents",
            "prescribed_concentrations",
            "prescribed_forcing",
            "concentration_switch_years",
            "efficacy_climate",
            "ozone_parameters",
            "permafrost_parameters",
            "nitrogen_parameters",
            "nitrogen_inputs",
            "halocarbon_forcing",
            "halocarbon_species",
            "SECTORAL_EMISSIONS",
            "INPUT_FILES",
            "PRESCRIBED_AGENTS",
            "NORMYEAR_PATTERNS",
        ],
    },
}


@define
class PackageInfo:
    """
    Package information used to help us auto-generate the docs
    """

    full_name: str
    stem: str
    summary: str


def write_subpackage_pages(package: object) -> tuple[PackageInfo, ...]:
    """
    Write pages for the sub-packages of a package
    """
    sub_packages = []
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
        # Skip "private" packages
        if name.startswith("_"):
            continue
        if package.__name__ in FLATTENED_PACKAGES and not is_pkg:
            continue
        subpackage_full_name = package.__name__ + "." + name
        sub_package_info = write_module_page(subpackage_full_name)
        sub_packages.append(sub_package_info)

    return tuple(sub_packages)


def get_write_file(package_full_name: str) -> Path:
    """Get directory in which to write the doc file"""
    write_dir = ROOT_DIR
    for sub_dir in package_full_name.split(".")[:-1]:
        write_dir = write_dir / sub_dir

    write_file = write_dir / package_full_name.rsplit(".", maxsplit=1)[-1] / "index.md"

    return write_file


def create_sub_packages_table(sub_packages: Iterable[PackageInfo]) -> str:
    """Create the table summarising the sub-packages"""
    links = [f"[{sp.stem}][{sp.full_name}]" for sp in sub_packages]
    sub_package_header = "Sub-package"
    sub_package_width = max([len(v) for v in [sub_package_header, *links]])

    descriptions = [sp.summary for sp in sub_packages]
    description_header = "Description"
    description_width = max([len(v) for v in [description_header, *descriptions]])

    sp_column = [sub_package_header, *links]
    description_column = [description_header, *descriptions]

    sub_packages_table_l = []
    for i, (sub_package_value, description) in enumerate(
        zip(sp_column, description_column)
    ):
        sp_padded = sub_package_value.ljust(sub_package_width)
        desc_padded = description.ljust(description_width)

        line = f"| {sp_padded} | {desc_padded} |"
        sub_packages_table_l.append(line)

        if i == 0:
            underline = f"| {'-' * sub_package_width} | {'-' * description_width} |"
            sub_packages_table_l.append(underline)

    sub_packages_table = "\n".join(sub_packages_table_l)
    return sub_packages_table


def write_module_page(
    package_full_name: str,
) -> PackageInfo:
    """
    Write the docs pages for a module/package
    """
    package = importlib.import_module(package_full_name)

    if hasattr(package, "__path__"):
        sub_packages = write_subpackage_pages(package)

    else:
        sub_packages = None

    package_name = package_full_name.rsplit(".", maxsplit=1)[-1]

    write_file = get_write_file(package_full_name)

    nav[package_full_name.split(".")] = write_file.relative_to(ROOT_DIR).as_posix()

    with mkdocs_gen_files.open(write_file, "w") as fh:
        fh.write(f"::: {package_full_name}\n")
        options = DIRECTIVE_OPTIONS.get(package_full_name)
        if options:
            fh.write("    options:\n")
            for key, value in options.items():
                fh.write(f"      {key}: {json.dumps(value)}\n")

        if sub_packages:
            fh.write("\n## Sub-packages\n\n")
            fh.write(f"{create_sub_packages_table(sub_packages)}\n")

    if not package.__doc__:
        summary = "No documentation available"
    else:
        package_doc_split = package.__doc__.splitlines()

        if not package_doc_split[0]:
            summary = package_doc_split[1]
        else:
            summary = package_doc_split[0]

    return PackageInfo(package_full_name, package_name, summary)


# =============================================================================
# Component Documentation Generation
# =============================================================================

COMPONENT_META_DIR = Path("docs/component_metadata")
COMPONENT_DOCS_DIR = Path("components")


def load_component_metadata() -> dict[str, dict[str, Any]]:
    """Load Rust component metadata from JSON files.

    Rust components are documented via JSON metadata files in
    `docs/component_metadata/`, generated by the `rscm-doc-gen` crate
    which parses Rust source files to extract component definitions,
    docstrings, and equations.

    For Python components, see `load_python_components()` which uses
    the `Component.get_registered_components()` registry.

    Returns
    -------
    Dictionary mapping component names to their metadata dicts
    """
    metadata = {}
    if not COMPONENT_META_DIR.exists():
        return metadata

    for json_file in COMPONENT_META_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                metadata[data["name"]] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return metadata


def load_python_components() -> dict[str, dict[str, Any]]:
    """Load Python component metadata from the Component registry.

    Python components (subclasses of `rscm.component.Component`) are
    automatically registered when their modules are imported. This function
    queries that registry to extract metadata for documentation.

    For Rust components, see `load_component_metadata()` which loads
    JSON metadata generated by the `rscm-doc-gen` crate.

    Returns
    -------
    Dictionary mapping component names to their metadata dicts
    """
    metadata = {}

    try:
        # Import modules that contain components to trigger registration
        import rscm.example_components  # noqa: F401, PLC0415
        from rscm.component import Component  # noqa: PLC0415
    except ImportError as e:
        print(f"Warning: Could not import Python components: {e}")
        return metadata

    # Get all registered components from the registry
    for name, cls in Component.get_registered_components().items():
        module = cls.__module__
        meta = extract_python_component_metadata(cls, module)
        if meta:
            metadata[name] = meta

    return metadata


def extract_python_component_metadata(
    cls: type, module_name: str
) -> dict[str, Any] | None:
    """Extract metadata from a Python Component class.

    Parameters
    ----------
    cls
        The component class to extract metadata from
    module_name
        The fully qualified module name (e.g., "rscm.example_components")
    """
    from rscm.component import Input, Output, State  # noqa: PLC0415

    name = cls.__name__
    docstring = inspect.getdoc(cls) or ""

    # The page heading names the component, so drop a one-line title paragraph.
    title, _, body = docstring.partition("\n\n")
    description = body if body and "\n" not in title else docstring

    # Get inputs, outputs, states from class attributes
    inputs = []
    outputs = []
    states = []

    for attr_name, attr_value in vars(cls).items():
        if isinstance(attr_value, Input):
            inputs.append(
                {
                    "rust_name": attr_name,
                    "name": attr_value.name,
                    "unit": attr_value.unit,
                    "grid": attr_value.grid,
                }
            )
        elif isinstance(attr_value, Output):
            outputs.append(
                {
                    "rust_name": attr_name,
                    "name": attr_value.name,
                    "unit": attr_value.unit,
                    "grid": attr_value.grid,
                }
            )
        elif isinstance(attr_value, State):
            states.append(
                {
                    "rust_name": attr_name,
                    "name": attr_value.name,
                    "unit": attr_value.unit,
                    "grid": attr_value.grid,
                }
            )

    # Extract parameters from __init__ signature
    parameters = []
    init_sig = inspect.signature(cls.__init__)
    init_doc = inspect.getdoc(cls.__init__) or ""

    all_defaults = True
    for param_name, param in init_sig.parameters.items():
        if param_name == "self":
            continue
        all_defaults = all_defaults and param.default is not inspect.Parameter.empty

        param_type = "Any"
        if param.annotation != inspect.Parameter.empty:
            param_type = getattr(param.annotation, "__name__", str(param.annotation))

        # Try to extract description from docstring
        param_desc = ""
        # Look for numpy-style parameter docs
        param_pattern = rf"{param_name}\s*\n\s+(.+?)(?=\n\s*\w+\s*\n|\Z)"
        match = re.search(param_pattern, init_doc, re.DOTALL)
        if match:
            param_desc = match.group(1).strip()

        parameters.append(
            {
                "name": param_name,
                "type": param_type,
                "description": param_desc,
            }
        )

    return {
        "name": name,
        "module_path": f"{module_name}.{name}",
        "language": "python",
        "description": description,
        "category": "Python Examples",
        "tags": ["python", "example"],
        "inputs": inputs,
        "outputs": outputs,
        "states": states,
        "parameters": parameters,
        "all_defaults": all_defaults,
        "source_file": f"python/rscm/{module_name.rsplit('.', maxsplit=1)[-1]}.py",
    }


PYTHON_BUILDER_MODULES = ("rscm.components", "rscm.two_layer", "rscm.magicc")


def find_python_builder_module(component_name: str) -> str | None:
    """Return the Python module exposing ``<component_name>Builder``, if any."""
    for module_name in PYTHON_BUILDER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, f"{component_name}Builder"):
            return module_name
    return None


def rust_import_path(module_path: str) -> str:
    """Return the public Rust path an item in ``module_path`` is re-exported from."""
    # Items live in private modules that their parent module re-exports.
    segments = module_path.split("::")[:-1]
    return "::".join(s for s in segments if s != "components")


def returns_result(meta: dict[str, Any]) -> bool:
    """Check whether the component's ``from_parameters`` is fallible."""
    source = Path(meta.get("source_file", ""))
    if not source.is_file():
        return False
    return bool(
        re.search(r"fn from_parameters\([^)]*\)\s*->\s*RscmResult", source.read_text())
    )


def find_parameters_struct(component_name: str) -> tuple[str, bool] | None:
    """Return the import path of ``<Name>Parameters`` and whether it has a default."""
    struct = f"{component_name}Parameters"
    for source in Path("crates").glob("*/src/**/*.rs"):
        text = source.read_text()
        if not re.search(rf"pub struct {struct}\b", text):
            continue
        relative = source.relative_to("crates").with_suffix("")
        crate, *modules = relative.parts
        module_path = "::".join(
            [crate.replace("-", "_"), *(m for m in modules if m not in ("src", "mod"))]
        )
        has_default = bool(re.search(rf"impl Default for {struct}\b", text))
        return f"{rust_import_path(module_path)}::{struct}", has_default
    return None


# -----------------------------------------------------------------------------
# Rustdoc Markdown normalisation
#
# Component descriptions are rustdoc (CommonMark) Markdown.
# Python-Markdown needs a blank line before lists, tables and display maths,
# four-space list nesting, and has no intra-doc links.
# -----------------------------------------------------------------------------

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])(\s+)(.*)$")
REFERENCE_DEFINITION = re.compile(r"^\s*\[([^\]]+)\]:\s+\S")
# An intra-doc link, or a code span (matched first, so links inside code are left alone)
INTRA_DOC_LINK = re.compile(
    r"(?P<code>(?<!\[)(`+)(?!\])[^`]+?\2(?![\]`]))"
    r"|\[(?P<text>`[^`\]]+`|[^\[\]`]+)\](?:\((?P<target>[^)\s]*)\))?(?![\[(:])"
)
RUST_PATH = re.compile(
    r"^(crate|super|self|Self)(::\w+)*$|^\w+(::\w+)+$|^[A-Za-z_]\w*$"
)


def is_fenced(lines: list[str]) -> list[bool]:
    """Flag the lines that are inside (or delimit) a fenced code block."""
    flags = []
    in_fence = False
    for line in lines:
        if FENCE.match(line):
            flags.append(True)
            in_fence = not in_fence
        else:
            flags.append(in_fence)
    return flags


def is_maths(lines: list[str]) -> list[int]:
    """Label each display maths line with its block number, counting from 1.

    Blocks run from ``$$`` to ``$$``. Lines outside maths get 0.
    """
    flags = []
    block = 0
    in_maths = False
    for line in lines:
        stripped = line.strip()
        if in_maths:
            flags.append(block)
            in_maths = not stripped.endswith("$$")
        elif stripped.startswith("$$"):
            block += 1
            flags.append(block)
            in_maths = len(stripped) <= 2 or not stripped.endswith("$$")  # noqa: PLR2004
        else:
            flags.append(0)
    return flags


def strip_rust_prefix(path: str) -> str:
    """Drop the ``crate::``, ``super::`` and ``self::`` prefixes of a Rust path."""
    return re.sub(r"^((crate|super|self)::)+", "", path)


def rewrite_intra_doc_links(line: str, references: set[str]) -> str:
    """Replace rustdoc intra-doc links, which have no target here, with code or text."""

    def replace(match: re.Match[str]) -> str:
        if match.group("code"):
            return match.group(0)
        text = match.group("text")
        target = match.group("target")
        if target is None and (
            text in references or not RUST_PATH.match(text.strip("`"))
        ):
            return match.group(0)
        if target is not None and not RUST_PATH.match(target):
            return match.group(0)
        if text.startswith("`"):
            return f"`{strip_rust_prefix(text.strip('`'))}`"
        if target is None:
            return f"`{strip_rust_prefix(text)}`"
        return text

    return INTRA_DOC_LINK.sub(replace, line)


def demote_headings_and_links(text: str, levels: int) -> list[str]:
    """Demote headings by ``levels`` and rewrite intra-doc links outside code blocks."""
    lines = text.splitlines()
    references = {
        m.group(1) for line in lines if (m := REFERENCE_DEFINITION.match(line))
    }
    out = []
    for line, fenced in zip(lines, is_fenced(lines)):
        if fenced:
            out.append(line)
            continue
        heading = HEADING.match(line)
        if heading:
            hashes = "#" * min(len(heading.group(1)) + levels, 6)
            out.append(
                rewrite_intra_doc_links(f"{hashes} {heading.group(2)}", references)
            )
        else:
            out.append(rewrite_intra_doc_links(line, references))
    return out


def can_interrupt_paragraph(match: re.Match[str]) -> bool:
    """Whether a list item may start a list directly after a paragraph (CommonMark)."""
    marker = match.group(2)
    return not marker[0].isdigit() or marker[:-1] == "1"


def reindent_lists(lines: list[str]) -> list[str]:
    """Nest list items four spaces deep and separate lists from preceding paragraphs."""
    out: list[str] = []
    # Source content column of each open list item, outermost first
    stack: list[int] = []
    prev_blank = True
    prev_in_paragraph = False
    verbatim = [f or bool(m) for f, m in zip(is_fenced(lines), is_maths(lines))]
    for line, fenced in zip(lines, verbatim):
        if not line.strip():
            out.append("")
            prev_blank = True
            prev_in_paragraph = False
            continue
        indent = len(line) - len(line.lstrip())
        item = None if fenced else LIST_ITEM.match(line)
        if item and (stack or can_interrupt_paragraph(item) or not prev_in_paragraph):
            if not stack and prev_in_paragraph:
                out.append("")
            while stack and stack[-1] > indent:
                stack.pop()
            depth = len(stack)
            stack.append(indent + len(item.group(2)) + len(item.group(3)))
            out.append(" " * (4 * depth) + f"{item.group(2)} {item.group(4)}")
        elif stack:
            if prev_blank:
                while stack and stack[-1] > indent:
                    stack.pop()
            if stack:
                extra = max(indent - stack[-1], 0)
                out.append(" " * (4 * len(stack) + extra) + line.lstrip())
            else:
                out.append(line)
        else:
            out.append(line)
        prev_blank = False
        prev_in_paragraph = not fenced and not HEADING.match(line)
    return out


def separate_blocks(lines: list[str]) -> list[str]:
    """Put blank lines around display maths and tables to keep them apart from text."""
    out: list[str] = []
    in_table = False
    maths = is_maths(lines)

    def ensure_blank() -> None:
        if out and out[-1].strip():
            out.append("")

    for i, (line, fenced) in enumerate(zip(lines, is_fenced(lines))):
        stripped = line.strip()
        if fenced:
            in_table = False
            out.append(line)
            continue
        if maths[i]:
            if i == 0 or maths[i - 1] != maths[i]:
                ensure_blank()
            out.append(line)
            if i + 1 == len(lines) or maths[i + 1] != maths[i]:
                out.append("")
            continue
        if stripped.startswith("|"):
            if not in_table:
                ensure_blank()
            in_table = True
        else:
            in_table = False
        if not stripped and out and not out[-1].strip():
            continue
        out.append(line)
    return out


def normalise_rustdoc(text: str, heading_levels: int = 1) -> str:
    """Convert rustdoc Markdown into Markdown that Python-Markdown renders alike."""
    lines = demote_headings_and_links(text, heading_levels)
    lines = reindent_lists(lines)
    lines = separate_blocks(lines)
    return "\n".join(lines).strip()


# -----------------------------------------------------------------------------
# Inputs and outputs described in prose
# -----------------------------------------------------------------------------

# Title and verb for each kind of IO, in page order
IO_KINDS = {
    "inputs": ("Inputs", "reads"),
    "outputs": ("Outputs", "writes"),
    "states": ("States", "tracks"),
}
IO_HEADING = re.compile(r"^#\s+(Inputs|Outputs|States)\b", re.IGNORECASE)


def expand_braces(name: str) -> list[str]:
    """Expand ``A|{B,C}`` into ``A|B`` and ``A|C``."""
    match = re.search(r"\{([^}]*)\}", name)
    if not match:
        return [name]
    head, tail = name[: match.start()], name[match.end() :]
    return [
        n
        for part in match.group(1).split(",")
        for n in expand_braces(head + part + tail)
    ]


def parse_io_bullet(text: str) -> tuple[list[str], str]:
    """Split a prose IO bullet into its variable names and description."""
    paren = text.find("(")
    names_part = text if paren < 0 else text[:paren]
    names = [
        n for raw in re.findall(r"`([^`]+)`", names_part) for n in expand_braces(raw)
    ]
    if paren >= 0:
        depth = 0
        for end, char in enumerate(text[paren:], start=paren):
            depth += {"(": 1, ")": -1}.get(char, 0)
            if depth == 0:
                break
        rest = text[end + 1 :]
    else:
        rest = re.sub(r"^.*`", "", names_part)
    rest = re.sub(r"^\s*([-:,]|\u2013)?\s*", "", rest).strip()
    return names, rest[:1].upper() + rest[1:]


def extract_io_sections(
    description: str, meta: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, list[str]]]:
    """Remove the prose IO sections that repeat the IO tables.

    Returns the remaining description, a description for each variable in the tables,
    and the prose bullets of variables the tables lack, by section.
    """
    known = {v["name"] for kind in IO_KINDS for v in meta.get(kind, [])}
    lines = description.splitlines()
    fenced = is_fenced(lines)
    kept: list[str] = []
    notes: dict[str, str] = {}
    extras: dict[str, list[str]] = {}
    section = None
    bullets: list[list[str]] = []

    def flush() -> None:
        for bullet in bullets:
            text = " ".join(part.strip() for part in bullet)
            names, note = parse_io_bullet(text)
            matched = [n for n in names if n in known]
            if matched and len(matched) == len(names):
                for name in matched:
                    if note:
                        notes[name] = note
            else:
                extras.setdefault(section, []).append(text)
        bullets.clear()

    for line, is_code in zip(lines, fenced):
        if not is_code and line.startswith("#"):
            if section:
                flush()
            heading = IO_HEADING.match(line)
            section = heading.group(1).lower() if heading else None
            if section:
                continue
        if section and not is_code:
            if line.startswith("- "):
                bullets.append([line[2:]])
                continue
            if bullets and line.startswith(" ") and line.strip():
                bullets[-1].append(line)
                continue
        kept.append(line)
    if section:
        flush()
    return "\n".join(kept), notes, extras


# -----------------------------------------------------------------------------
# Component pages
# -----------------------------------------------------------------------------

REPO_URL = "https://github.com/lewisjared/rscm"


def table_cell(text: str) -> str:
    """Make text safe for a single Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def io_table(variables: list[dict[str, Any]], notes: dict[str, str]) -> list[str]:
    """Render an IO table, with a Description column only when a variable has one."""
    descriptions = {v["name"]: notes.get(v["name"], "") for v in variables}
    with_description = any(descriptions.values())
    header = "| Variable | Unit | Grid |" + (
        " Description |" if with_description else ""
    )
    rule = "|----------|------|------|" + ("-------------|" if with_description else "")
    rows = [header, rule]
    for v in variables:
        name = v["name"]
        # Code spans keep their pipes in Python-Markdown tables
        row = f"| `{name}` | {v.get('unit', '')} | {v.get('grid', 'scalar')} |"
        if with_description:
            row += f" {table_cell(descriptions[name])} |"
        rows.append(row)
    return rows


def split_parameter_description(description: str) -> tuple[str, str]:
    """Split a parameter's doc comment into its unit and the rest."""
    unit = ""
    rest = []
    for line in description.splitlines():
        match = re.match(r"^\s*units?:\s*(.+?)\s*$", line, re.IGNORECASE)
        if match and not unit:
            unit = match.group(1).strip("`")
        else:
            rest.append(line)
    return unit, " ".join(" ".join(rest).split())


def parameters_table(parameters: list[dict[str, Any]]) -> list[str]:
    """Render the parameters a Rust component reads from its parameter struct."""
    split = [split_parameter_description(p.get("description", "")) for p in parameters]
    with_unit = any(unit for unit, _ in split)
    header = "| Name | Type |" + (" Unit |" if with_unit else "") + " Description |"
    rule = "|------|------|" + ("------|" if with_unit else "") + "-------------|"
    rows = [header, rule]
    for param, (unit, desc) in zip(parameters, split):
        row = f"| `{param.get('name', '')}` | `{param.get('type', '')}` |"
        if with_unit:
            row += f" {table_cell(unit)} |"
        rows.append(f"{row} {table_cell(desc)} |")
    return rows


def usage_section(meta: dict[str, Any]) -> list[str]:
    """Show how to build a component from Python and Rust."""
    name = meta["name"]
    lines = ["## Usage", ""]
    if meta.get("language") != "rust":
        module_path = meta.get("module_path", "").rsplit(".", 1)[0]
        arguments = "" if meta.get("all_defaults") else "**parameters"
        return [
            *lines,
            "```python",
            f"from {module_path} import {name}",
            "",
            f"component = {name}({arguments})",
            "```",
            "",
        ]

    has_parameter_table = bool(meta.get("parameters"))
    builder_module = find_python_builder_module(name)
    if builder_module:
        builder = f"{name}Builder"
        defaults = builder_module == "rscm.magicc"
        lines.append(f"From Python, use [`{builder}`][{builder_module}.{builder}].")
        if defaults:
            lines.append("Missing parameter keys take their defaults.")
        elif has_parameter_table:
            lines.append(
                "`parameters` is a dict with the keys in the parameters table."
            )
        lines.extend(
            [
                "",
                "```python",
                f"from {builder_module} import {builder}",
                "",
                f"component = {builder}.from_parameters("
                + ("{}" if defaults else "parameters")
                + ").build()",
                "```",
                "",
            ]
        )

    struct = find_parameters_struct(name)
    construct = f"{name}::from_parameters(parameters)"
    imports = [f"use {rust_import_path(meta['module_path'])}::{name};"]
    rust_intro = "From Rust:"
    if struct:
        struct_path, has_default = struct
        struct_name = struct_path.rsplit("::", 1)[-1]
        imports.append(f"use {struct_path};")
        if has_default:
            construct = f"{name}::from_parameters({struct_name}::default())"
        else:
            rust_intro = f"From Rust, where `parameters` is a `{struct_name}`:"
    lines.extend([rust_intro, ""])
    if returns_result(meta):
        construct += "?"
    lines.extend(
        ["```rust", *sorted(imports), "", f"let component = {construct};", "```", ""]
    )
    return lines


def generate_component_page(meta: dict[str, Any]) -> str:
    """Generate markdown for a component documentation page."""
    name = meta["name"]
    lines = [f"# {name} {{ #component-{name.lower()} }}", ""]

    if meta.get("category"):
        lines.append(f"**Category:** {meta['category']}  ")
    if meta.get("tags"):
        tags_str = " ".join(f"`{t}`" for t in meta["tags"])
        lines.append(f"**Tags:** {tags_str}  ")
    lines.append(f"**Language:** {meta.get('language', 'rust').title()}")
    lines.append("")

    description, notes, extras = extract_io_sections(
        meta.get("description") or "", meta
    )
    if description.strip():
        lines.extend([normalise_rustdoc(description), ""])

    for kind, (title, verb) in IO_KINDS.items():
        variables = meta.get(kind) or []
        if not variables and not extras.get(kind):
            continue
        lines.extend([f"## {title}", ""])
        if variables:
            lines.extend([*io_table(variables, notes), ""])
        if extras.get(kind):
            lines.extend([f"Depending on its parameters, it also {verb}:", ""])
            lines.extend(f"- {text}" for text in extras[kind])
            lines.append("")

    if meta.get("parameters"):
        lines.extend(["## Parameters", "", *parameters_table(meta["parameters"]), ""])

    lines.extend(usage_section(meta))

    if meta.get("source_file"):
        source = meta["source_file"]
        lines.extend(
            ["---", "", f"Source: [`{source}`]({REPO_URL}/blob/main/{source})"]
        )

    return "\n".join(lines) + "\n"


def summarise(description: str, max_length: int = 100) -> str:
    """Return the first sentence of a description as plain text for a table cell."""
    paragraph = ""
    for raw in re.split(r"\n\s*\n", description):
        block = raw.strip()
        if block and not block.startswith(("#", "$$", "|", "```", "- ")):
            paragraph = " ".join(block.split())
            break
    sentence = re.split(r"(?<=[.!?])\s", paragraph, maxsplit=1)[0]
    sentence = re.sub(r"\[`?([^\]`]+)`?\](\([^)]*\))?", r"\1", sentence).replace(
        "`", ""
    )
    if len(sentence) > max_length:
        sentence = sentence[:max_length].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return table_cell(sentence)


def generate_component_index(all_metadata: dict[str, dict[str, Any]]) -> str:
    """Generate the components index page."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for meta in all_metadata.values():
        by_category.setdefault(meta.get("category") or "Other", []).append(meta)

    lines = [
        "# Components",
        "",
        "Each page lists a component's inputs, outputs and states,",
        "its parameters where they are documented, and how to build it.",
        "",
    ]

    undocumented = [
        (module_name, builder)
        for module_name in PYTHON_BUILDER_MODULES
        for builder in sorted(
            getattr(importlib.import_module(module_name), "__all__", [])
        )
        if builder.endswith("Builder")
        and builder.removesuffix("Builder") not in all_metadata
    ]
    if undocumented:
        links = [
            f"[`{builder}`][{module}.{builder}]" for module, builder in undocumented
        ]
        lines.extend(
            [
                "These components have no page yet, so their builders document them: "
                + ", ".join(links)
                + ".",
                "",
            ]
        )

    for category, components in sorted(by_category.items()):
        lines.extend(
            [
                f"## {category}",
                "",
                "| Component | Language | Description | Tags |",
                "|-----------|----------|-------------|------|",
            ]
        )
        for meta in sorted(components, key=lambda m: m["name"]):
            name = meta["name"]
            lang = meta.get("language", "rust").title()
            desc = summarise(meta.get("description") or "")
            tags = " ".join(f"`{t}`" for t in meta.get("tags", []))
            lines.append(f"| [{name}]({name.lower()}.md) | {lang} | {desc} | {tags} |")
        lines.append("")

    return "\n".join(lines)


def write_component_docs():
    """Write all component documentation pages."""
    # Load Rust component metadata from JSON files
    all_metadata = load_component_metadata()

    # Load Python component metadata
    python_metadata = load_python_components()
    all_metadata.update(python_metadata)

    if not all_metadata:
        print("No component metadata found. Run rscm-doc-gen first.")
        return

    # Write individual component pages
    for name, meta in all_metadata.items():
        page_content = generate_component_page(meta)
        page_path = COMPONENT_DOCS_DIR / f"{name.lower()}.md"

        with mkdocs_gen_files.open(page_path, "w") as fh:
            fh.write(page_content)

    # Write index page
    index_content = generate_component_index(all_metadata)
    index_path = COMPONENT_DOCS_DIR / "index.md"

    with mkdocs_gen_files.open(index_path, "w") as fh:
        fh.write(index_content)

    print(f"Generated documentation for {len(all_metadata)} components")


# Write module pages
write_module_page("rscm")

# Generate component documentation
write_component_docs()

# Render navigation
with mkdocs_gen_files.open(ROOT_DIR / "NAVIGATION.md", "w") as fh:
    fh.writelines(nav.build_literate_nav(indentation=2))
