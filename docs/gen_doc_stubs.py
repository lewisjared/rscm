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
import json
import pkgutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mkdocs_gen_files
from attrs import define

ROOT_DIR = Path("api")
nav = mkdocs_gen_files.Nav()


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
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        # Skip "private" packages
        if name.startswith("_"):
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

    write_file = write_dir / package_full_name.split(".")[-1] / "index.md"

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

    package_name = package_full_name.split(".")[-1]

    write_file = get_write_file(package_full_name)

    nav[package_full_name.split(".")] = write_file.relative_to(ROOT_DIR).as_posix()

    with mkdocs_gen_files.open(write_file, "w") as fh:
        fh.write(f"# {package_full_name}\n")

        if sub_packages:
            fh.write("\n")
            fh.write(f"{create_sub_packages_table(sub_packages)}\n")

        fh.write("\n")
        fh.write(f"::: {package_full_name}")

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
    """Load all component metadata JSON files."""
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


def generate_component_page(meta: dict[str, Any]) -> str:  # noqa: PLR0912, PLR0915
    """Generate markdown for a component documentation page."""
    lines = [
        f"# {meta['name']}",
        "",
    ]

    # Metadata badges
    if meta.get("category"):
        lines.append(f"**Category:** {meta['category']}  ")
    if meta.get("tags"):
        tags_str = " ".join(f"`{t}`" for t in meta["tags"])
        lines.append(f"**Tags:** {tags_str}  ")
    lines.append(f"**Language:** {meta.get('language', 'rust').title()}")
    lines.append("")

    # Description
    if meta.get("description"):
        lines.append(meta["description"])
        lines.append("")

    # Mathematical formulation
    if meta.get("equations"):
        lines.extend(
            [
                "## Mathematical Formulation",
                "",
                meta["equations"],
                "",
            ]
        )

    # Inputs
    if meta.get("inputs"):
        lines.extend(
            [
                "## Inputs",
                "",
                "| Variable | Unit | Grid | Description |",
                "|----------|------|------|-------------|",
            ]
        )
        for inp in meta["inputs"]:
            var_name = inp.get("variable_name", inp.get("rust_name", ""))
            unit = inp.get("unit", "")
            grid = inp.get("grid", "Scalar")
            desc = inp.get("description", "")
            lines.append(f"| `{var_name}` | {unit} | {grid} | {desc} |")
        lines.append("")

    # Outputs
    if meta.get("outputs"):
        lines.extend(
            [
                "## Outputs",
                "",
                "| Variable | Unit | Grid | Description |",
                "|----------|------|------|-------------|",
            ]
        )
        for out in meta["outputs"]:
            var_name = out.get("variable_name", out.get("rust_name", ""))
            unit = out.get("unit", "")
            grid = out.get("grid", "Scalar")
            desc = out.get("description", "")
            lines.append(f"| `{var_name}` | {unit} | {grid} | {desc} |")
        lines.append("")

    # States
    if meta.get("states"):
        lines.extend(
            [
                "## States",
                "",
                "| Variable | Unit | Grid | Description |",
                "|----------|------|------|-------------|",
            ]
        )
        for state in meta["states"]:
            var_name = state.get("variable_name", state.get("rust_name", ""))
            unit = state.get("unit", "")
            grid = state.get("grid", "Scalar")
            desc = state.get("description", "")
            lines.append(f"| `{var_name}` | {unit} | {grid} | {desc} |")
        lines.append("")

    # Parameters
    if meta.get("parameters"):
        lines.extend(
            [
                "## Parameters",
                "",
                "| Name | Type | Unit | Description |",
                "|------|------|------|-------------|",
            ]
        )
        for param in meta["parameters"]:
            name = param.get("name", "")
            param_type = param.get("type", "")
            unit = param.get("unit", "")
            desc = param.get("description", "")
            lines.append(f"| `{name}` | {param_type} | {unit} | {desc} |")
        lines.append("")

    # Usage examples
    lines.extend(["## Usage", ""])

    component_name = meta["name"]
    if meta.get("language") == "rust":
        # Python usage (via builder)
        if meta.get("python_builder"):
            builder_name = meta["python_builder"].split(".")[-1]
            lines.extend(
                [
                    "### Python",
                    "",
                    "```python",
                    f"from rscm.components import {builder_name}",
                    "",
                    f"component = {builder_name}.from_parameters({{",
                    "    # parameters here",
                    "}}).build()",
                    "```",
                    "",
                ]
            )
        else:
            # Infer builder name
            builder_name = f"{component_name}Builder"
            lines.extend(
                [
                    "### Python",
                    "",
                    "```python",
                    f"from rscm.components import {builder_name}",
                    "",
                    f"component = {builder_name}.from_parameters({{",
                    "    # parameters here",
                    "}}).build()",
                    "```",
                    "",
                ]
            )

        # Rust usage
        lines.extend(
            [
                "### Rust",
                "",
                "```rust",
                f"use rscm_components::{component_name};",
                "",
                f"let component = {component_name}::from_parameters({component_name}Parameters {{",  # noqa: E501
                "    // parameters here",
                "}});",
                "```",
                "",
            ]
        )
    else:
        # Python-only component
        module_path = meta.get("module_path", "").rsplit(".", 1)[0]
        lines.extend(
            [
                "### Python",
                "",
                "```python",
                f"from {module_path} import {component_name}",
                "",
                f"component = {component_name}(",
                "    # parameters here",
                ")",
                "```",
                "",
            ]
        )

    # Source reference
    if meta.get("source_file"):
        lines.extend(
            [
                "---",
                "",
                f"*Source: `{meta['source_file']}`*",
            ]
        )

    return "\n".join(lines)


def generate_component_index(
    all_metadata: dict[str, dict[str, Any]], long_description_cutoff: int = 60
) -> str:
    """Generate the components index page with filtering."""
    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    all_tags: set[str] = set()

    for meta in all_metadata.values():
        category = meta.get("category", "Other")
        if category is None:
            category = "Other"
        by_category.setdefault(category, []).append(meta)
        all_tags.update(meta.get("tags", []))

    lines = [
        "# Components",
        "",
        "This page lists all RSCM components with their inputs, outputs, and parameters.",  # noqa: E501
        "",
    ]

    # Add component tables by category
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
            slug = name.lower()
            lang = meta.get("language", "rust").title()
            desc = (meta.get("description") or "")[:long_description_cutoff]
            if len(meta.get("description", "") or "") > long_description_cutoff:
                desc += "..."
            # Clean up description for table
            desc = desc.replace("\n", " ").replace("|", "\\|")
            tags = " ".join(f"`{t}`" for t in meta.get("tags", []))
            lines.append(f"| [{name}]({slug}.md) | {lang} | {desc} | {tags} |")
        lines.append("")

    return "\n".join(lines)


def write_component_docs():
    """Write all component documentation pages."""
    all_metadata = load_component_metadata()

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
