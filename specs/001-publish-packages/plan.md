# Implementation Plan: Publish Packages to PyPI and crates.io

**Branch**: `001-publish-packages` | **Date**: 2026-01-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-publish-packages/spec.md`

## Summary

Enable publishing of `rscm-core` and `rscm-components` to crates.io and the `rscm` Python package to PyPI with cross-platform wheels. This requires adding license files, completing package metadata, and extending the existing GitHub Actions release workflow.

## Technical Context

**Language/Version**: Rust 1.75+ (2021 edition), Python 3.10+
**Primary Dependencies**: maturin (PyO3 bindings), GitHub Actions, cargo, uv
**Storage**: N/A (CI/CD configuration only)
**Testing**: Workflow tested via tag push; local dry-run with `cargo publish --dry-run` and `uv build`
**Target Platform**: GitHub Actions runners (ubuntu-latest, macos-latest, windows-latest)
**Project Type**: Hybrid Rust/Python with workspace crates
**Performance Goals**: Release completes within 30 minutes (SC-004)
**Constraints**: Atomic release - all registries succeed or release fails entirely
**Scale/Scope**: 2 Rust crates to crates.io, 1 Python package to PyPI with 5 wheel targets

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Code Quality First | PASS | Configuration files only; no production code changes |
| II. Testing Standards | PASS | Workflow can be validated via dry-run and manual tag tests |
| III. User Experience Consistency | PASS | No API changes; package metadata improves discoverability |
| IV. Performance Requirements | PASS | CI/CD only; no runtime performance impact |

**Quality Gates Impact**:
- Build gate: Unchanged (existing CI)
- Rust Tests gate: Unchanged
- Python Tests gate: Unchanged
- Linting gate: Unchanged
- Documentation gate: Unchanged

No constitution violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-publish-packages/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (workflow configuration model)
├── quickstart.md        # Phase 1 output (release guide)
├── contracts/           # Phase 1 output (not applicable - no API contracts)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
# Files to create/modify
LICENSE                           # New: Apache-2.0 license file
Cargo.toml                        # Modify: add repository, license, authors
rscm-core/Cargo.toml              # Modify: add repository, license, authors
rscm-components/Cargo.toml        # Modify: add repository, license, authors
pyproject.toml                    # Modify: add authors, license, project.urls
.github/workflows/release.yaml    # Modify: add crates.io and PyPI publish steps
```

**Structure Decision**: No new directories needed. Modifications to existing configuration files and one new LICENSE file at repository root.

## Complexity Tracking

No constitution violations requiring justification.
