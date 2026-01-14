# Feature Specification: Publish Packages to PyPI and crates.io

**Feature Branch**: `001-publish-packages`
**Created**: 2026-01-14
**Status**: Draft
**Input**: User description: "I want to publish this repository on pypi and crates.io. This should resolve issues #5 and #6"
**Related Issues**: #5 (Versioning in rust), #6 (Publication)

## Clarifications

### Session 2026-01-14

- Q: When a release workflow partially succeeds (e.g., PyPI publish works but crates.io fails), what should happen? → A: Rollback - Fail entire release, require manual retry after fixing the issue.
- Q: What should happen if someone tries to publish a version that already exists on a registry? → A: Skip - Detect existing version, skip that registry, continue with others.
- Q: Should the release workflow support publishing pre-release versions (alpha, beta, rc)? → A: Yes - Support pre-releases on both registries using standard semver format (e.g., 0.2.0-alpha.1).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Rust Developer Installs Core Crate (Priority: P1)

A Rust developer building a climate model wants to use the RSCM framework. They add `rscm-core` to their Cargo.toml dependencies and the crate downloads from crates.io with all required metadata visible.

**Why this priority**: Core crate availability is foundational - without it, no external Rust developers can use the framework.

**Independent Test**: Can be fully tested by running `cargo add rscm-core` in a new Rust project and verifying the crate compiles with `cargo build`.

**Acceptance Scenarios**:

1. **Given** a new Rust project, **When** a developer adds `rscm-core = "0.1"` to Cargo.toml and runs `cargo build`, **Then** the crate downloads from crates.io and compiles successfully.
2. **Given** the crates.io page for `rscm-core`, **When** a developer views the crate page, **Then** they see description, repository link, license, and documentation link.

---

### User Story 2 - Python User Installs Package (Priority: P1)

A Python climate scientist wants to use RSCM for their research. They install the package using pip and it works on their platform (Linux, macOS, or Windows).

**Why this priority**: Python package availability enables the primary user base (climate scientists using Python) to access the framework.

**Independent Test**: Can be fully tested by running `pip install rscm` in a fresh virtual environment and importing the package.

**Acceptance Scenarios**:

1. **Given** a Python 3.10+ environment on Linux, macOS, or Windows, **When** a user runs `pip install rscm`, **Then** the package installs successfully with pre-built wheels.
2. **Given** an installed rscm package, **When** a user runs `import rscm`, **Then** the module loads without errors.
3. **Given** the PyPI page for rscm, **When** a user views the package page, **Then** they see description, repository link, license, and available wheels for multiple platforms.

---

### User Story 3 - Rust Developer Uses Components Crate (Priority: P2)

A Rust developer wants to use pre-built climate model components. They add `rscm-components` to their project and access the provided component implementations.

**Why this priority**: Components crate provides value-add functionality but depends on core crate being published first.

**Independent Test**: Can be fully tested by adding `rscm-components` to a Rust project and using one of the provided components.

**Acceptance Scenarios**:

1. **Given** a Rust project with `rscm-core` already working, **When** a developer adds `rscm-components`, **Then** both crates compile together successfully.
2. **Given** the crates.io page for `rscm-components`, **When** a developer views the page, **Then** they see the dependency relationship to `rscm-core`.

---

### User Story 4 - Maintainer Publishes New Release (Priority: P2)

A project maintainer wants to publish a new version after merging changes. They trigger the release workflow and packages are automatically published to both registries.

**Why this priority**: Automated publishing reduces manual effort and ensures consistent releases across platforms.

**Independent Test**: Can be tested by creating a version bump and verifying packages appear on both registries.

**Acceptance Scenarios**:

1. **Given** changes merged to main branch, **When** a maintainer triggers a version bump, **Then** the workflow creates a git tag and GitHub release.
2. **Given** a new version tag pushed to GitHub, **When** the release workflow runs, **Then** Python wheels are built for Linux, macOS, and Windows and uploaded to PyPI.
3. **Given** a new version tag pushed to GitHub, **When** the release workflow runs, **Then** `rscm-core` and `rscm-components` are published to crates.io in correct dependency order.

---

### Edge Cases

- Registry unavailability: Release fails, maintainer retries after service is restored.
- Partial registry success: Entire release fails, maintainer must manually retry after resolving the issue.
- Duplicate version: Workflow detects existing version on a registry, skips that registry, and continues with others.
- Pre-release versions: Supported on both registries using standard semver format (e.g., 0.2.0-alpha.1).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Project MUST have a valid open-source license file in the repository root.
- **FR-002**: All Cargo.toml files MUST include required crates.io metadata (repository, license, authors).
- **FR-003**: pyproject.toml MUST include required PyPI metadata (authors, license, project URLs).
- **FR-004**: Release workflow MUST build Python wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64).
- **FR-005**: Release workflow MUST publish Python package to PyPI when a version tag is pushed.
- **FR-006**: Release workflow MUST publish `rscm-core` and `rscm-components` to crates.io when a version tag is pushed.
- **FR-007**: Rust crates MUST be published in dependency order (rscm-core before rscm-components).
- **FR-008**: Release workflow MUST fail the entire release if any registry publish fails, providing clear error messages and requiring manual retry.
- **FR-009**: Repository MUST have CI/CD secrets configured for PyPI and crates.io authentication.
- **FR-010**: Release workflow MUST detect if a version already exists on a registry and skip publishing to that registry rather than failing.
- **FR-011**: Release workflow MUST support publishing pre-release versions (alpha, beta, rc) using standard semver format to both registries.

### Key Entities

- **rscm-core crate**: Core Rust crate with framework traits and abstractions. Published to crates.io.
- **rscm-components crate**: Rust crate with concrete component implementations. Depends on rscm-core. Published to crates.io.
- **rscm Python package**: Python bindings via PyO3/maturin. Published to PyPI with platform-specific wheels.
- **Release workflow**: GitHub Actions workflow that orchestrates building and publishing to both registries.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Rust developers can install `rscm-core` and `rscm-components` from crates.io using standard Cargo commands.
- **SC-002**: Python users can install `rscm` from PyPI using pip on Linux, macOS, and Windows without compilation.
- **SC-003**: Package metadata (license, repository, authors) is visible on both crates.io and PyPI package pages.
- **SC-004**: A new release can be published to both registries within 30 minutes of triggering the workflow.
- **SC-005**: Release workflow provides clear success/failure status for each registry publish step.

## Assumptions

- The project will use the Apache-2.0 license.
- Repository secrets for PyPI (trusted publisher or API token) and crates.io (API token) will be configured by the maintainer.
- The existing bump-my-version and towncrier configuration will continue to be used for version management.
- Cross-platform wheel building will use maturin's GitHub Actions with cross-compilation support.
- The main `rscm` Rust crate (PyO3 bindings) does not need to be published to crates.io, only to PyPI as a Python package.

## Out of Scope

- Publishing documentation to docs.rs (can be added later).
- Automated testing of published packages post-release.
- Support for additional Python versions beyond 3.10+.
- Publishing to conda-forge (separate distribution channel).
