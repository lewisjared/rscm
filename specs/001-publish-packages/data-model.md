# Data Model: Release Workflow Configuration

**Feature**: 001-publish-packages
**Date**: 2026-01-14

This feature does not introduce new application data entities. Instead, it defines configuration structures for the release workflow.

## Workflow Configuration Entities

### ReleaseJob

Represents a single job in the release workflow.

| Field | Type | Description |
|-------|------|-------------|
| name | string | Job identifier (e.g., "build-wheels", "publish-pypi") |
| runs-on | string | GitHub Actions runner (e.g., "ubuntu-latest") |
| needs | string[] | Job dependencies |
| environment | string? | GitHub environment name (optional) |
| permissions | Permission[] | Required permissions |

### BuildTarget

Represents a cross-compilation target for wheel building.

| Field | Type | Description |
|-------|------|-------------|
| os | string | Runner OS ("ubuntu-latest", "macos-13", "macos-14", "windows-latest") |
| target | string | Rust target triple (e.g., "x86_64-unknown-linux-gnu") |
| manylinux | string? | Manylinux version ("auto", "2014", or null for non-Linux) |

**Valid Targets**:

| Target | OS | Manylinux |
|--------|-----|-----------|
| x86_64-unknown-linux-gnu | ubuntu-latest | auto |
| aarch64-unknown-linux-gnu | ubuntu-latest | auto |
| x86_64-apple-darwin | macos-13 | - |
| aarch64-apple-darwin | macos-14 | - |
| x86_64-pc-windows-msvc | windows-latest | - |

### PackageMetadata

Required metadata for package registries.

| Field | Cargo.toml | pyproject.toml | Required |
|-------|------------|----------------|----------|
| name | package.name | project.name | Yes |
| version | package.version | project.version | Yes |
| description | package.description | project.description | Yes |
| license | package.license | project.license | Yes |
| repository | package.repository | project.urls.Repository | Yes |
| authors | package.authors | project.authors | Yes |
| readme | package.readme | project.readme | Yes |
| keywords | package.keywords | project.keywords | No |
| categories | package.categories | project.classifiers | No |

### GitHubSecret

Required repository secrets.

| Name | Purpose | Scope |
|------|---------|-------|
| CARGO_REGISTRY_TOKEN | crates.io API token | crates.io publishing |
| PAT | Personal access token | GitHub release creation |

Note: PyPI uses OIDC Trusted Publishers, no secret required.

### GitHubEnvironment

Required repository environment.

| Name | Purpose | Protection Rules |
|------|---------|------------------|
| release | PyPI publishing | Optional: require approval |

## State Transitions

### Release Workflow States

```
[Tag Pushed]
    → [Build Wheels]
        → [Publish to crates.io]
            → [Publish to PyPI]
                → [Create GitHub Release]
```

### Per-Registry Publish States

```
[Check Version Exists]
    ├─ Exists → [Skip]
    └─ Not Exists → [Publish]
                      ├─ Success → [Continue]
                      └─ Failure → [Fail Entire Release]
```

## Validation Rules

1. **Version Consistency**: All Cargo.toml files and pyproject.toml MUST have matching versions.
2. **License File**: LICENSE file MUST exist in repository root.
3. **Metadata Completeness**: All required metadata fields MUST be populated.
4. **Dependency Order**: rscm-core MUST be published before rscm-components.
5. **Atomic Release**: If any publish step fails, the entire release fails.
