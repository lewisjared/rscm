# Research: Publishing to PyPI and crates.io

**Feature**: 001-publish-packages
**Date**: 2026-01-14

## Decision Summary

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Cross-platform wheels | PyO3/maturin-action@v1 with matrix builds | Official action, handles manylinux compliance automatically |
| PyPI authentication | Trusted Publishers (OIDC) | No API tokens needed, short-lived credentials, more secure |
| crates.io authentication | CARGO_REGISTRY_TOKEN secret | OIDC not yet available for crates.io |
| Workspace publishing | katyo/publish-crates@v2 | Automatic dependency ordering, handles propagation delays |
| Version existence check | REST API queries | Simple curl checks against PyPI JSON API and crates.io API |

## Cross-Platform Wheel Building

### Recommended Action: PyO3/maturin-action@v1

The official maturin-action provides cross-compilation support with preconfigured Docker images for manylinux compliance.

**Matrix Configuration**:

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      # Linux x86_64 (manylinux2014)
      - os: ubuntu-latest
        target: x86_64-unknown-linux-gnu
        manylinux: auto
      # Linux aarch64 (cross-compilation)
      - os: ubuntu-latest
        target: aarch64-unknown-linux-gnu
        manylinux: auto
      # macOS x86_64
      - os: macos-13
        target: x86_64-apple-darwin
      # macOS arm64 (Apple Silicon)
      - os: macos-14
        target: aarch64-apple-darwin
      # Windows x86_64
      - os: windows-latest
        target: x86_64-pc-windows-msvc
```

**Key Points**:
- `manylinux: auto` selects appropriate manylinux version (2014+ for Rust 1.64+)
- macOS runners: macos-13 for Intel, macos-14 for Apple Silicon
- Action handles Docker container setup for Linux cross-compilation

## PyPI Trusted Publishers

PyPI Trusted Publishers use OIDC for authentication, eliminating the need for API tokens.

### Setup Requirements

1. **On PyPI** (https://pypi.org/manage/project/rscm/settings/publishing/):
   - Add trusted publisher
   - Owner: `lewisjared`
   - Repository: `rscm`
   - Workflow: `release.yaml`
   - Environment: `release` (recommended)

2. **GitHub Repository**:
   - Create `release` environment in Settings → Environments
   - Optionally add deployment protection rules

### Workflow Configuration

```yaml
publish-pypi:
  environment: release
  permissions:
    id-token: write  # Required for OIDC
  steps:
    - uses: pypa/gh-action-pypi-publish@release/v1
      # No credentials needed
```

**Critical**: `permissions: id-token: write` is required for OIDC token exchange.

## crates.io Publishing

### Recommended Action: katyo/publish-crates@v2

This action handles workspace dependency ordering automatically.

**Key Features**:
- Uses `cargo metadata` to understand dependency graph
- Publishes in correct order (rscm-core → rscm-components)
- Waits for dependencies to become available on crates.io
- Skips versions that already exist

### Workspace Publishing Order

1. `rscm-core` (no workspace dependencies)
2. `rscm-components` (depends on rscm-core)
3. `rscm` - NOT published to crates.io (PyO3 bindings only)

### Setup

Create `CARGO_REGISTRY_TOKEN` secret:
```bash
cargo login
# Copy token from ~/.cargo/credentials.toml to GitHub Secrets
```

## Version Existence Detection (FR-010)

### PyPI Check

```bash
VERSION="0.1.0"
if curl -s "https://pypi.org/pypi/rscm/$VERSION/json" | jq -e '.info' >/dev/null 2>&1; then
  echo "Version already exists"
fi
```

### crates.io Check

```bash
CRATE="rscm-core"
VERSION="0.1.0"
if curl -sf "https://crates.io/api/v1/crates/$CRATE/$VERSION" >/dev/null; then
  echo "Version already exists"
fi
```

## Package Metadata Requirements

### Cargo.toml (crates.io)

Required fields for publishing:
```toml
[package]
name = "rscm-core"
version = "0.1.0"
license = "Apache-2.0"
repository = "https://github.com/lewisjared/rscm"
authors = ["Jared Lewis <jared.lewis@climate-energy-college.org>"]
# Already present: description, readme, keywords, categories
```

### pyproject.toml (PyPI)

Required fields for publishing:
```toml
[project]
name = "rscm"
version = "0.1.0"
license = {text = "Apache-2.0"}
authors = [
    {name = "Jared Lewis", email = "jared.lewis@climate-energy-college.org"}
]

[project.urls]
Homepage = "https://github.com/lewisjared/rscm"
Repository = "https://github.com/lewisjared/rscm"
Documentation = "https://lewisjared.github.io/rscm/"
```

## Alternatives Considered

### PyPI Authentication

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Trusted Publishers (OIDC) | No tokens, short-lived, secure | Requires PyPI setup | **Selected** |
| API Token | Simple setup | Long-lived secret, less secure | Rejected |

### Cross-Platform Builds

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| maturin-action matrix | Official, simple | Multiple jobs | **Selected** |
| cibuildwheel | Popular, comprehensive | More complex setup | Rejected |
| Single-platform only | Fastest | Limited user base | Rejected |

### Workspace Publishing

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| katyo/publish-crates | Automatic ordering, skip existing | External action | **Selected** |
| Manual cargo publish | Full control | Complex ordering logic needed | Rejected |

## Sources

- [PyO3/maturin-action](https://github.com/PyO3/maturin-action)
- [Maturin Distribution Guide](https://www.maturin.rs/distribution.html)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [katyo/publish-crates](https://github.com/katyo/publish-crates)
- [crates.io Registry API](https://doc.rust-lang.org/cargo/reference/registry-web-api.html)
