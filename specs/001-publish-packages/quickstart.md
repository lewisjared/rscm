# Quickstart: Publishing RSCM Releases

**Feature**: 001-publish-packages
**Date**: 2026-01-14

## Prerequisites

Before your first release, complete these one-time setup steps:

### 1. Configure PyPI Trusted Publisher

1. Go to <https://pypi.org/manage/project/rscm/settings/publishing/>
2. Click "Add a new pending publisher" (or "Add publisher" if package exists)
3. Fill in:
   - Owner: `lewisjared`
   - Repository name: `rscm`
   - Workflow name: `release.yaml`
   - Environment name: `release`
4. Click "Add"

### 2. Create GitHub Environment

1. Go to Repository → Settings → Environments
2. Click "New environment"
3. Name: `release`
4. (Optional) Add deployment protection rules

### 3. Configure crates.io Token

1. Run locally:

   ```bash
   cargo login
   ```

2. Copy token from `~/.cargo/credentials.toml`
3. Go to Repository → Settings → Secrets and variables → Actions
4. Create secret: `CARGO_REGISTRY_TOKEN` with the token value

## Publishing a Release

### Step 1: Prepare Changes

Ensure all changes are merged to `main` and CI is passing.

### Step 2: Create Changelog Fragments

Add changelog entries in `changelog/` directory:

```bash
# Example: for PR #47
echo "Added support for publishing to PyPI and crates.io." > changelog/47.feature.md
```

### Step 3: Trigger Version Bump

Run the bump workflow:

```bash
# For a patch release (0.1.0 → 0.1.1)
gh workflow run bump.yaml -f bump_rule=patch

# For a minor release (0.1.0 → 0.2.0)
gh workflow run bump.yaml -f bump_rule=minor

# For a major release (0.1.0 → 1.0.0)
gh workflow run bump.yaml -f bump_rule=major

# For a pre-release (0.1.0 → 0.2.0-alpha.0)
gh workflow run bump.yaml -f bump_rule=prerelease
```

This will:

1. Run `bump-my-version` to update all version files
2. Generate changelog via towncrier
3. Create a commit and tag
4. Push the tag to GitHub

### Step 4: Monitor Release Workflow

The tag push triggers the release workflow automatically.

Monitor at: <https://github.com/lewisjared/rscm/actions/workflows/release.yaml>

The workflow will:

1. Build wheels for all platforms (Linux, macOS, Windows)
2. Publish Rust crates to crates.io (rscm-core, then rscm-components)
3. Publish Python package to PyPI
4. Create a GitHub release with artifacts

### Step 5: Verify Publication

After the workflow completes:

- **crates.io**: <https://crates.io/crates/rscm-core>
- **PyPI**: <https://pypi.org/project/rscm/>
- **GitHub Releases**: <https://github.com/lewisjared/rscm/releases>

## Troubleshooting

### "Version already exists" Error

If you see this error, the version was already published. This can happen if:

- A previous release attempt partially succeeded
- The version was manually published

**Resolution**: Bump to a new version and retry.

### crates.io Publish Fails

Common causes:

- `CARGO_REGISTRY_TOKEN` secret is missing or expired
- Required metadata missing from Cargo.toml
- Dependency not yet available (propagation delay)

**Resolution**:

1. Check token validity: `cargo login` and update secret if needed
2. Verify metadata: `cargo publish --dry-run`
3. Wait 5-10 minutes and retry if dependency issue

### PyPI Publish Fails

Common causes:

- Trusted publisher not configured correctly
- `id-token: write` permission missing
- Environment name mismatch

**Resolution**:

1. Verify trusted publisher settings on PyPI
2. Check workflow has correct environment and permissions
3. Ensure `release` environment exists in GitHub

### Build Fails for Specific Platform

**Resolution**:

1. Check the failed job logs
2. Verify the target is correctly configured
3. For Linux aarch64, ensure manylinux is set to `auto`

## Local Testing

Before pushing a release, you can test locally:

```bash
# Test Rust crate publishing (dry run)
cargo publish --dry-run -p rscm-core
cargo publish --dry-run -p rscm-components

# Test wheel building
uv run maturin build --release

# Verify wheel contents
unzip -l target/wheels/*.whl
```

## Release Checklist

- [ ] All tests passing on main branch
- [ ] Changelog fragments added for all changes
- [ ] Version bump type selected (patch/minor/major/prerelease)
- [ ] Bump workflow triggered
- [ ] Release workflow completed successfully
- [ ] Packages visible on crates.io and PyPI
- [ ] GitHub release created with artifacts
