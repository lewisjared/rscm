## Prerequisites

This change depends on the `dx` branch (improve-component-dx) being merged first. The dx branch introduces the `rscm-macros/` crate which will be moved as part of this restructure.

## 1. Create crates directory and move existing crates

- [ ] 1.1 Create `crates/` directory
- [ ] 1.2 Move `rscm-core/` to `crates/rscm-core/`
- [ ] 1.3 Move `rscm-components/` to `crates/rscm-components/`
- [ ] 1.4 Move `rscm-macros/` to `crates/rscm-macros/`
- [ ] 1.5 Move `src/` to `crates/rscm/src/`
- [ ] 1.6 Create `crates/rscm/Cargo.toml` from root package config

## 2. Update root Cargo.toml to workspace-only

- [ ] 2.1 Remove `[package]` section from root `Cargo.toml`
- [ ] 2.2 Remove `[lib]` and `[dependencies]` sections
- [ ] 2.3 Update `[workspace].members` to use `crates/*` paths
- [ ] 2.4 Update `[workspace.dependencies]` paths

## 3. Split spatial module into subdirectory

- [ ] 3.1 Create `crates/rscm-core/src/spatial/` directory
- [ ] 3.2 Create `spatial/mod.rs` with `SpatialGrid` trait and module docs
- [ ] 3.3 Create `spatial/scalar.rs` with `ScalarRegion`, `ScalarGrid`
- [ ] 3.4 Create `spatial/four_box.rs` with `FourBoxRegion`, `FourBoxGrid`
- [ ] 3.5 Create `spatial/hemispheric.rs` with `HemisphericRegion`, `HemisphericGrid`
- [ ] 3.6 Move tests to appropriate files or `spatial/tests.rs`
- [ ] 3.7 Update `lib.rs` to use `pub mod spatial;` (directory module)
- [ ] 3.8 Verify all re-exports work (`use rscm_core::spatial::*`)

## 4. Split python/spatial module to mirror Rust

- [ ] 4.1 Create `crates/rscm-core/src/python/spatial/` directory
- [ ] 4.2 Create `python/spatial/mod.rs` with module registration
- [ ] 4.3 Create `python/spatial/scalar.rs` with `PyScalarRegion`, `PyScalarGrid`
- [ ] 4.4 Create `python/spatial/four_box.rs` with `PyFourBoxRegion`, `PyFourBoxGrid`
- [ ] 4.5 Create `python/spatial/hemispheric.rs` with `PyHemisphericRegion`, `PyHemisphericGrid`
- [ ] 4.6 Update `python/mod.rs` to use directory module

## 5. Rename example_components to testing module

- [ ] 5.1 Rename `crates/rscm-core/src/example_components.rs` to `testing.rs`
- [ ] 5.2 Rename `crates/rscm-core/src/python/example_component.rs` to `testing.rs`
- [ ] 5.3 Create `#[pymodule] testing` in `python/testing.rs`
- [ ] 5.4 Update `crates/rscm-core/src/lib.rs` module declaration
- [ ] 5.5 Update `crates/rscm-core/src/python/mod.rs` to register `testing` submodule
- [ ] 5.6 Update `crates/rscm/src/python/mod.rs` to register `rscm._lib.testing`
- [ ] 5.7 Update `tests/test_example_component.py` import to use `rscm._lib.testing`
- [ ] 5.8 Update `python/rscm/_lib/core.pyi` to remove `TestComponentBuilder`
- [ ] 5.9 Create `python/rscm/_lib/testing.pyi` with `TestComponentBuilder`

## 6. Extract two_layer into separate crate

- [ ] 6.1 Create `crates/rscm-two-layer/` directory structure
- [ ] 6.2 Create `crates/rscm-two-layer/Cargo.toml`
- [ ] 6.3 Move `two_layer.rs` to `crates/rscm-two-layer/src/component.rs`
- [ ] 6.4 Create `crates/rscm-two-layer/src/lib.rs` exporting component
- [ ] 6.5 Create `crates/rscm-two-layer/src/python/mod.rs` with PyO3 submodule
- [ ] 6.6 Add `rscm-two-layer` to workspace dependencies

## 7. Create rscm-magicc scaffold

- [ ] 7.1 Create `crates/rscm-magicc/` directory structure
- [ ] 7.2 Create `crates/rscm-magicc/Cargo.toml`
- [ ] 7.3 Create `crates/rscm-magicc/src/lib.rs` with module declarations
- [ ] 7.4 Create empty domain subdirectories (`chemistry/`, `forcing/`, `climate/`, `carbon/`)
- [ ] 7.5 Create `crates/rscm-magicc/src/python/mod.rs` scaffold

## 8. Update main rscm crate for submodule registration

- [ ] 8.1 Update `crates/rscm/Cargo.toml` dependencies
- [ ] 8.2 Update `crates/rscm/src/lib.rs` to register two_layer and testing submodules
- [ ] 8.3 Update `crates/rscm/src/python/mod.rs` for new module structure
- [ ] 8.4 Remove `TwoLayerComponentBuilder` from root exports

## 9. Create Python namespace packages

- [ ] 9.1 Create `python/rscm/two_layer/__init__.py`
- [ ] 9.2 Create `python/rscm/magicc/__init__.py` (empty scaffold)
- [ ] 9.3 Update `python/rscm/__init__.py` to remove `TwoLayerComponentBuilder`
- [ ] 9.4 Update type stubs (`python/rscm/_lib/*.pyi`) for new structure

## 10. Update build configuration

- [ ] 10.1 Update `pyproject.toml` with `manifest-path = "crates/rscm/Cargo.toml"`
- [ ] 10.2 Update `Makefile` if paths are hardcoded
- [ ] 10.3 Update `.bumpversion.toml` file paths for new crate locations

## 11. Update documentation and tooling

- [ ] 11.1 Update `CLAUDE.md` workspace structure section
- [ ] 11.2 Update `openspec/project.md` architecture section
- [ ] 11.3 Update README.md if it references directory structure

## 12. Testing and validation

- [ ] 12.1 Run `cargo build --workspace` and fix any path issues
- [ ] 12.2 Run `cargo test --workspace` and fix any test failures
- [ ] 12.3 Run `make build-dev` and verify Python extension builds
- [ ] 12.4 Run `uv run pytest` and fix any import errors
- [ ] 12.5 Verify `from rscm.two_layer import TwoLayerComponentBuilder` works
- [ ] 12.6 Verify `from rscm._lib.testing import TestComponentBuilder` works
- [ ] 12.7 Run `make lint` and fix any issues
- [ ] 12.8 Run `make format` to ensure consistent formatting

## 13. Create changelog entry

- [ ] 13.1 Create `changelog/XX.breaking.md` documenting import path changes
