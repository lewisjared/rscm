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

## 5. Extract two_layer into separate crate

- [ ] 5.1 Create `crates/rscm-two-layer/` directory structure
- [ ] 5.2 Create `crates/rscm-two-layer/Cargo.toml`
- [ ] 5.3 Move `two_layer.rs` to `crates/rscm-two-layer/src/component.rs`
- [ ] 5.4 Create `crates/rscm-two-layer/src/lib.rs` exporting component
- [ ] 5.5 Create `crates/rscm-two-layer/src/python/mod.rs` with PyO3 submodule
- [ ] 5.6 Add `rscm-two-layer` to workspace dependencies

## 6. Create rscm-magicc scaffold

- [ ] 6.1 Create `crates/rscm-magicc/` directory structure
- [ ] 6.2 Create `crates/rscm-magicc/Cargo.toml`
- [ ] 6.3 Create `crates/rscm-magicc/src/lib.rs` with module declarations
- [ ] 6.4 Create empty domain subdirectories (`chemistry/`, `forcing/`, `climate/`, `carbon/`)
- [ ] 6.5 Create `crates/rscm-magicc/src/python/mod.rs` scaffold

## 7. Update main rscm crate for submodule registration

- [ ] 7.1 Update `crates/rscm/Cargo.toml` dependencies
- [ ] 7.2 Update `crates/rscm/src/lib.rs` to register two_layer submodule
- [ ] 7.3 Update `crates/rscm/src/python/mod.rs` for new module structure
- [ ] 7.4 Remove `TwoLayerComponentBuilder` from root exports

## 8. Create Python namespace packages

- [ ] 8.1 Create `python/rscm/two_layer/__init__.py`
- [ ] 8.2 Create `python/rscm/magicc/__init__.py` (empty scaffold)
- [ ] 8.3 Update `python/rscm/__init__.py` to remove `TwoLayerComponentBuilder`
- [ ] 8.4 Update type stubs (`python/rscm/_lib/*.pyi`) for new structure

## 9. Update build configuration

- [ ] 9.1 Update `pyproject.toml` with `manifest-path = "crates/rscm/Cargo.toml"`
- [ ] 9.2 Update `Makefile` if paths are hardcoded
- [ ] 9.3 Update `.bumpversion.toml` file paths for new crate locations

## 10. Update documentation and tooling

- [ ] 10.1 Update `CLAUDE.md` workspace structure section
- [ ] 10.2 Update `openspec/project.md` architecture section
- [ ] 10.3 Update README.md if it references directory structure

## 11. Testing and validation

- [ ] 11.1 Run `cargo build --workspace` and fix any path issues
- [ ] 11.2 Run `cargo test --workspace` and fix any test failures
- [ ] 11.3 Run `make build-dev` and verify Python extension builds
- [ ] 11.4 Run `uv run pytest` and fix any import errors
- [ ] 11.5 Verify `from rscm.two_layer import TwoLayerComponentBuilder` works
- [ ] 11.6 Run `make lint` and fix any issues
- [ ] 11.7 Run `make format` to ensure consistent formatting

## 12. Create changelog entry

- [ ] 12.1 Create `changelog/XX.breaking.md` documenting import path change
