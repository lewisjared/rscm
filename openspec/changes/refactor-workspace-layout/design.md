## Prerequisites

This change depends on the `dx` branch (improve-component-dx) being merged first. That branch introduces `rscm-macros/` which will be moved to `crates/rscm-macros/` as part of this restructure.

## Context

RSCM is growing from a simple prototype to a full MAGICC reimplementation. The current flat workspace structure with `rscm-core/`, `rscm-components/`, and root `src/` for PyO3 bindings works but doesn't scale well. Adding 16+ MAGICC components requires clear separation between the framework and model implementations.

Key stakeholders:
- Climate scientists using Python API
- Rust developers adding components
- Future maintainers extracting rscm-magicc

## Goals / Non-Goals

**Goals:**
- Establish clear workspace organisation that scales to 5+ crates
- Enable future extraction of rscm-magicc as standalone package
- Provide intuitive Python namespace for accessing model components
- Maintain single pip install for end users

**Non-Goals:**
- Implement MAGICC components (scaffold only)
- Change core Component/Model APIs
- Support multiple Python extension modules (single _lib remains)

## Decisions

### Directory Structure

```
rscm/
├── Cargo.toml                 # Workspace-only (no [package])
├── pyproject.toml             # manifest-path = "crates/rscm/Cargo.toml"
├── python/
│   └── rscm/
│       ├── __init__.py        # Core exports only
│       ├── two_layer/
│       │   └── __init__.py    # from rscm._lib.two_layer import *
│       └── magicc/
│           └── __init__.py    # from rscm._lib.magicc import * (future)
├── crates/
│   ├── rscm/                  # PyO3 extension module
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs         # #[pymodule] _lib
│   │       └── python/
│   ├── rscm-core/             # Core traits and abstractions
│   │   └── src/
│   │       ├── spatial/       # Split from spatial.rs (923 lines)
│   │       │   ├── mod.rs     # SpatialGrid trait, re-exports
│   │       │   ├── scalar.rs  # ScalarGrid, ScalarRegion
│   │       │   ├── four_box.rs # FourBoxGrid, FourBoxRegion
│   │       │   └── hemispheric.rs
│   │       └── python/
│   │           └── spatial/   # Mirrors Rust structure
│   │               ├── mod.rs
│   │               ├── scalar.rs
│   │               ├── four_box.rs
│   │               └── hemispheric.rs
│   ├── rscm-components/       # Shared/generic components
│   ├── rscm-macros/           # Proc macros (from dx branch)
│   ├── rscm-two-layer/        # Two-layer model component
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── component.rs   # TwoLayerComponent impl
│   │       └── python/
│   │           └── mod.rs     # PyO3 submodule
│   └── rscm-magicc/           # MAGICC components (scaffold)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── chemistry/
│           ├── forcing/
│           ├── climate/
│           ├── carbon/
│           └── python/
│               └── mod.rs
```

**Rationale:** The `crates/` convention is standard in Rust monorepos. It clearly separates workspace configuration from crate source code and makes adding new crates intuitive.

### Python Submodule Registration

Each component crate exposes a `#[pymodule]` that gets registered as a submodule of `_lib`:

```rust
// crates/rscm/src/lib.rs
#[pymodule]
#[pyo3(name = "_lib")]
fn rscm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(core))?;
    m.add_wrapped(wrap_pymodule!(two_layer))?;  // from rscm-two-layer
    m.add_wrapped(wrap_pymodule!(magicc))?;     // from rscm-magicc
    // ...
}
```

Python packages then re-export from these submodules:

```python
# python/rscm/two_layer/__init__.py
from rscm._lib.two_layer import TwoLayerComponentBuilder

__all__ = ["TwoLayerComponentBuilder"]
```

**Rationale:** This keeps a single compiled extension (`_lib`) for simplicity while allowing logical Python namespaces. Each crate maintains its own Python bindings, making extraction straightforward.

### Spatial Module Split

The `spatial.rs` file (923 lines) contains three independent grid implementations plus the `SpatialGrid` trait. Splitting into a subdirectory:

```
spatial/
├── mod.rs           # SpatialGrid trait, module docs (~310 lines), re-exports
├── scalar.rs        # ScalarRegion, ScalarGrid (~75 lines)
├── four_box.rs      # FourBoxRegion, FourBoxGrid (~180 lines)
└── hemispheric.rs   # HemisphericRegion, HemisphericGrid (~145 lines)
```

**Rationale:**
- Each grid type is self-contained with its own region enum, struct, and `SpatialGrid` impl
- Tests can live alongside each grid type or in a separate `tests.rs`
- Adding new grid types (e.g., `LatitudinalGrid`) becomes straightforward
- Python bindings mirror this structure for consistency

The `python/spatial.rs` (206 lines) follows the same split pattern, keeping PyO3 wrappers adjacent to their Rust counterparts.

### Testing Module Rename

The `example_components.rs` module contains `TestComponent` used for testing the component infrastructure. This will be renamed to `testing.rs` and the Python bindings moved from `rscm._lib.core` to `rscm._lib.testing`:

```rust
// crates/rscm-core/src/testing.rs (renamed from example_components.rs)
pub struct TestComponent { ... }
pub struct TestComponentParameters { ... }

// crates/rscm-core/src/python/testing.rs (renamed from example_component.rs)
create_component_builder!(TestComponentBuilder, TestComponent, TestComponentParameters);

#[pymodule]
pub fn testing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TestComponentBuilder>()?;
    Ok(())
}
```

**Rationale:** Separating test utilities from core exports clarifies their purpose and keeps the `core` module focused on production types.

### Alternatives Considered

1. **Multiple extension modules (rscm._core, rscm._two_layer):** Would complicate installation and require multiple maturin builds. Rejected for added complexity without clear benefit.

2. **Keep flat structure with naming conventions:** Doesn't solve the extraction goal and becomes unwieldy with 20+ source files in root.

3. **Workspaces of workspaces (rscm-models/ workspace):** Over-engineering at current scale. Can revisit if needed.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Breaking Python imports for `TwoLayerComponentBuilder` | Document migration in changelog; deprecation warning in v0.2.x |
| Build time increase from more crates | Minimal impact; Rust incremental compilation handles well |
| Complexity for contributors | Clear CONTRIBUTING.md section; consistent pattern across crates |

## Migration Plan

1. Create `crates/` directory and move existing crates
2. Update all `Cargo.toml` path dependencies
3. Split `rscm-core/src/spatial.rs` into `spatial/` subdirectory
4. Split `rscm-core/src/python/spatial.rs` to mirror Rust structure
5. Create `rscm-two-layer` crate, move `two_layer.rs`
6. Create `rscm-magicc` scaffold
7. Update `pyproject.toml` manifest-path
8. Create Python subpackages (`python/rscm/two_layer/`)
9. Update `python/rscm/__init__.py` exports
10. Update `.bumpversion.toml` paths
11. Run full test suite
12. Update documentation

**Rollback:** Git revert. No data migrations required.

## Open Questions

- Should `rscm-components` be renamed to `rscm-shared-components` for clarity?
- Do we want a `rscm-utils` crate for common helpers, or keep them in `rscm-core`?
