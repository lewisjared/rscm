# Add Python Grid Output Support

## Summary

Enable Python components to return grid outputs (FourBox/Hemispheric) that are written directly to grid timeseries, matching the Rust component behaviour. Currently, Python component outputs are always converted to scalars, which loses spatial resolution.

## Problem Statement

The Rust core fully supports grid-based outputs via `StateValue::FourBox` and `StateValue::Hemispheric`. However, the Python bindings aggregate all component outputs to scalars:

1. **PythonComponent wrapper** (`crates/rscm-core/src/python/component.rs:169-174`): Converts Python dict outputs to `StateValue::Scalar` only
2. **RustComponent.solve()** (`crates/rscm-core/src/python/component.rs:68-73`): Returns `HashMap<String, FloatValue>` with scalar values only
3. **Python Outputs.to_dict()** (`python/rscm/component.py:357-361`): Converts slices to lists but Rust side ignores grid structure

This prevents Python components from producing spatially-resolved outputs and breaks the symmetry between Rust and Python components.

## Goals

1. Python components can return `FourBoxSlice` or `HemisphericSlice` outputs that are written to grid timeseries
2. `RustComponent.solve()` returns grid-aware values that preserve spatial structure
3. Maintain backwards compatibility with existing scalar-only Python components
4. Type stubs accurately reflect the new return types

## Non-Goals

- Changing how grid inputs work (already functional)
- Adding new grid types beyond FourBox and Hemispheric
- Modifying the Rust core `OutputState` type (already supports grids)

## Approach

Expose `StateValue` as a native PyO3 class (`PyStateValue`) with factory methods and accessors. This is the most type-safe option and aligns with how `FourBoxSlice` and `HemisphericSlice` are already exposed.

**Breaking changes:**
- `RustComponent.solve()` return type changes from `dict[str, float]` to `dict[str, StateValue]`
- Existing code must call `.to_scalar()` or `.as_scalar()` to get float values

## Stakeholders

- Python component authors who need spatial resolution in outputs
- Model users accessing grid timeseries results from Python
