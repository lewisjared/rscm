# Proposal: Support Grid Values in OutputState

## Summary

Enable components to return spatially-resolved (grid) output values natively, instead of requiring aggregation to scalars.
This completes the grid support story: inputs can already be grid-based, but outputs are currently limited to scalar values only.

## Motivation

Currently, components that compute regional values must aggregate them to a global scalar before returning,
losing valuable spatial information. For example, `FourBoxOceanHeatUptakeComponent` computes regional uptake
but must aggregate to global because `OutputState = HashMap<String, FloatValue>` only supports scalars.

This limitation:

1. **Loses spatial resolution** - Regional detail computed inside components cannot flow to downstream components
2. **Prevents grid-to-grid coupling** - A component producing FourBox output cannot feed a component expecting FourBox input without external intervention
3. **Requires workarounds** - Components must aggregate internally, duplicating logic that belongs in the model layer
4. **Blocks MAGICC implementation** - MAGICC's carbon cycle and climate response components require spatially-resolved state flow

## Scope

This change affects:

- `rscm-core/src/state.rs`: `OutputState` type definition and `StateValue` enum
- `rscm-core/src/model.rs`: Model step logic to write grid outputs to timeseries
- `rscm-macros/src/lib.rs`: Code generation for `From<Outputs> for OutputState`
- `rscm-core/src/model.rs`: `ModelBuilder` initial values support
- Existing components that return grid outputs (currently aggregating)

## Approach

### OutputState Type Change

Change from:
```rust
pub type OutputState = HashMap<String, FloatValue>;
```

To:
```rust
pub type OutputState = HashMap<String, StateValue>;
```

### StateValue Enum Extension

Extend `StateValue` with grid-specific variants:
```rust
pub enum StateValue {
    Scalar(FloatValue),
    FourBox(FourBoxSlice),
    Hemispheric(HemisphericSlice),
}
```

This provides:
- Type safety at the enum level
- Zero-cost conversion from typed slices
- Explicit grid type matching in Model.step_model_component

### ComponentIO Macro Update

The macro will generate `From<Outputs> for OutputState` that preserves grid types:

```rust
// For FourBox output:
impl From<MyOutputs> for OutputState {
    fn from(outputs: MyOutputs) -> Self {
        let mut map = HashMap::new();
        map.insert("Heat Flux".to_string(), StateValue::FourBox(outputs.heat_flux));
        map
    }
}
```

### Model Integration

`Model.step_model_component` will match on `StateValue` variants:
```rust
match value {
    StateValue::Scalar(v) => ts.set(index, ScalarRegion::Global, v),
    StateValue::FourBox(slice) => {
        for (region, val) in slice.iter_regions() {
            ts.set(index, region, val);
        }
    },
    StateValue::Hemispheric(slice) => { /* similar */ },
}
```

### Initial Values Extension

Extend `ModelBuilder::initial_values` to support grid types:
```rust
initial_values: HashMap<String, StateValue>
```

With convenience methods:
```rust
fn with_initial_value(&mut self, name: &str, value: FloatValue) -> &mut Self
fn with_initial_four_box(&mut self, name: &str, values: FourBoxSlice) -> &mut Self
fn with_initial_hemispheric(&mut self, name: &str, values: HemisphericSlice) -> &mut Self
```

## Migration

This is a **breaking change** for components returning `OutputState` directly.
Components using the `ComponentIO` derive macro will be updated automatically.
Manual component implementations need migration:

```rust
// Before:
let mut output = HashMap::new();
output.insert("Temperature".to_string(), 288.0);

// After:
let mut output = HashMap::new();
output.insert("Temperature".to_string(), StateValue::Scalar(288.0));
```

## Non-Goals

- Grid transformations in the model layer (e.g., auto-convert FourBox to Scalar) - out of scope, explicit transformations required
- Python bindings for grid OutputState - will be addressed in separate change
- Runtime grid type validation beyond compile-time checks - existing `GridType` mismatch detection is sufficient

## Dependencies

- Requires: `grid-timeseries` spec (complete)
- Requires: `component-state` spec (complete, will be updated)

## Success Criteria

1. Components can return `FourBoxSlice` and `HemisphericSlice` outputs
2. Model correctly writes grid outputs to appropriate timeseries
3. Grid outputs flow between components without aggregation
4. Existing scalar-only components continue to work after migration
5. All tests pass with explicit migration to new pattern
