# Design: Support Grid Values in OutputState

## Architectural Decision Record

### Context

The RSCM framework supports spatially-resolved (grid) timeseries for inputs via `GridTimeseries<T, G>`,
`TimeseriesWindow`, and `GridTimeseriesWindow`. However, `OutputState` is currently defined as
`HashMap<String, FloatValue>`, limiting component outputs to scalars only.

Components that compute regional values (e.g., FourBox ocean heat uptake) must aggregate to global
scalars before returning, losing spatial resolution that downstream components may need.

### Decision: Grid-Specific StateValue Variants

**Chosen approach:** Extend `StateValue` with explicit grid-type variants.

```rust
pub enum StateValue {
    Scalar(FloatValue),
    FourBox(FourBoxSlice),
    Hemispheric(HemisphericSlice),
}
```

**Rationale:**

1. **Type safety** - Grid type is explicit at the enum level, enabling exhaustive matching
2. **Zero-cost abstraction** - `FourBoxSlice` and `HemisphericSlice` are `#[repr(transparent)]` over arrays
3. **Consistency** - Mirrors `TimeseriesData` enum which has `Scalar`, `FourBox`, `Hemispheric` variants
4. **Clear error messages** - Mismatched grid types are explicit rather than hidden in Vec lengths

**Alternatives considered:**

1. **`StateValue::Grid(Vec<FloatValue>)`** - Simpler enum, but loses type information. Grid type would need
   to be inferred from `RequirementDefinition`, adding runtime coupling and error potential.

2. **Separate output types per grid** - `ScalarOutputState`, `FourBoxOutputState`, etc. Would require
   generic components or trait objects, significantly complicating the Component trait.

### Decision: Explicit Migration (Breaking Change)

**Chosen approach:** Make this a breaking change requiring explicit migration.

**Rationale:**

1. **Clarity** - All code paths explicitly use `StateValue`, no hidden conversions
2. **Type safety** - Compiler catches all places needing updates
3. **Clean codebase** - No deprecated compatibility shims to maintain
4. **Small surface area** - Only affects `OutputState` construction, not component logic

**Alternatives considered:**

1. **Auto-wrap scalars** - Implement `From<FloatValue> for StateValue` and have HashMap insertion
   auto-convert. Rejected because it hides the type system change and may cause subtle bugs where
   grid values are accidentally treated as scalars.

### Decision: Full Macro Support

**Chosen approach:** Update `ComponentIO` macro to generate grid-aware conversions.

The macro currently generates:

```rust
impl From<MyOutputs> for OutputState {
    fn from(outputs: MyOutputs) -> Self {
        let mut map = HashMap::new();
        // Grid outputs: aggregate to scalar
        let mean = outputs.heat_flux.as_array().iter().sum::<f64>() / 4.0;
        map.insert("Heat Flux".to_string(), mean);
        map
    }
}
```

Will generate:

```rust
impl From<MyOutputs> for OutputState {
    fn from(outputs: MyOutputs) -> Self {
        let mut map = HashMap::new();
        // Grid outputs: preserve as StateValue::FourBox
        map.insert("Heat Flux".to_string(), StateValue::FourBox(outputs.heat_flux));
        map
    }
}
```

**Rationale:**

1. **End-to-end type safety** - Components define grid type in macro, flows through to output
2. **DRY** - Grid type declared once, used consistently
3. **Compile-time validation** - Wrong slice type in struct fails compilation

### Decision: Grid Initial Values Support

**Chosen approach:** Extend `ModelBuilder::initial_values` to `HashMap<String, StateValue>`.

**Rationale:**

1. **Consistency** - State variables can be grid-typed, their initial values should match
2. **Completeness** - Without this, grid state variables cannot be initialised
3. **Simple extension** - Uses same `StateValue` enum, no new types needed

## Component Interactions

### State Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Model.step_model_component                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Extract InputState from TimeseriesCollection                         │
│     ├─ Scalar vars → TimeseriesWindow                                    │
│     ├─ FourBox vars → GridTimeseriesWindow<FourBoxGrid>                  │
│     └─ Hemispheric vars → GridTimeseriesWindow<HemisphericGrid>          │
│                                                                          │
│  2. Component.solve(t_current, t_next, &input_state)                     │
│     ├─ Access inputs via windows                                         │
│     ├─ Compute outputs (scalar or grid)                                  │
│     └─ Return OutputState: HashMap<String, StateValue>                   │
│                                                                          │
│  3. Write OutputState to TimeseriesCollection                            │
│     ├─ StateValue::Scalar(v) → ts.set(index, ScalarRegion::Global, v)    │
│     ├─ StateValue::FourBox(s) → ts.set_four_box(index, s)  [NEW]         │
│     └─ StateValue::Hemispheric(s) → ts.set_hemispheric(index, s)  [NEW]  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Type Matching

The `RequirementDefinition.grid_type` from component definitions is used by `ModelBuilder` to create
the correct `TimeseriesData` variant. When writing outputs, the `StateValue` variant must match:

| RequirementDefinition.grid_type | TimeseriesData variant      | Expected StateValue     |
| ------------------------------- | --------------------------- | ----------------------- |
| GridType::Scalar                | TimeseriesData::Scalar      | StateValue::Scalar      |
| GridType::FourBox               | TimeseriesData::FourBox     | StateValue::FourBox     |
| GridType::Hemispheric           | TimeseriesData::Hemispheric | StateValue::Hemispheric |

Grid type mismatches between component output and StateValue will result in runtime errors with
clear messages indicating the mismatch.

## Implementation Notes

### GridTimeseries.set_all Method

Need to add a method to set all regional values at once:

```rust
impl<T, G: SpatialGrid> GridTimeseries<T, G> {
    pub fn set_all(&mut self, time_index: usize, values: &[T]) {
        assert_eq!(values.len(), self.grid().size());
        for (region_index, value) in values.iter().enumerate() {
            self.values[[time_index, region_index]] = *value;
        }
        if time_index > self.latest {
            self.latest = time_index;
        }
    }
}
```

### Error Handling

Grid type mismatches at runtime should produce clear errors:

```rust
RSCMError::GridOutputMismatch {
    variable: String,
    expected_grid: String,  // From TimeseriesData variant
    actual_grid: String,    // From StateValue variant
}
```

## Testing Strategy

1. **Unit tests** - StateValue conversion, FourBoxSlice/HemisphericSlice to StateValue
2. **Integration tests** - Full model run with grid-producing component feeding grid-consuming component
3. **Migration tests** - Verify existing components work after explicit StateValue wrapping
4. **Error case tests** - Grid type mismatch detection and error messages
