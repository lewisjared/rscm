# Design: Schema-Driven Grid Auto-Aggregation

## Context

Climate models often work at different spatial resolutions. Some components produce regional outputs (FourBox), while others only need global averages (Scalar). Currently, connecting these requires either:

1. Duplicate variable names per resolution
2. Manual transformer components
3. Modifying components to match grids

The VariableSchema already declares grid types for variables. We can extend this to support automatic grid transformation.

**Constraints:**

- Must not break existing models (backwards compatible)
- Aggregation weights must be physically meaningful (area-based)
- Performance overhead should be minimal
- Timestep semantics must be preserved (caller controls `at_start()` vs `at_end()`)

## Decision

### Schema as Source of Truth

The VariableSchema declares the "native" grid type for each variable. This is the resolution at which the variable is stored.

```rust
let schema = VariableSchema::new()
    .variable_with_grid("Surface Temperature", "K", GridType::FourBox)
    .variable("Atmospheric Concentration|CO2", "ppm");  // Scalar default
```

### Auto-Aggregation (Read and Write)

The system provides symmetric aggregation for both inputs and outputs:

**Read-side (inputs):** When a component declares an input at a coarser resolution than the schema, InputState aggregates data when the component reads it.

**Write-side (outputs):** When a component produces output at a finer resolution than the schema, the Model aggregates data when writing to the collection.

**Valid transformations (aggregation only):**

| Component Grid | Schema Grid   | Direction | Action         |
|----------------|---------------|-----------|----------------|
| Scalar         | FourBox       | Read      | Error (no broadcast) |
| Scalar         | Hemispheric   | Read      | Error (no broadcast) |
| FourBox        | Scalar        | Read      | Aggregate      |
| FourBox        | Hemispheric   | Read      | Aggregate      |
| Hemispheric    | Scalar        | Read      | Aggregate      |
| FourBox        | Scalar        | Write     | Aggregate      |
| FourBox        | Hemispheric   | Write     | Aggregate      |
| Hemispheric    | Scalar        | Write     | Aggregate      |
| Any            | Same          | Either    | Direct (no-op) |

**Invalid transformations (error):**

| Component Grid | Schema Grid   | Direction | Reason                    |
|----------------|---------------|-----------|---------------------------|
| Scalar         | FourBox       | Read      | Cannot disaggregate       |
| Scalar         | Hemispheric   | Read      | Cannot disaggregate       |
| Hemispheric    | FourBox       | Read/Write| Cannot disaggregate       |
| Scalar         | FourBox       | Write     | Cannot disaggregate       |
| Scalar         | Hemispheric   | Write     | Cannot disaggregate       |

**Key principle:** Aggregation (coarsening) is always allowed. Disaggregation (broadcast) is never implicit.

### Weight Configuration

Aggregation requires area weights. These come from the grid configuration:

1. **Default weights** - Each grid type has standard defaults:
   - `FourBoxGrid::magicc_standard()` - equal weights [0.25, 0.25, 0.25, 0.25]
   - `HemisphericGrid::default()` - equal weights [0.5, 0.5]

2. **Custom weights via ModelBuilder**:

   ```rust
   let model = ModelBuilder::new()
       .with_schema(schema)
       .with_grid_weights(GridType::FourBox, [0.36, 0.14, 0.36, 0.14])  // area-based
       .with_component(...)
       .build()?;
   ```

3. **Weights stored in Model** - Available for all transformations during execution.

### Transform on Read (InputState)

Transformations happen at access time in InputState, not via virtual components:

```rust
// Component declares scalar input for "Temperature"
// Schema says "Temperature" is FourBox
// In component's solve():
let inputs = MyComponentInputs::from_input_state(input_state);
let temp = inputs.temperature.at_start();  // Returns aggregated scalar
```

**Implementation:**

1. InputState holds a reference to the Model's grid weights and required transformations
2. When `get_scalar_window(name)` is called for a FourBox variable:
   - InputState detects the mismatch from the transformation registry
   - Returns a `TimeseriesWindow` that lazily aggregates on access
3. The caller's choice of `at_start()` or `at_end()` determines which timestep index is aggregated

**Benefits:**

- No timestep complexity at transformation level - caller specifies via `at_start()`/`at_end()`
- No intermediate variable names (`|_to_scalar`)
- No input remapping - component reads the variable it declared
- Simpler component graph - no extra nodes

### Transform on Write (Model.step)

Write-side transformations happen in the Model's step function:

```rust
// Component produces FourBox output for "Heat Flux"
// Schema declares "Heat Flux" as Scalar
// In Model.step():
for (name, value) in component_output {
    if let Some(transform) = self.write_transforms.get(&name) {
        let aggregated = transform.apply(value, &self.grid_weights);
        collection.set(name, aggregated);
    } else {
        collection.set(name, value);
    }
}
```

### Transformation Tracking

Required transformations are recorded during model build for:

1. **Validation** - Ensure only valid aggregations are requested
2. **Runtime execution** - InputState and Model know which transforms to apply
3. **Introspection** - `model.required_transformations()` returns the list for debugging

## Alternatives Rejected

### Virtual Transformer Components

Initially considered inserting `GridTransformerComponent` nodes into the component graph (similar to `AggregatorComponent`). Rejected because:

- **Timestep complexity** - Transformers must use `at_end()` for upstream outputs with fallback to `at_start()`, adding subtle correctness requirements
- **Input remapping** - Components declare `"Temperature"` but must read from `"Temperature|_to_scalar"`
- **Graph complexity** - Extra nodes for each transformation
- **Intermediate variables** - Internal naming scheme pollutes the namespace

Transform-on-read eliminates all these issues by pushing transformation to access time.

### Per-Variable Weight Configuration

Rejected because:

- Weights are a property of the physical grid, not individual variables
- Would require extensive schema API changes
- Less intuitive than model-level configuration

### Implicit Broadcast

Rejected because:

- Broadcasting (Scalar→FourBox) invents spatial structure
- Only valid for well-mixed quantities (CO2 concentration, not temperature)
- Can lead to subtle physical errors
- Can be added later with explicit opt-in if needed

### Component-Level Fallbacks

Rejected because:

- Adds complexity to component declarations
- Components shouldn't need to know about available resolutions
- Schema-level is cleaner separation of concerns

## Trade-offs Accepted

1. **Hidden transformations** - Transformations don't appear in the component graph. Mitigated by:
   - `model.required_transformations()` for introspection
   - Debug logging when transformations occur
   - Component declarations still show the requested grid type

2. **Potential repeated computation** - Multiple calls to `at_start()` on the same aggregated window may recompute. Mitigated by:
   - Typical usage is one call per timestep
   - Can add per-timestep caching if profiling shows it's needed

3. **Aggregation loses information** - Auto-aggregation discards spatial detail. This is acceptable because:
   - Components explicitly request coarser resolution
   - Information loss is visible in component declarations
   - Native data is preserved in storage
