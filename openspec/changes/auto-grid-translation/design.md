# Design: Schema-Driven Grid Auto-Aggregation

## Context

Climate models often work at different spatial resolutions. Some components produce regional outputs (FourBox), while others only need global averages (Scalar). Currently, connecting these requires either:

1. Duplicate variable names per resolution
2. Manual transformer components
3. Modifying components to match grids

The VariableSchema already declares grid types for variables and inserts virtual AggregatorComponent nodes for aggregates. This infrastructure can be extended for grid transformations.

**Constraints:**

- Must not break existing models (backwards compatible)
- Aggregation weights must be physically meaningful (area-based)
- Performance overhead should be minimal
- Transformations should be visible in the component graph

## Decision

### Schema as Source of Truth

The VariableSchema declares the "native" grid type for each variable. This is the resolution at which the variable is stored and at which writers must produce values.

```rust
let schema = VariableSchema::new()
    .variable_with_grid("Surface Temperature", "K", GridType::FourBox)
    .variable("Atmospheric Concentration|CO2", "ppm");  // Scalar default
```

### Auto-Aggregation (Read and Write)

The system provides symmetric aggregation for both inputs and outputs:

**Read-side (inputs):** When a component declares an input at a coarser resolution than the schema, the model aggregates data before the component reads it.

**Write-side (outputs):** When a component produces output at a finer resolution than the schema, the model aggregates data before storing it.

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

### Virtual Transformer Components

Similar to `AggregatorComponent`, `GridTransformerComponent` is a virtual component inserted during `ModelBuilder::build()`:

- **Input**: Variable at native resolution
- **Output**: Variable at requested resolution (with internal naming like `"VarName|_to_scalar"`)
- **Execution**: Reads native grid, applies `transform_to()`, writes coarser grid

These appear in the component graph (`.to_dot()`) for debugging.

### Transformation Caching

Multiple components may request the same transformation (e.g., three components all want scalar temperature from FourBox). Only one transformer is created per unique (variable, target_grid) pair.

## Alternatives Rejected

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

1. **Writers must match native resolution** - Components cannot output finer data than schema declares. This ensures schema is authoritative but may require component changes.

2. **Aggregation loses information** - Auto-aggregation discards spatial detail. This is acceptable because:
   - Components explicitly request coarser resolution
   - Information loss is visible in component declarations
   - Native data is preserved in storage

3. **Graph complexity increases** - Transformer nodes add to the component graph. Mitigated by:
   - Clear naming convention (`|_to_scalar`)
   - Visible in `.to_dot()` for debugging
   - Minimal runtime overhead (simple weighted average)
