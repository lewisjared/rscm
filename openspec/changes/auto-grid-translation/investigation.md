# Investigation: Automatic Grid Translation

## Context

Currently, when components declare variables with different grid types (Scalar, FourBox, Hemispheric), the model build fails with `GridTypeMismatch` if a producer outputs a different grid than a consumer expects. This forces users to either:

1. Create separate variables for each grid resolution (e.g., "Temperature" and "Temperature|FourBox")
2. Manually insert transformer components between mismatched producers/consumers
3. Ensure all connected components use the same grid type

The user wants components to declare the grid they need, and have the model automatically handle translation between grids where valid transformations exist.

## Codebase Analysis

### Relevant Existing Code

**Grid Transformation Logic:**

- `crates/rscm-core/src/spatial/four_box.rs:160-198` - `FourBoxGrid::transform_to()` implements:
  - FourBox→Scalar: weighted aggregation (works)
  - FourBox→Hemispheric: per-hemisphere aggregation (works)
  - FourBox→FourBox: identity (works)
- `crates/rscm-core/src/spatial/scalar.rs:60-78` - `ScalarGrid::transform_to()` implements:
  - Scalar→Scalar: identity (works)
  - Scalar→any: broadcasts value to all regions (works, but semantically questionable)
- `crates/rscm-core/src/spatial/hemispheric.rs:116-148` - `HemisphericGrid::transform_to()`:
  - Hemispheric→Scalar: weighted aggregation (works)
  - Hemispheric→Hemispheric: identity (works)
  - Hemispheric→FourBox: **returns error** (cannot disaggregate without assumptions)

**Current Grid Matching (where errors occur):**

- `crates/rscm-core/src/model.rs:95-134` - `verify_definition()` compares grid types and returns `GridTypeMismatch` error on line 117
- `crates/rscm-core/src/model.rs:306-316` - Schema validation also checks grid types

**Component I/O Declaration:**

- `crates/rscm-macros/src/lib.rs` - ComponentIO macro generates `RequirementDefinition::with_grid()` calls
- Components declare grid via `grid = "FourBox"` in `#[inputs(...)]` / `#[outputs(...)]`

**Variable Storage:**

- `crates/rscm-core/src/timeseries_collection.rs:71-85` - `TimeseriesData` enum has separate variants for each grid type
- Variables are stored with a specific grid type at creation time

**State Access (InputState):**

- `crates/rscm-core/src/state.rs:715-775` - Separate methods: `get_scalar_window()`, `get_four_box_window()`, `get_hemispheric_window()`
- The macro generates code that calls the appropriate method based on declared grid type

### Transformation Matrix (from spatial/mod.rs docs)

| From \ To       | Scalar    | Hemispheric | FourBox   |
|-----------------|-----------|-------------|-----------|
| **Scalar**      | Identity  | Broadcast*  | Broadcast*|
| **Hemispheric** | Aggregate | Identity    | ERROR     |
| **FourBox**     | Aggregate | Aggregate   | Identity  |

(*Broadcast should only be used for well-mixed species)

### Patterns to Follow

1. **Aggregator Pattern** (`crates/rscm-core/src/schema.rs:814-956`): Virtual `AggregatorComponent` is inserted into the graph to compute schema aggregates. This pattern could be extended for grid transformations.

2. **Edge-based Dependencies**: The component graph uses edges with `RequirementDefinition` to track data flow. Transformers could be inserted as additional nodes/edges.

3. **Build-time Validation**: Currently all validation happens in `ModelBuilder::build()`. Transformation insertion would happen here too.

### Potential Conflicts or Concerns

1. **Broadcast Semantics**: Scalar→FourBox/Hemispheric is semantically "broadcasting" a global value to regions. This is only valid for well-mixed quantities (e.g., CO2 concentration). For other variables (e.g., temperature), broadcasting doesn't make physical sense.

2. **Information Loss**: Aggregation (FourBox→Scalar) loses spatial information. This is fine for derived outputs but problematic if a downstream component expects to modify and feed back regional data.

3. **Performance**: Each transformation adds a virtual component to the execution graph. For models with many grid mismatches, this could add overhead.

4. **Debugging Complexity**: Implicit transformations might make it harder to trace data flow. Users might not realize aggregation is happening.

5. **Hemispheric→FourBox**: Currently impossible - cannot disaggregate without assumptions. Auto-translation would need to either error or provide explicit disaggregation policies.

## Approaches Considered

### Approach A: Implicit Virtual Transformer Components

**Description:** ModelBuilder automatically inserts virtual transformer components when grid mismatches are detected, similar to how AggregatorComponent works for schema aggregates.

**Implementation:**

- During `build()`, when `verify_definition()` detects grid mismatch
- Instead of error, check if transformation is valid via `transform_to()`
- If valid, insert a `GridTransformerComponent` that calls `transform_to()` on the data
- Create intermediate variable with unique name (e.g., `"Original Var|_transformed_scalar"`)
- Wire consumer to read from transformed variable

**Pros:**

- Transparent to users - existing code continues to work
- Leverages existing `transform_to()` implementations
- Follows established AggregatorComponent pattern
- Single variable name used by components (no naming duplication)

**Cons:**

- Implicit magic could hide aggregation happening
- No control over transformation behaviour (e.g., cannot customize weights)
- Broadcast (Scalar→FourBox) is auto-applied even when semantically wrong
- Harder to debug data flow

**Estimated scope:** medium

### Approach B: Explicit Transformation Declaration in Schema

**Description:** Require users to declare grid transformations in the VariableSchema. The schema defines the "canonical" grid for a variable, and components can read/write at different resolutions with explicit transformation rules.

**Implementation:**

- Add `transformation(from_var, to_grid, policy)` to VariableSchema builder
- Policies: `Aggregate(weights)`, `Broadcast`, `Error`
- ModelBuilder inserts transformers only for declared transformations
- Undeclared mismatches still error

**Pros:**

- Explicit - users know when transformation happens
- Customizable - can specify weights, block broadcasts
- Schema documents the transformation strategy
- Maintains single variable names

**Cons:**

- More boilerplate for users
- Requires schema to use auto-translation
- Need to update schema for each new component pairing

**Estimated scope:** medium-large

### Approach C: Component Declares Preferred Grid + Fallbacks

**Description:** Components declare their preferred grid and acceptable fallbacks. Model matches preferences or applies fallback transformations.

**Implementation:**

- Extend `RequirementDefinition` with `acceptable_grids: Vec<GridType>`
- Component declares: `#[inputs(temp { name = "T", grid = "FourBox", fallback = ["Scalar"] })]`
- If FourBox available, use it; otherwise accept Scalar (broadcast)
- Model matches best available grid

**Pros:**

- Components are explicit about what they can handle
- No schema boilerplate
- Works without schema at all

**Cons:**

- Macro complexity increases
- Components need to handle multiple grid types internally (or trust broadcast)
- Still implicit transformation could surprise users
- Broadcast semantics problem remains

**Estimated scope:** medium

### Approach D: Aggregation-Only Auto-Translation

**Description:** Only allow automatic aggregation (coarsening) - never broadcast. Scalar inputs can read any grid (auto-aggregate), but grid inputs require exact match or explicit broadcast declaration.

**Implementation:**

- Scalar inputs: allow connection to FourBox/Hemispheric producers (auto-aggregate)
- FourBox/Hemispheric inputs: require exact match
- Add explicit `#[inputs(temp { name = "T", grid = "Scalar", broadcast_from = "FourBox" })]` for broadcast

**Pros:**

- Safe default - aggregation is always valid
- Broadcast requires explicit opt-in (prevents semantic errors)
- Minimal change for aggregation (the common case)
- Simple mental model: "coarsening is free, disaggregation is explicit"

**Cons:**

- Asymmetric - aggregation is automatic but broadcast is not
- Still need explicit declaration for broadcast scenarios

**Estimated scope:** small-medium

## Open Questions

- [ ] Should broadcast (Scalar→FourBox) be allowed implicitly, or require explicit opt-in?
- [ ] What should happen for Hemispheric→FourBox (currently impossible) - error, or allow explicit policy?
- [ ] Should transformations be visible in the component graph (`.to_dot()`)?

## Schema Integration Insight

The VariableSchema already:
1. Declares variables with `name`, `unit`, and `grid_type`
2. Validates that components match schema grid types (`ComponentSchemaGridMismatch`)
3. Inserts virtual `AggregatorComponent` nodes for aggregates

This provides a natural place to define transformation behaviour.

### Refined Approach: Schema-Driven Grid Translation

**Core Idea:** The schema declares the "native" (finest) resolution for each variable. Components can read at any coarser resolution via automatic aggregation.

```rust
let schema = VariableSchema::new()
    // Native resolution is FourBox - components can read as Scalar or Hemispheric
    .variable_with_grid("Surface Temperature", "K", GridType::FourBox)
    // Native resolution is Scalar - no transformation needed
    .variable("Atmospheric Concentration|CO2", "ppm");
```

**Rules:**

1. **Writers must match native resolution** - A component writing to "Surface Temperature" must output FourBox
2. **Readers can request coarser resolution** - A component declaring scalar input for "Surface Temperature" gets auto-aggregated values
3. **No implicit broadcast** - A component cannot read FourBox from a Scalar variable without explicit declaration

**Implementation:**

1. **Modify `validate_component_against_schema()`** - Allow input grid to be coarser than schema grid
2. **Insert `GridTransformerComponent`** - Virtual component (like AggregatorComponent) that aggregates for consumers
3. **Store at native resolution** - TimeseriesCollection stores at schema's grid type
4. **Transform on read** - InputState provides aggregated view to components that need it

**Example Flow:**

```
Schema: "Temperature" → FourBox

Component A (writer):
  outputs: Temperature [FourBox] ✓ matches schema

Component B (reader):
  inputs: Temperature [Scalar]  ✓ coarser than schema → auto-aggregate

Model Graph:
  [Component A] → "Temperature" (FourBox) → [GridTransformer] → "Temperature|_scalar" → [Component B]
```

### Explicit Broadcast Declaration

For the rare case where a scalar needs to be broadcast to regions:

```rust
let schema = VariableSchema::new()
    .variable("Global ERF", "W/m^2")  // Scalar
    .broadcast("Global ERF", GridType::FourBox);  // Allow components to read as FourBox
```

This makes broadcast explicit in the schema, documenting the intent.

## Recommendation

**Schema-Driven Grid Translation with Transform-on-Read:**

1. **Schema is source of truth** - Defines native resolution for each variable
2. **Aggregation is automatic** - Components can read at coarser resolution
3. **Broadcast is explicit** - Requires `.broadcast()` declaration in schema
4. **Transform at access time** - InputState aggregates when component reads, Model aggregates when writing

**Why not virtual transformer components?**

Initially considered inserting `GridTransformerComponent` nodes (like `AggregatorComponent`), but this approach has issues:
- Timestep complexity - transformers must handle `at_end()` vs `at_start()` correctly
- Input remapping - components declare "Temperature" but read from "Temperature|_to_scalar"
- Graph complexity - extra nodes for each transformation

Transform-on-read is simpler: the caller specifies which timestep via `at_start()`/`at_end()`, and the transformation naturally follows.

**Benefits:**
- Single variable names (no "Temperature" vs "Temperature|FourBox")
- Schema documents resolution decisions
- Safe default (aggregation only)
- No timestep complexity at transformation layer
- Simpler component graph

**Changes Required:**
1. Relax `ComponentSchemaGridMismatch` for inputs when component grid is coarser
2. Record required transformations during model build
3. Modify InputState to aggregate on read when transformation is needed
4. Modify Model.step() to aggregate on write when transformation is needed
5. Add `.broadcast()` method to VariableSchema builder (optional, for explicit broadcast)

This approach is consistent with the project's design principle of **explicitness** - transformations are recorded and can be queried, and unsafe operations (broadcast) require explicit declaration.
