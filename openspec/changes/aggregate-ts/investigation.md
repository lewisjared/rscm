# Investigation: Timeseries Aggregation System

## Context

The user needs a mechanism to aggregate multiple timeseries variables into totals. The primary use case is radiative forcing: multiple components produce forcing values (CO2, CH4, aerosols, etc.) that need to be summed into a total forcing value. Some downstream components (e.g., temperature response) only need the total, not the individual contributors.

Key constraints:

1. **Runtime flexibility**: At compile time, we don't know the full hierarchy of variables. A CO2-only model has `total == CO2`, while a full model sums many components.
2. **Developer visibility**: Data flows should be explicit and easy to trace.
3. **Python component support**: Custom Python components should be able to contribute to aggregates.

## Codebase Analysis

### Relevant Existing Code

- `crates/rscm-core/src/component.rs:85-103` - `RequirementDefinition` with name, unit, requirement_type, grid_type
- `crates/rscm-core/src/model.rs:250-429` - `ModelBuilder.build()` constructs dependency graph, creates timeseries collection
- `crates/rscm-core/src/model.rs:500-550` - `Model.solve_component()` writes outputs to collection at `time_index + 1`
- `crates/rscm-components/src/components/four_box_ocean_heat_uptake.rs:77` - Example consuming `"Effective Radiative Forcing|Aggregated"` (already expects aggregated input)
- `crates/rscm-core/src/spatial/mod.rs:334-346` - `SpatialGrid::aggregate_global()` for grid-to-scalar aggregation (different concept - spatial aggregation)

### Existing Patterns

1. **Variable naming**: Hierarchical with `|` delimiter: `"Effective Radiative Forcing|CO2"`, `"Emissions|CO2|Anthropogenic"`
2. **No existing timeseries aggregation**: Current "aggregation" is spatial only (FourBox -> Scalar)
3. **Component outputs are unique**: No two components can produce the same variable name (enforced at build time)
4. **Dependency graph**: BFS traversal ensures producers run before consumers

### Key Insight: Naming Convention Already Suggests Hierarchy

The existing component `FourBoxOceanHeatUptake` consumes `"Effective Radiative Forcing|Aggregated"`, suggesting the pattern was anticipated. Variables like:

- `"Effective Radiative Forcing|CO2"`
- `"Effective Radiative Forcing|CH4"`
- `"Effective Radiative Forcing|Aerosols"`

...could naturally sum to `"Effective Radiative Forcing"` or `"Effective Radiative Forcing"`.

### Potential Conflicts or Concerns

1. **Unit consistency**: All contributors must have matching units. The model already validates units match when same variable is referenced.
2. **Grid type consistency**: Contributors must have compatible grid types. Summing FourBox + Scalar requires decision (broadcast scalar? error?).
3. **Circular dependencies**: Aggregate must be computed after all contributors. Need to ensure dependency ordering.
4. **Performance**: Adding summation at each timestep has cost, though likely minimal compared to ODE solving.
5. **Python components**: Need to declare contributions from Python side via PyO3 bindings.

## Approaches Considered

### Approach A: Aggregator Component

**Description:** Create a special `Aggregator` component that explicitly lists its inputs and sums them.

```rust
let aggregator = Aggregator::new("Effective Radiative Forcing", "W/m^2")
    .with_contributor("Effective Radiative Forcing|CO2")
    .with_contributor("Effective Radiative Forcing|CH4");
```

**Pros:**

- Explicit: aggregation is a visible component in the graph
- Flexible: can add custom aggregation logic (weighted sums, etc.)
- No new concepts: uses existing Component trait
- Easy to understand data flow

**Cons:**

- Verbose: requires creating component for every aggregate
- Ordering issues: how do you know all contributors exist before building?
- Doesn't solve "runtime flexibility" - must list all contributors at build time

**Estimated scope:** Small

### Approach B: Declarative Aggregation via Naming Convention

**Description:** Use variable naming pattern where `"Category|Subcategory"` automatically contributes to `"Category"` or `"Category|Total"`. The ModelBuilder detects these patterns and creates virtual aggregation.

```rust
// Component declares output
#[outputs(
    erf_co2 { name = "Effective Radiative Forcing|CO2", unit = "W/m^2" },
)]

// Another component consumes the aggregate
#[inputs(
    total_erf { name = "Effective Radiative Forcing", unit = "W/m^2" },  // Auto-summed
)]
```

**Pros:**

- Zero boilerplate for common case
- Naturally extensible: add a new forcing component and it automatically contributes
- Naming convention is self-documenting

**Cons:**

- Implicit magic: may be surprising behaviour
- Inflexible: only supports hierarchical naming pattern
- Ambiguous: what if both `"ERF"` and `"ERF|Total"` exist?
- Hard to exclude a contributor

**Estimated scope:** Medium

### Approach C: Explicit Aggregation Registry

**Description:** Introduce an `AggregationRegistry` configured at model build time that declares which outputs contribute to which aggregates.

```rust
let registry = AggregationRegistry::new()
    .aggregate("Effective Radiative Forcing")
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build()
    .aggregate("Emissions|Total")
        .from_pattern("Emissions|*")  // Glob pattern
        .build();

ModelBuilder::new()
    .with_aggregations(registry)
    .with_component(co2_erf)
    .with_component(ch4_erf)
    .build()
```

**Pros:**

- Explicit and centralized: easy to see all aggregations
- Flexible: supports explicit lists and patterns
- Ordering is clear: aggregates computed after all contributors
- Can validate at build time that contributors exist

**Cons:**

- More ceremony than naming convention approach
- Patterns could be confusing (what matches `"Emissions|*"`?)
- Separate from component definitions

**Estimated scope:** Medium

### Approach D: Component-Level Declaration with Tags

**Description:** Components declare "I contribute to X" via a new attribute on outputs. The model collects all contributors and generates aggregates automatically.

```rust
#[derive(ComponentIO)]
#[outputs(
    erf_co2 {
        name = "Effective Radiative Forcing|CO2",
        unit = "W/m^2",
        contributes_to = "Effective Radiative Forcing"  // NEW
    },
)]
pub struct CO2ERF { ... }
```

**Pros:**

- Declaration at source: each component declares its contribution
- Automatically extensible: new components self-register
- Build-time validation: missing aggregate is an error
- No separate registry to maintain

**Cons:**

- Macro changes required
- Aggregate must be "declared" somewhere (or auto-created?)
- Cross-cutting concern embedded in component definitions
- Multiple components referencing same aggregate name must agree

**Estimated scope:** Medium-Large

## Open Questions (Resolved)

1. **Should aggregates be auto-created or explicitly declared?**
   → Explicitly declared in a variable schema, separate from components.

2. **How should grid type mismatches be handled?**
   → Out of scope. Grid auto-conversion is a follow-up change.

3. **Should aggregation support operations other than sum?**
   → Yes: `sum`, `mean`, and `weighted` operations.

4. **Should Python components be able to define aggregations?**
   → Yes. Both Rust and Python can define schemas and components.

5. **What happens if an aggregate has zero contributors?**
   → NaN (missing value), consistent with unreferenced variables.

6. **What about unreferenced schema variables?**
   → They remain NaN. This enables flexible configurations (e.g., RF-driven run that declares concentrations but doesn't use them).

7. **Who owns aggregate variables?**
   → No ownership. Variables are declared in schema, components just read/write to them.

## Refined Approach: Explicit Variable Schema

Based on discussion, the cleanest design separates variable declaration from components entirely.

### Core Concept

Variables (including aggregates) are first-class entities declared at the model level:

```rust
let schema = VariableSchema::new()
    // Regular variables
    .variable("Effective Radiative Forcing|CO2", "W/m^2")
    .variable("Effective Radiative Forcing|CH4", "W/m^2")
    // Aggregate variable
    .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build();

ModelBuilder::new()
    .with_schema(schema)
    .with_component(co2_erf)
    .with_component(ch4_erf)
    .build()
```

### Component Declaration (Unchanged)

Components continue to declare their inputs/outputs. The model validates against the schema:

```rust
#[derive(ComponentIO)]
#[inputs(
    concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
)]
#[outputs(
    erf { name = "Effective Radiative Forcing|CO2", unit = "W/m^2" },
)]
pub struct CO2ERF { ... }
```

### Validation Rules

| Check | Behaviour |
|-------|-----------|
| Component writes to undefined variable | Error at build time |
| Component reads undefined variable | Error at build time |
| Schema variable not written by any component | NaN values (allowed) |
| Unit mismatch between schema and component | Error at build time |
| Aggregate contributor not in schema | Error at build time |

### Aggregation Execution

1. **Build time:** ModelBuilder creates dependency graph where aggregate depends on all its contributors
2. **Runtime:** After all contributor components solve, aggregate is computed from their outputs
3. **Graph visibility:** Aggregate appears as a node in the component graph (virtual aggregator)

### Benefits

| Aspect | Benefit |
|--------|---------|
| Separation of concerns | Variables are data schema, components are logic |
| Runtime flexibility | Schema can be JSON/TOML configured |
| No ownership confusion | Variables are shared model state |
| Flexible configurations | Unreferenced variables are NaN, not errors |
| Clear validation | Schema is source of truth |

## Recommendation

Proceed with **Explicit Variable Schema** approach:

1. Create `VariableSchema` type with builder API
2. Support `variable()` for regular variables and `aggregate()` for aggregates
3. Aggregate operations: `Sum`, `Mean`, `Weighted`
4. ModelBuilder validates components against schema
5. Virtual aggregator nodes appear in component graph
6. Python bindings for schema definition
7. Schema serializable to JSON/TOML for configuration
