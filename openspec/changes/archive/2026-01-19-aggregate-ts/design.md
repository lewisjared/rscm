# Design: Variable Schema with Timeseries Aggregation

## Context

RSCM models consist of components that produce and consume timeseries variables. Currently:

- Variables are implicitly created when components declare outputs
- No mechanism exists to aggregate multiple outputs into a derived value
- Variable definitions are scattered across component definitions

Users need to:

- Sum radiative forcings from multiple components into a total
- Run models with varying component sets (CO2-only vs full chemistry)
- Configure models declaratively without code changes
- Visualise data flows including aggregations

## Decision

Introduce an explicit **Variable Schema** that declares all model variables upfront, including aggregates.

### API Design

```rust
// Schema declaration
let schema = VariableSchema::new()
    // Regular variables (optional - can be inferred from components)
    .variable("Atmospheric Concentration|CO2", "ppm")
    .variable("Effective Radiative Forcing|CO2", "W/m^2")
    .variable("Effective Radiative Forcing|CH4", "W/m^2")
    // Aggregate variable (parent in hierarchy)
    .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build();

// Model construction
ModelBuilder::new()
    .with_schema(schema)  // Optional - enables aggregation
    .with_component(carbon_cycle)
    .with_component(co2_erf)
    .with_component(ch4_erf)
    .build()
```

### Type Definitions

```rust
/// Operation for computing aggregate values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateOp {
    /// Sum all contributor values
    Sum,
    /// Arithmetic mean of contributor values
    Mean,
    /// Weighted sum (weights provided per contributor)
    Weighted(Vec<f64>),
}

/// Definition of a single variable in the schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDefinition {
    pub name: String,
    pub unit: String,
    pub grid_type: GridType,
}

/// Definition of an aggregate variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateDefinition {
    pub name: String,
    pub unit: String,
    pub grid_type: GridType,
    pub operation: AggregateOp,
    pub contributors: Vec<String>,
}

/// Complete variable schema for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableSchema {
    pub variables: HashMap<String, VariableDefinition>,
    pub aggregates: HashMap<String, AggregateDefinition>,
}
```

### Validation Rules

ModelBuilder performs these validations when a schema is provided:

| Rule                                                                   | Error                  |
| ---------------------------------------------------------------------- | ---------------------- |
| Component output not in schema variables or aggregates                 | `UndefinedVariable`    |
| Component input not in schema or component outputs                     | `UndefinedVariable`    |
| Unit mismatch between schema and component                             | `UnitMismatch`         |
| Aggregate contributor not defined in schema (as variable or aggregate) | `UndefinedContributor` |
| Aggregate contributor unit mismatch                                    | `UnitMismatch`         |
| Aggregate contributor grid type mismatch                               | `GridTypeMismatch`     |
| Circular aggregate dependency                                          | `CircularDependency`   |
| Weighted aggregate weights count != contributors count                 | `WeightCountMismatch`  |

Note: Aggregates can reference other aggregates as contributors. The dependency graph ensures aggregates are computed in the correct order.

### Execution Flow

1. **Build time:**
   - Validate components against schema
   - Create dependency edges: aggregate → contributors
   - Insert virtual aggregator nodes into component graph

2. **Runtime (per timestep):**
   - Solve components in dependency order (unchanged)
   - After all contributors resolve, compute aggregate values
   - Write aggregate values to timeseries collection

3. **Aggregation computation:**

   ```rust
   /// Compute aggregate from non-NaN contributor values.
   /// `contributors` contains only valid (non-NaN) values after filtering.
   fn compute_aggregate(contributors: &[f64], op: &AggregateOp) -> f64 {
       if contributors.is_empty() {
           return f64::NAN;
       }
       match op {
           AggregateOp::Sum => contributors.iter().sum(),
           // Mean divides by count of valid values, not total contributor count
           AggregateOp::Mean => contributors.iter().sum::<f64>() / contributors.len() as f64,
           AggregateOp::Weighted(weights) => {
               // Weights correspond to valid contributors only (after NaN filtering)
               contributors.iter().zip(weights).map(|(v, w)| v * w).sum()
           }
       }
   }
   ```

### Handling Missing Contributors

If an aggregate has no contributors that produced values (all NaN), the aggregate is NaN.
If some contributors are NaN, they are excluded from the computation (treated as missing data).

For `Weighted` aggregates with NaN contributors:

- Both the value and its corresponding weight are excluded
- The aggregate is computed from remaining value-weight pairs
- If all contributors are NaN, the result is NaN

### Schema Without Components

If a schema variable has no component writing to it, the timeseries remains NaN.
This enables:

- RF-driven runs that define concentration variables but don't compute them
- Partial model configurations
- Future extensibility

### Virtual Aggregator Nodes

For graph visualisation, each aggregate creates a virtual node:

```
[CO2ERF] ──→ [ERF|CO2] ──┐
                          ├──→ [Aggregator: ERF] ──→ [ERF]
[CH4ERF] ──→ [ERF|CH4] ──┘
```

Where `ERF` = `Effective Radiative Forcing` (the parent aggregates its children).

This makes aggregation visible in `model.to_dot()` output.

## Alternatives Rejected

1. **Component-level `contributes_to` attribute**: Requires macro changes, creates ownership confusion
2. **Implicit naming convention aggregation**: Too magical, inflexible
3. **Aggregator components**: Requires manual maintenance, doesn't solve runtime flexibility

See `investigation.md` for full comparison.

## Trade-offs Accepted

1. **More ceremony for simple cases**: Users must declare schema for aggregates (but schema is optional for non-aggregate use)
2. **Two sources of variable info**: Schema and component definitions both specify units/types (validation ensures consistency)
3. **Lazy aggregation only**: Cannot consume partial aggregates mid-resolution (acceptable for typical use cases)

## Future Work

- **Grid type conversion**: Auto-convert FourBox → Scalar when aggregating (separate proposal)
- **Pattern-based contributors**: `from_pattern("Effective Radiative Forcing|*")`
