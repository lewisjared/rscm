# Change: Schema-Driven Grid Auto-Aggregation

## Why

Components currently fail to connect when they use different grid resolutions for the same variable, forcing users to create duplicate variable names (e.g., "Temperature" and "Temperature|FourBox"). This adds complexity and obscures the model structure.

## What Changes

- Components can read variables at coarser resolution than the schema declares (read-side aggregation)
- Components can write variables at finer resolution than the schema declares (write-side aggregation)
- Model automatically inserts `GridTransformerComponent` nodes for aggregation (similar to `AggregatorComponent`)
- Aggregation uses configurable area weights (defaulting to grid's standard weights)
- Schema validation relaxed for both inputs and outputs where aggregation is valid
- No broadcast/disaggregation support (finer-from-coarser is always an error)

## Chosen Approach

Schema-driven auto-aggregation where:

1. Schema declares the "interface" resolution for each variable (what gets stored)
2. Readers can request any coarser resolution - model auto-aggregates before read
3. Writers can produce any finer resolution - model auto-aggregates before storage
4. Disaggregation (broadcast) is never implicit - always an error

## Impact

- Affected specs: `variable-schema`, `component`, `grid-timeseries`
- Affected code:
  - `crates/rscm-core/src/schema.rs` - GridTransformerComponent, validation changes
  - `crates/rscm-core/src/model.rs` - Build-time transformer insertion, relaxed validation
  - `crates/rscm-core/src/spatial/` - Weight configuration access
