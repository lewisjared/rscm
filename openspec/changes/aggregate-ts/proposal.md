# Change: Variable Schema with Timeseries Aggregation

## Why

Models need to aggregate multiple timeseries into totals (e.g., sum individual radiative forcings into total forcing). Currently, variables are implicitly created from component definitions with no way to express aggregation relationships. This makes it difficult to:

- Compose flexible models where the set of contributors varies at runtime
- Trace data flows when multiple components contribute to a derived value
- Configure models declaratively (e.g., JSON/TOML)

## What Changes

- **NEW:** `VariableSchema` type for declaring model variables and aggregates
- **NEW:** `AggregateOp` enum with `Sum`, `Mean`, `Weighted` operations
- **MODIFIED:** `ModelBuilder` validates components against schema
- **MODIFIED:** Model execution computes aggregates after contributors resolve
- **NEW:** Virtual aggregator nodes appear in component graph for visibility
- **NEW:** Python bindings for schema definition

## Chosen Approach

**Explicit Variable Schema** - Variables (including aggregates) are declared separately from components at the model level. Components declare which variables they read/write, and the ModelBuilder validates consistency. See `investigation.md` for alternatives considered.

Key design decisions:

- Variables are first-class entities, not side effects of component definitions
- Aggregates are variables with an operation and list of contributors
- Unreferenced schema variables are NaN (enables flexible configurations)
- Lazy aggregation: computed after all contributors resolve

## Impact

- Affected specs: `state` (new schema types), `component` (validation changes)
- Affected code:
  - `crates/rscm-core/src/schema.rs` (new)
  - `crates/rscm-core/src/model.rs` (validation, aggregation execution)
  - `crates/rscm-core/src/lib.rs` (exports)
  - `crates/rscm/src/python/` (bindings)
