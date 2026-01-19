# Change: Clarify Timestep Access Semantics

## Why

The current state spec's "Exogenous vs Endogenous Variable Handling" requirement is misleading. The real distinction is about **which timestep index to read from** based on execution order, not about variable classification. Components need explicit control over whether to read from the start of timestep (index N) or end of timestep (index N+1, where upstream components just wrote).

## What Changes

- **REMOVED** "Exogenous vs Endogenous Variable Handling" requirement from state spec
- Add `at_start()` method to `TimeseriesWindow` and `GridTimeseriesWindow` - returns value at start of timestep (index N)
- Add `at_end()` method to `TimeseriesWindow` and `GridTimeseriesWindow` - returns value at end of timestep (index N+1)
- **Deprecate** `current()` method (keep as alias to `at_start()` during transition)
- Remove recently added `get_scalar_value()`, `get_four_box_values()`, `get_hemispheric_values()` methods from `InputState`
- Update `AggregatorComponent` to use `at_end()` instead of `at_offset(1)`
- Add "Timestep Access Semantics" requirement documenting when to use each method

## Chosen Approach

Add explicit timestep-semantic methods (`at_start()`, `at_end()`) rather than automatic resolution based on variable type. See investigation.md for alternatives considered.

**Key insight:** The determining factor is execution order, not variable classification:

| What you're reading       | Who wrote it        | Use          |
| ------------------------- | ------------------- | ------------ |
| Your own state variable   | You haven't run yet | `at_start()` |
| Upstream component output | They already ran    | `at_end()`   |
| Exogenous input           | Pre-populated       | `at_start()` |

Component authors must explicitly choose - there is no universal default.

## Impact

- Affected specs: `state`
- Affected code:
  - `crates/rscm-core/src/state.rs` - add new methods, deprecate `current()`
  - `crates/rscm-core/src/schema.rs` - update `AggregatorComponent` to use `at_end()`
  - Existing components using `current()` - update to use `at_start()` or `at_end()` as appropriate
- **Not breaking:** `current()` remains as deprecated alias
