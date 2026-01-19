# Investigation: Clarify Timestep Access Semantics

## Context

The user has identified confusion around how components access timeseries data, specifically around the "endogenous vs exogenous" distinction in the state spec. After investigation, the real issue is about **which timestep index** to read from, not about interpolation.

In a model chain like:

```
Emissions → Carbon Cycle → CO2 Concentration → CO2 ERF → Total ERF → Temperature
```

At each timestep N:

1. Carbon Cycle reads concentration at N (state initial condition), integrates, writes to N+1
2. CO2 ERF should read the **just-computed** concentration at N+1 to compute ERF
3. Total ERF aggregator needs values computed **this timestep** (at N+1)

The physics requires reading upstream outputs computed this timestep, not old values.

## Codebase Analysis

### Relevant Existing Code

**Model writes outputs to time_index + 1:**

- `crates/rscm-core/src/model.rs:975-979`

  ```rust
  // The next time index is used as this output state represents the value of a
  // variable at the end of the current time step.
  // This is the same as the start of the next timestep.
  StateValue::Scalar(v) => data.set_scalar(key, self.time_index + 1, *v),
  ```

**TimeseriesWindow.current() reads from current_index (N):**

- `crates/rscm-core/src/state.rs:59-63`

  ```rust
  pub fn current(&self) -> FloatValue {
      self.timeseries
          .at(self.current_index, ScalarRegion::Global)
          .expect("Current index out of bounds")
  }
  ```

**AggregatorComponent uses at_offset(1) to read just-computed values:**

- `crates/rscm-core/src/schema.rs:883-896`

  ```rust
  // Aggregators need to read values that were just computed in this timestep.
  // Component outputs are written to time_index + 1, so we read at offset +1
  window.at_offset(1).unwrap_or_else(|| window.current())
  ```

**Exogenous data is pre-interpolated onto model time axis:**

- `crates/rscm-core/src/model.rs:763-768`

  ```rust
  timeseries.to_owned().interpolate_into(self.time_axis.clone())
  ```

**The endogenous/exogenous distinction in get_global():**

- `crates/rscm-core/src/state.rs:673-690`

  ```rust
  VariableType::Exogenous => ts.at_time(self.current_time, ...).ok(),
  VariableType::Endogenous => ts.latest_value(),
  ```

### Patterns to Follow

The model follows a **forward-Euler time-stepping scheme**:

- Values at index N represent state at START of timestep N
- Components read from N, compute, write to N+1
- Index N+1 = end of current step = start of next step

The AggregatorComponent's `at_offset(1)` pattern is the correct way to read upstream outputs computed in the same timestep.

### Potential Conflicts or Concerns

1. **Spec vs Implementation mismatch**: The state spec (lines 260-274) describes endogenous/exogenous handling, but:
   - Exogenous data is already pre-interpolated at build time
   - The window API (`current()`) doesn't implement this distinction
   - Only `get_global()` implements it, but components use windows

2. **Naming confusion**: `current()` doesn't mean "most recently computed" - it means "at the current time index". This is correct for state initial conditions but wrong for reading upstream outputs.

3. **Macro-generated code uses windows**: The ComponentIO macro generates code using `get_scalar_window(name).current()`, which reads from index N, not N+1.

4. **Recently added accessor methods**: The `get_scalar_value()`, `get_four_box_values()`, `get_hemispheric_values()` methods implement the endogenous/exogenous distinction, but this may be the wrong abstraction.

## Approaches Considered

### Approach A: Clarify Window API with Timestep-Semantic Names

**Description:** Add new methods to TimeseriesWindow that make timestep semantics explicit: `at_start()` for index N, `at_end()` for index N+1.

**Changes:**

- Add `at_start(&self) -> FloatValue` (alias for current behavior)
- Add `at_end(&self) -> FloatValue` (equivalent to `at_offset(1)`)
- Deprecate `current()` or keep as alias to `at_start()`
- Update spec to document timestep semantics clearly
- Remove confusing endogenous/exogenous distinction from spec

**Pros:**

- Semantics match the physics (start vs end of timestep)
- Makes the aggregator pattern explicit and named
- Self-documenting API
- Backwards compatible (existing `current()` still works)

**Cons:**

- Yet more methods on the window API
- Need to update all existing components to use new names
- `at_end()` may return None at final timestep

**Estimated scope:** small

### Approach B: Keep Current API, Only Document Better

**Description:** Keep `current()` and `at_offset(1)` as-is, but improve documentation and update the spec to remove the misleading endogenous/exogenous distinction.

**Changes:**

- Update spec to remove "Exogenous vs Endogenous Variable Handling" requirement
- Document that exogenous is pre-interpolated at build time
- Document that `at_offset(1)` is the pattern for reading upstream outputs
- Remove the recently added accessor methods (`get_scalar_value`, etc.)

**Pros:**

- Minimal code changes
- No new API surface
- Existing patterns (aggregator's `at_offset(1)`) already work

**Cons:**

- `at_offset(1)` is less self-documenting than `at_end()`
- The pattern is non-obvious and requires reading aggregator code to discover
- Doesn't address the naming confusion

**Estimated scope:** small

### Approach C: Automatic Resolution Based on Variable Type

**Description:** Make `current()` automatically return the right value based on whether the variable is exogenous (start of timestep) or endogenous (latest computed).

**Changes:**

- Pass VariableType to TimeseriesWindow
- Modify `current()` to check type and return appropriate value
- Endogenous returns `latest_value()`, exogenous returns `at(current_index)`

**Pros:**

- Components don't need to think about which method to call
- Matches the original spec intention

**Cons:**

- Hides complexity, harder to reason about
- Breaks mental model of "window at index N"
- Makes `previous()`, `at_offset()` semantics unclear for endogenous
- The endogenous/exogenous distinction isn't the real issue (it's which timestep)

**Estimated scope:** medium

## Open Questions

1. ~~Should we deprecate `current()` in favor of `at_start()`/`at_end()`, or keep all three?~~ **Resolved: Deprecate `current()`**
2. ~~Should the macro-generated code be updated to use the new methods, or should components explicitly choose?~~ **Resolved: Components explicitly choose - no default**
3. ~~Do we want different behavior for state variables (read start) vs diagnostic chains (read end)?~~ **Resolved: See below**

### Resolution: Execution Order Determines Access Pattern

The determining factor is **whether the value at N+1 exists when you read it**:

| What you're reading | Who wrote it | Use |
|---------------------|--------------|-----|
| Your own state variable | You haven't run yet | `at_start()` |
| Upstream component output | They already ran | `at_end()` |
| Exogenous input | Pre-populated | `at_start()` |

**There is no universal default.** The component author must explicitly choose based on their component's role in the dependency graph:

1. **Own state variable** (appears in both inputs and outputs): `at_start()` - the N+1 slot is empty, you're about to write to it
2. **Upstream output** (dependency): `at_end()` - the upstream already wrote to N+1
3. **Exogenous forcing**: `at_start()` or `interpolate(t)` for ODE integration
4. **Aggregating just-computed values**: `at_end()`

## Recommendation

**Approach A** with these decisions:

1. Add `at_start()` and `at_end()` methods to both `TimeseriesWindow` and `GridTimeseriesWindow`
2. Deprecate `current()` (keep as alias to `at_start()` during transition)
3. Update the state spec to:
   - Remove "Exogenous vs Endogenous Variable Handling" requirement (it's misleading)
   - Add "Timestep Access Semantics" requirement documenting when to use each method
4. Remove the recently added `get_scalar_value()`, `get_four_box_values()`, `get_hemispheric_values()` methods (they implement the wrong abstraction)
5. Update aggregator to use `at_end()` instead of `at_offset(1)` for clarity
6. Document guidance for component authors on which method to use

This approach:

- Makes the physics-based distinction explicit in the API
- No magic defaults - explicit choice required
- Removes the confusing endogenous/exogenous framing
- Documents the actual time-stepping semantics
