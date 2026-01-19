# Tasks: Clarify Timestep Access Semantics

## 1. Add Timestep-Semantic Methods to Window Types

- [x] 1.1 Add `at_start()` method to `TimeseriesWindow` (returns value at index N)
- [x] 1.2 Add `at_end()` method to `TimeseriesWindow` (returns `Option<FloatValue>` at index N+1)
- [x] 1.3 Add `at_start()` method to `GridTimeseriesWindow` (single region access)
- [x] 1.4 Add `at_end()` method to `GridTimeseriesWindow` (single region access, returns Option)
- [x] 1.5 Add `current_all_at_start()` method to `GridTimeseriesWindow` (all regions at N)
- [x] 1.6 Add `current_all_at_end()` method to `GridTimeseriesWindow` (all regions at N+1, returns Option)
- [x] 1.7 Add tests for new methods

**File:** `crates/rscm-core/src/state.rs`

**Verification:** `cargo test --package rscm-core state::`

## 2. Deprecate Current Methods

- [x] 2.1 Add `#[deprecated]` attribute to `TimeseriesWindow::current()` with message pointing to `at_start()`
- [x] 2.2 Add `#[deprecated]` attribute to `GridTimeseriesWindow::current()` with message pointing to `at_start()`
- [x] 2.3 Add `#[deprecated]` attribute to `GridTimeseriesWindow::current_all()` with message pointing to `current_all_at_start()`

**File:** `crates/rscm-core/src/state.rs`

**Verification:** Compile with `-W deprecated` to see warnings

## 3. Remove Wrong-Abstraction Methods

- [x] 3.1 Remove `get_scalar_value()` from `InputState`
- [x] 3.2 Remove `get_four_box_values()` from `InputState`
- [x] 3.3 Remove `get_hemispheric_values()` from `InputState`
- [x] 3.4 Remove associated tests for these methods

**File:** `crates/rscm-core/src/state.rs`

**Verification:** `cargo test --package rscm-core`

## 4. Update AggregatorComponent

- [x] 4.1 Replace `at_offset(1).unwrap_or_else(|| window.current())` with `at_end().unwrap_or_else(|| window.at_start())`
- [x] 4.2 Update comment to reference new method names
- [x] 4.3 Update for grid windows similarly

**File:** `crates/rscm-core/src/schema.rs`

**Verification:** `cargo test --package rscm-core schema::`

## 5. Update Existing Components

- [x] 5.1 Update `CO2ERF` to use appropriate timestep accessor
- [x] 5.2 Update `CarbonCycle` to use `at_start()` for state variable initial conditions
- [x] 5.3 Update `FourBoxOceanHeatUptake` to use appropriate timestep accessor
- [x] 5.4 Update `TwoLayer` to use `at_start()` for state variables
- [x] 5.5 Update `OceanSurfacePartialPressure` to use `at_start()` for inputs

**Files:** `crates/rscm-components/src/components/*.rs`, `crates/rscm-two-layer/src/component.rs`

**Verification:** `cargo test --workspace`

## 6. Documentation

- [x] 6.1 Add rustdoc to `at_start()` explaining when to use it (state variables, exogenous)
- [x] 6.2 Add rustdoc to `at_end()` explaining when to use it (upstream outputs, aggregation)
- [x] 6.3 Update CLAUDE.md component development section with guidance on which method to use

**Verification:** `cargo doc --package rscm-core --open`

## 7. Final Validation

- [x] 7.1 Run full test suite: `cargo test --workspace`
- [x] 7.2 Run clippy: `cargo clippy --workspace --tests`
- [x] 7.3 Verify no compile warnings for deprecated usage in core crates
