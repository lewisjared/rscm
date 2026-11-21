# Resolve coupling once, from labelled components

## Outcome

One function, `resolve`, makes every step coupling decision for a model.
It returns a `StepPlan` that the runtime steps, the coupling report projects and standalone `Component.solve` uses.
Components carry labels, and labels key every coupling decision and every lag.
One storage rule decides where each variable lives, with the schema as an override rather than a second code path.
A checkpoint and a run output are the same `Record`.
It holds the serialised builder, the step index, the timeseries values and the component states.
Restore re-runs `resolve`, so there is one construction path.

The builder still owns producer selection, execution order and lags under [ADR 0001](../adr/0001-builder-owns-step-coupling.md).
[ADR 0002](../adr/0002-point-and-average-variables.md) still governs point and average storage.

This is a breaking release.
It breaks the checkpoint format, low-level Rust constructors, the `Model` map accessors, lag strings that name Rust types,
and components that return undeclared outputs.

## Settled decisions

- **Labels.** `with_component(c)` labels a component with its type name.
  `with_component_as(label, c)` sets the label.
  A duplicate label is a build error that asks for an explicit one.
  An aggregate's label is its schema name.
  A Python component's default label is its class `__name__`.
- **Lags.** `with_lag(variable)` lags every reader.
  `with_lag_for(variable, label)` lags one labelled consumer.
  Python `Lag(variable, into=[labels])` maps onto these.
  Two instances of one type lag independently.
- **Storage rule.** A variable is stored as its schema entry says.
  With no entry it is stored as its producer declares it.
  A prescribed-only variable keeps the unit and grid of the data supplied.
  Every read converts stored to declared and every write converts declared to stored, unit first, then grid.
  Finer to coarser aggregates on either side.
  Coarser to finer is a build error.
- **One plan.** `resolve` runs inside `build` and produces the `StepPlan` eagerly.
  Nothing re-derives coupling afterwards.
- **Record.** `Record { format, rscm, builder, time_index, values, states }`, serialised as JSON only.
  `ModelBuilder` derives `Serialize` and `Deserialize` and is the spec.
  `model.checkpoint()` and `model.output()` both return a `Record`.
  `Model::restore(record)` re-resolves, then errors if the rebuilt storage (variable set, unit, grid, timeseries type) differs from the saved values.
- **Versions.** A `format` mismatch is an error naming both numbers.
  A different `rscm` version restores with a warning.
- **Branching.** `Model::branch(record, Branch { prescribed, parameters })` replaces future prescribed data and component parameters.
  The time axis cannot change.
  Saved history wins before the branch point.
  If new parameters change a component's declarations, the storage check fails.
- **Python.** `to_toml` and `from_toml` go.
  Pickle and `deepcopy` go through `Record` JSON.
  `ModelSpec` stays as a thin wrapper that forwards labels and lags to the Rust builder and does no wiring itself.
- **Outputs.** A component that returns an undeclared output fails the step with the component, variable and step named.
  All of one component's writes are validated before any is committed.

## Stacked PRs

Each PR targets the branch before it.
Each must leave `cargo test --workspace --exclude rscm`, `make build-dev`, `uv run pytest -m "not slow"` and `make parity` green.
Each gets an independent Codex review before the next one starts.

### 1. `fix/reject-undeclared-outputs`

- Reject undeclared outputs in `runtime.rs` (`take_undeclared` path, `step_model_component`).
  The error names the component, variable and step bounds.
- Validate and convert every output of a component into a staging buffer before writing any of them.
  An aggregation or `set_*` failure leaves the collection untouched for that component.
- Before flipping to an error, confirm no Rust or Python component in the workspace returns an undeclared output, using `make test` and `make parity`.
- Tests: undeclared output fails even when a stored timeseries has that name, and earlier outputs of that component are not written.
  A failing aggregation commits nothing.
- Changelog: `breaking`.
- Includes this plan.

### 2. `feat/labelled-components`

- Add a label to each registered component in `ModelBuilder` and use it wherever `Model::extract_component_name` keys coupling today:
  `variable_sources`, `unit_conversions`, lag matching, `validate_lags`, the coupling report and error messages.
- Add `with_component_as(label, component)` in Rust and Python, and default labels as above.
  Aggregates use their schema name, so `with_lag_for(x, "<aggregate>")` works.
- Python `ModelSpec` passes its dict keys as labels.
  `Lag(variable, into=[...])` replaces the single-consumer form.
  Migrate the MAGICC7 and FaIR runners, notebooks and docs.
- Tests: two instances of one type with different units and different lags couple independently.
  Duplicate labels fail.
  A lag on an aggregate consumer works.
- Changelog: `breaking` (lag strings now name labels) and `fix` (same-type instances no longer share coupling).

### 3. `refactor/storage-rule`

- Replace the schema and no-schema branches in `builder.rs` and `validation.rs` with the one storage rule.
  An absent schema becomes an inferred one, so `verify_definition`'s schema branch and the `unit_conversions.clear()` path go.
- One `units::factor(variable, from, to) -> RSCMResult<Option<f64>>` replaces `check_unit_compatibility`, `validate_unit_compatibility` and `write_factor`.
  Remove the duplicate check in `python/rscm/names.py` if the Rust one covers it.
- Keep the plain-read fast path (scalar, factor within `f64::EPSILON` of 1, prescribed, point) and the `factor != 1.0` write check exactly, for ulp parity.
- Tests: a prescribed-only variable read in two units stays in the supplied unit.
  A finer write aggregates to a coarser store with and without a schema.
  Coarser to finer is a build error.
- Changelog: `breaking` (storage units of prescribed-only variables).

### 4. `refactor/resolve-step-plan`

- Add `model/plan.rs` with `resolve(&ModelBuilder) -> RSCMResult<(StepPlan, TimeseriesCollection)>`
  and `StepPlan::report() -> Coupling`.
  `Step`, `Read`, `Write` and `Conversion` are keyed by label and carry slot indices.
- Build the plan inside `build` beside the existing maps.
  Do not change stepping yet.
- The report lists conversions per read or write with the component label, replacing the variable-level grid transform list.
- Test that the new report agrees with the old `coupling()` (allowing for the per-binding shape) on every model in the test suite and on the MAGICC7 and FaIR runners.
  Two consumers at different grids with non-uniform weights report two entries.

### 5. `refactor/runtime-step-plan`

- `Model::step` walks the `StepPlan` built in `build`, replacing the lazy `ComponentPlan` and the map lookups.
- `Component.solve` in Python and `extract_state` resolve a one-component builder with every input prescribed.
  Remove `TransformContext`, `build_with_transforms` and `extract_state_with_transforms`.
- `coupling()` returns `plan.report()`.
- Tests: values for two consumers at different grids with non-uniform weights.
  A lagged point sees the previous step for modelled and prescribed data.
  A state variable converts correctly in both directions.
  Reported order equals executed order.
- Compare a release-build `run_magicc7` timing before and after.

### 6. `feat/model-record`

- `ModelBuilder`, `InitialValue` and the schema derive `Serialize` and `Deserialize`.
- Add `Record`, `Model::checkpoint`, `Model::output`, `Model::restore` and `Model::branch`, with the format and version rules above.
- Python: `Model.checkpoint()`, `Model.output()`, `Model.restore()`, `Model.branch()`.
  `__reduce__` goes through `Record` JSON.
  Remove `to_toml` and `from_toml`.
- Update `docs/notebooks/state_serialisation.py` and `scenario_pipeline.py` to restore and branch.
- Tests: a mid-run checkpoint resumes to the same output as an uninterrupted run, including a Python component.
  A branch with new future prescribed data equals a fresh run with that data.
  A format mismatch errors.
  A changed declared unit fails restore and names the variable.
- Changelog: `breaking` (checkpoint format, TOML removal) and `feature` (output records, branching).

### 7. `refactor/remove-coupling-maps`

- Delete the `Model` constructors `new`, `with_grid_weights`, `with_transforms`, `with_unit_conversions` and `with_variable_sources`.
  Make `Model` construction `pub(crate)`.
- Delete `read_transforms`, `write_transforms`, `unit_conversions`, `variable_sources`, `required_transformations`, their fields and `tuple_map_serde`.
- Delete `grid_transform.rs`, `RequiredTransformation`, `UnitConversionInfo`, `TransformInfo` and the old `ConversionInfo`.
- Remove `inputs`, `outputs`, `input_names` and `output_names` from the `Component` trait, leaving `definitions()`.
- Remove `CGraph` and the synthetic root from `Model` if nothing else needs them.
- Update `docs/grids.md`, `docs/key_concepts.md` and the `.pyi` stubs.
- Gate: `make lint`, `make validate-pyi`, `make test`, `make parity`.
- Changelog: `breaking` (removed constructors, accessors and trait methods).
