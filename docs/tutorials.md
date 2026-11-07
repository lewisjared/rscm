# Tutorials

Learn RSCM through hands-on examples.
Pick a learning path below, or jump to any tutorial.
See the [Python API](api/rscm/index.md) for full class documentation.

## Learning Paths

### Running Existing Models

1. [Getting Started](getting_started.md): Installation and first model
2. [Coupled Models](notebooks/coupled_model.py): Combine components
3. [Scenario Pipeline](notebooks/scenario_pipeline.py): Historical and future runs

### Calibrating Models

1. [Getting Started](getting_started.md): Installation and basics
2. [Coupled Models](notebooks/coupled_model.py): Understand model structure
3. [Calibration](notebooks/calibration_tutorial.py): Parameter estimation and MCMC

### Running MAGICC

1. [Coupled Models](notebooks/coupled_model.py): Combine components
2. [A Coupled MAGICC Model](notebooks/magicc_coupled_model.py): Assemble MAGICC7's full chain
3. [MAGICC Runs](magicc_runs.md): `run_magicc7` and the smaller runners

### Creating New Components

1. [Key Concepts](key_concepts.md): Core architecture
2. [Python Components](notebooks/component_python.py): Rapid prototyping
3. [Rust Components](notebooks/component_rust.md): Production performance
4. [Variable Schemas](notebooks/variable_schema.py): Aggregation patterns

---

## Tutorial 1: Coupled Models

**Learn to** combine multiple components into a complete climate model.

- Assemble components, a time axis, prescribed data and initial values with `ModelBuilder`
- Run the model with `step()` and `run()`, and read results from `TimeseriesCollection`
- Choose an interpolation strategy for prescribed data
- Visualise the component dependency graph

[**Start Tutorial: Coupled Models**](notebooks/coupled_model.py)

---

## Tutorial 2: Working with Spatial Grids

**Learn to** use spatially-resolved variables in components.

- Declare `FourBox` and `Hemispheric` inputs and outputs
- Read regional values with `get()` and return `FourBoxSlice` or `HemisphericSlice`
- Aggregate gridded outputs with a schema and `ModelBuilder.with_grid_weights()`
- Understand grid coupling rules and validation errors

[**Start Tutorial: Working with Spatial Grids**](notebooks/grid_variables.py)

[**Reference: Spatial Grids Documentation**](grids.md)

---

## Tutorial 3: Variable Schemas and Aggregation

**Learn to** combine multiple component outputs into derived values.

- Declare variables and aggregates in a `VariableSchema`
- Use `sum`, `mean` and `weighted_sum` aggregates, including aggregates of aggregates
- Handle missing data (NaN values) in aggregations
- Understand schema validation and its error messages

[**Start Tutorial: Variable Schemas and Aggregation**](notebooks/variable_schema.py)

---

## Tutorial 4: State Serialisation

**Learn to** save and restore model state for reproducibility.

- Serialise a model with `Model.to_toml()` and restore it with `Model.from_toml()`
- Continue runs from checkpoint files
- Find internal component state in the `component_states` section
- Know the limits: fixed time axis, no Python components, no format version

[**Start Tutorial: State Serialisation**](notebooks/state_serialisation.py)

---

## Tutorial 5: Model Debugging and Inspection

**Learn to** debug models and trace data flow through components.

- Inspect execution order, lags and variable sources with `coupling()`
- Visualise the dependency graph with `as_dot()`
- Step through a run with `model.step()`
- Diagnose build errors, missing prescribed data and runtime errors

[**Start Tutorial: Model Debugging and Inspection**](notebooks/debugging_inspection.py)

---

## Tutorial 6: Building a Complete Scenario Pipeline

**Learn to** run climate scenarios from a common historical baseline.

- Run a historical period with observed emissions
- Extract state at a branch point (e.g. 2015) to initialise scenario runs
- Run several SSP-like pathways from that point
- Compare and export the results with `pandas`

[**Start Tutorial: Building a Complete Scenario Pipeline**](notebooks/scenario_pipeline.py)

---

## Tutorial 7: Model Calibration and Uncertainty Quantification

**Learn to** estimate model parameters from observations and quantify uncertainty.

- Define priors with `ParameterSet` and targets with `Target` and `Observation`
- Find best-fit parameters with `PointEstimator`
- Sample the posterior with `EnsembleSampler` and check convergence (R-hat, ESS, autocorrelation)
- Save, load, merge and checkpoint chains

[**Start Tutorial: Model Calibration**](notebooks/calibration_tutorial.py)

---

## Tutorial 8: A Coupled MAGICC Model

**Learn to** assemble MAGICC7's full emissions-driven chain from RSCM components.

- Map MAGICC modules onto RSCM components and configure them from MAGICC7's parameter names
- Feed prescribed forcing in on the four-box grid
- Couple chemistry, carbon cycle, forcing, climate and sea level with `ModelBuilder`
- Check the hand-built model against MAGICC7 v7.5.3 and `run_magicc7`

[**Start Tutorial: A Coupled MAGICC Model**](notebooks/magicc_coupled_model.py)

[**Reference: MAGICC Runs**](magicc_runs.md)

---

## Quick Reference

### Common Patterns

**Building a model (Python)**:

```python
model = (
    ModelBuilder()
    .with_time_axis(TimeAxis.from_values(np.arange(2000, 2101)))
    .with_component(component)
    .with_prescribed_variable("Variable Name", timeseries)
    .with_initial_values({"State Variable": initial_value})
).build()
```

**Running and getting results**:

```python
model.run()
results = model.timeseries()
data = results.get_timeseries_by_name("Output Variable")
```

### Further Reading

- [Key Concepts](key_concepts.md): Detailed explanation of core abstractions
- [Spatial Grids](grids.md): Complete grid system documentation
- [Physical Units](notebooks/units_tutorial.py): Unit parsing, conversion, and climate-specific units
- [Python API](api/rscm/index.md): Full API documentation

---

## Extending RSCM

Want to write your own components? See the [Developer Guide](developers/index.md) for:

- [Python Components](notebooks/component_python.py): Write components with typed inputs and outputs
- [Rust Components](notebooks/component_rust.md): Build high-performance components in Rust
