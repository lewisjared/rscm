# Tutorials

Learn RSCM through hands-on examples. Each tutorial builds on concepts from previous ones. See the [Python API](api/rscm/) for full class documentation.

## Learning Paths

Choose your path based on what you want to achieve:

### Running Existing Models
If you want to use RSCM to run climate simulations with pre-built components:

1. [Getting Started](getting_started.md): Installation and first model
2. [Coupled Models](#tutorial-1-coupled-models): Combine components
3. [Scenario Pipeline](#tutorial-6-building-a-complete-scenario-pipeline): Historical + future runs

### Calibrating Models
If you want to estimate model parameters from observations:

1. [Getting Started](getting_started.md): Installation and basics
2. [Coupled Models](#tutorial-1-coupled-models): Understand model structure
3. [Calibration](#tutorial-7-model-calibration-and-uncertainty-quantification): Parameter estimation and MCMC

### Running MAGICC
If you want to run the MAGICC reimplementation from emissions or concentrations:

1. [Coupled Models](#tutorial-1-coupled-models): Combine components
2. [A Coupled MAGICC Model](#tutorial-8-a-coupled-magicc-model):
   Assemble MAGICC7's full chain and inspect the module graph
3. [MAGICC Runs](magicc_runs.md): `run_magicc7` and the smaller runners

### Creating New Components
If you want to extend RSCM with custom physics:

1. [Key Concepts](key_concepts.md): Core architecture
2. [Python Components](#extending-rscm): Rapid prototyping
3. [Rust Components](#extending-rscm): Production performance
4. [Variable Schemas](#tutorial-3-variable-schemas-and-aggregation): Aggregation patterns

---

## Tutorial 1: Coupled Models

**Learn to**: Combine multiple components into a complete climate model

**You will**:

- Use `ModelBuilder` to assemble components
- Define a time axis for model execution
- Provide exogenous data as `Timeseries` objects
- Set initial values for state variables
- Run the model and extract results
- Visualise the component dependency graph

**Topics covered**:

- The `ModelBuilder` pattern
- Time axis definition with `TimeAxis.from_values()` and `from_bounds()`
- Exogenous vs endogenous variables
- Model execution with `step()` and `run()`
- Extracting results from `TimeseriesCollection`
- Interpolation strategies for exogenous data: `Linear`, `Previous`, `Next`

[**Start Tutorial: Coupled Models**](notebooks/coupled_model.py)

---

## Tutorial 2: Working with Spatial Grids

**Learn to**: Use spatially-resolved variables in components

**You will**:

- Declare grid-based inputs and outputs (`FourBox`, `Hemispheric`)
- Access regional values in `solve()` methods
- Create and return `FourBoxSlice` and `HemisphericSlice` outputs
- Aggregate grid values to scalars
- Aggregate gridded outputs inside a model with a schema and `ModelBuilder.with_grid_weights()`
- Understand grid coupling rules and validation errors

**Topics covered**:

- Grid types: `Scalar`, `FourBox`, `Hemispheric`
- Declaring grids: `grid="FourBox"` in Python, `grid = "FourBox"` in Rust
- Accessing regional values: `at_start_all()`, `at_start(region)`
- Output types: `FourBoxSlice`, `HemisphericSlice`
- Reading grid results with `get_fourbox_timeseries_by_name()` and `get_hemispheric_timeseries_by_name()`
- Schema-driven aggregation and custom grid weights

[**Start Tutorial: Working with Spatial Grids**](notebooks/grid_variables.py)

[**Reference: Spatial Grids Documentation**](grids.md)

---

## Tutorial 3: Variable Schemas and Aggregation

**Learn to**: Combine multiple component outputs into derived values

**You will**:

- Create a `VariableSchema` declaring variables and aggregates
- Use Sum, Mean, and Weighted aggregation operations
- Build hierarchical aggregates (aggregates of aggregates)
- Handle missing data (NaN values) in aggregations
- Understand schema validation and error messages

**Topics covered**:

- The `VariableSchema` class and builder pattern
- Aggregation operations: `Sum`, `Mean`, `Weighted`
- Hierarchical forcing structures (e.g., Total ERF = GHG + Aerosol)
- NaN handling in aggregations
- Schema validation constraints
- Build-time validation of components against the schema
- Inspecting schema definitions (`variables`, `aggregates`)

[**Start Tutorial: Variable Schemas and Aggregation**](notebooks/variable_schema.py)

---

## Tutorial 4: State Serialisation

**Learn to**: Save and restore model state for reproducibility

**You will**:

- Serialise model state to TOML
- Restore models from serialised state
- Continue runs from checkpoint files
- Understand current serialisation limitations

**Topics covered**:

- `Model.to_toml()` and `Model.from_toml()`
- Writing and reading checkpoint files
- Internal component state in the `component_states` section
- Limitations: fixed time axis, no Python components, no format version

[**Start Tutorial: State Serialisation**](notebooks/state_serialisation.py)

---

## Tutorial 5: Model Debugging and Inspection

**Learn to**: Debug models and trace data flow through components

**You will**:

- Inspect execution order and variable sources with `debug_info()`
- Visualise the component dependency graph
- Inspect variables in `TimeseriesCollection`
- Trace data flow between components
- Step through model execution for debugging
- Use serialisation for checkpoints and state inspection
- Diagnose build errors, missing exogenous data and runtime errors

**Topics covered**:

- `debug_info()` in rich, plain and JSON formats
- The `as_dot()` method for graph visualisation
- Accessing variables with `get_timeseries_by_name()`
- Step-by-step execution with `model.step()`
- Model serialisation with `to_toml()` and `from_toml()`
- Diagnosing missing initial values and missing exogenous data
- Handling `RuntimeError` from `step()` and `run()`

[**Start Tutorial: Model Debugging and Inspection**](notebooks/debugging_inspection.py)

---

## Tutorial 6: Building a Complete Scenario Pipeline

**Learn to**: Run climate scenarios from a common historical baseline

**You will**:

- Run a historical spin-up period with observed emissions
- Extract state variables at a branch point (e.g. 2015) to initialise scenario runs
- Run multiple future scenarios (SSP-like pathways) from that point
- Collect and compare results across scenarios
- Export data for further analysis

**Topics covered**:

- Historical/future branching workflow
- State extraction with `model.timeseries()`
- Multi-scenario data management with `pandas`
- Time axis configuration for different periods
- End-of-century summary statistics

[**Start Tutorial: Building a Complete Scenario Pipeline**](notebooks/scenario_pipeline.py)

---

## Tutorial 7: Model Calibration and Uncertainty Quantification

**Learn to**: Estimate model parameters from observations and quantify uncertainty

**You will**:

- Define parameter priors with uniform, normal, and bounded distributions
- Create observation targets with measurement uncertainties
- Use point estimation (random search) to find best-fit parameters
- Run MCMC sampling to explore the full posterior distribution
- Check convergence with R-hat, effective sample size, and autocorrelation
- Visualise posterior distributions and parameter correlations
- Use checkpointing for long-running calibrations

**Topics covered**:

- The `ParameterSet` class for defining priors
- `Target` and `Observation` for calibration targets
- `PointEstimator` for maximum likelihood estimation
- `EnsembleSampler` for MCMC with the affine-invariant algorithm
- Convergence diagnostics: R-hat, ESS, autocorrelation time
- Chain persistence: save, load, merge, and checkpoint
- DataFrame integration for posterior analysis

[**Start Tutorial: Model Calibration**](notebooks/calibration_tutorial.py)

---

## Tutorial 8: A Coupled MAGICC Model

**Learn to**: Assemble MAGICC7's full emissions-driven chain from RSCM components

**You will**:

- Map MAGICC modules onto RSCM components and find their specs
- Configure the components from MAGICC7's own parameter names
- Feed prescribed aerosol, land-use and solar forcing in on the four-box grid
- Couple chemistry, carbon cycle, permafrost, forcing, climate and sea level components with `ModelBuilder`
- Visualise the component graph, including the lagged temperature, CO2 and climate sensitivity feedbacks
- Check the hand-built model against MAGICC7 v7.5.3 and `run_magicc7`
- Run a small climate sensitivity sweep and compare with the smaller runners

**Topics covered**:

- Registration order and start-of-step reads for feedback loops
- Four-box variables, grid weights and `FourBoxTimeseries`
- Forcing aggregation with efficacies and regional patterns
- Initial values for states and forcing
- Emissions-driven and concentration-prescribed MAGICC7 configurations
- `run_magicc7`, `run_magicc_emissions`, `run_magicc` and `MAGICCResult`
- What is validated against MAGICC7 v7.5.3

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
    .with_rust_component(component)
    .with_exogenous_variable("Variable Name", timeseries)
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
- [Python API](api/rscm/): Full API documentation

---

## Extending RSCM

Want to create custom components? See the [Developer Guide](developers/index.md) for:

- [Python Components](notebooks/component_python.py): Create custom components with typed inputs and outputs
- [Rust Components](notebooks/component_rust.md): Build high-performance components in Rust
