# Tutorials

Learn RSCM through hands-on examples. Each tutorial builds on concepts from previous ones.

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

[**Start Tutorial: Coupled Models**](notebooks/coupled_model.py)

---

## Tutorial 2: Working with Spatial Grids

**Learn to**: Use spatially-resolved variables in components

**You will**:

- Declare grid-based inputs and outputs (`FourBox`, `Hemispheric`)
- Access regional values in `solve()` methods
- Create and return `FourBoxSlice` and `HemisphericSlice` outputs
- Aggregate grid values to scalars
- Understand grid coupling validation

**Topics covered**:

- Grid types: `Scalar`, `FourBox`, `Hemispheric`
- Declaring grids: `grid="FourBox"` in Python, `grid = "FourBox"` in Rust
- Accessing regional values: `current_all()`, `current(region)`
- Output types: `FourBoxSlice`, `HemisphericSlice`
- Grid aggregation and transformation

[**Start Tutorial: Working with Spatial Grids**](notebooks/grid_variables.py)

[**Reference: Spatial Grids Documentation**](grids.md)

---

## Tutorial 3: State Serialisation

**Learn to**: Save and restore model state for reproducibility

**You will**:

- Serialise model state to JSON and TOML formats
- Restore models from serialised state
- Continue runs from checkpoints
- Share model configurations with collaborators

**Topics covered**:

- Model serialisation with `serde`
- JSON vs TOML format tradeoffs
- Checkpoint-restart workflows
- Version compatibility considerations

[**Start Tutorial: State Serialisation**](notebooks/state_serialisation.py)

---

## Tutorial 4: Model Debugging and Inspection

**Learn to**: Debug models and trace data flow through components

**You will**:

- Visualise the component dependency graph
- Inspect variables in `TimeseriesCollection`
- Trace data flow between components
- Step through model execution for debugging
- Use serialisation for checkpoints and state inspection
- Understand common error messages and their causes

**Topics covered**:

- The `as_dot()` method for graph visualisation
- Accessing variables with `get_timeseries_by_name()`
- Step-by-step execution with `model.step()`
- Model serialisation with `to_toml()` and `from_toml()`
- Diagnosing missing variable and initial value errors

[**Start Tutorial: Model Debugging and Inspection**](notebooks/debugging_inspection.py)

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
- [API Reference](api/): Full Python API documentation

---

## Extending RSCM

Want to create custom components? See the [Developer Guide](developers/index.md) for:

- [Python Components](notebooks/component_python.py): Create custom components with typed inputs and outputs
- [Rust Components](notebooks/component_rust.md): Build high-performance components in Rust
