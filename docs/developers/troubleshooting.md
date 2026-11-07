# Troubleshooting Common Issues

This guide covers common issues encountered when building, running, and developing with RSCM.

## Build Issues

### Maturin Build Failures

**Symptom:** `maturin develop` or `make build-dev` fails with cryptic errors.

**Common causes and fixes:**

1. **Wrong manifest path**

    The `pyproject.toml` must point to the PyO3 crate, not the workspace root:

    ```toml
    [tool.maturin]
    manifest-path = "crates/rscm/Cargo.toml"  # Correct
    # manifest-path = "Cargo.toml"            # Wrong - workspace root has no package
    ```

2. **Stale build artifacts**

    After significant Rust changes, clean and rebuild:

    ```bash
    cargo clean
    make build-dev
    ```

3. **Wrong Python version**

    RSCM requires Python 3.13+. Check your active version:

    ```bash
    python --version
    ```

    If using the wrong version, activate the correct virtual environment:

    ```bash
    source .venv/bin/activate
    ```

### Extension Module Not Updated

**Symptom:** Python code doesn't reflect your Rust changes.

**Fix:** Always run `make build-dev` after modifying any `.rs` file. The extension module (`.so`/`.pyd`) must be rebuilt for changes to take effect.

### Cargo Lock Conflicts

**Symptom:** Dependency resolution errors after pulling changes.

**Fix:**

```bash
cargo update
uv lock
```

## Runtime Errors

### RscmError Types

RSCM uses specific error types to help diagnose issues. Here's what each means and how to fix it:

#### ExtrapolationNotAllowed

```
Extrapolation is not allowed. Target=1750, start of interpolation range=1850
```

**Cause:** Requesting data at a time outside the timeseries bounds.

**Fixes:**

- Extend your input data to cover the required time range
- Timeseries created from Python always allow extrapolation, so this comes from a Rust-built timeseries.
  Enable extrapolation on its strategy:

    ```rust
    let strategy = InterpolationStrategy::from(LinearSplineStrategy::new(true));
    ```

- Use `InterpolationStrategy.Previous` or `.Next` for data that should hold constant outside bounds

#### MissingInitialValue

```
Missing initial value for state variable 'Surface Air Temperature Change' in component 'TwoLayer'.
```

**Cause:** State variables require initial values before the model can run.

**Fix:** Provide initial values when building the model:

```python
model = (
    ModelBuilder()
    .with_component(component)
    .with_time_axis(time_axis)
    .with_initial_values({"Surface Air Temperature Change": 0.0, "Ocean Temperature|Layer 2": 0.0})
    .build()
)
```

#### MissingPrescribedInput

```
Inputs with no producer, prescribed data or initial value: 'Effective Radiative Forcing' (read by TwoLayer).
```

**Cause:** No component produces an input, and no prescribed data or initial value supplies it.

**Fix:** Supply the input as prescribed data:

```python
model = (
    ModelBuilder()
    .with_component(component)
    .with_time_axis(time_axis)
    .with_prescribed_variable("Effective Radiative Forcing", erf)
    .with_initial_values({"Surface Air Temperature Change": 0.0, "Ocean Temperature|Layer 2": 0.0})
    .build()
)
```

If the input is meant to be empty, call `with_missing_prescribed_allowed()` before `build()`.
The input is then NaN.

#### Missing input when calling `solve` directly

```
pyo3_runtime.PanicException: No timeseries with variable='Emissions|CO2'
```

**Cause:** A component's `solve` was called directly without one of its declared inputs.
Inside a model this does not happen.
A declared input with no producer and no prescribed data fails the build with `MissingPrescribedInput` instead.

**Fixes:**

- Check the exact variable name matches what the producing component outputs
- Verify all required components and prescribed variables are added to the model
- Check for typos in variable names (names are case-sensitive)
- If outputs are unexpectedly NaN, check `model.coupling()`.
  Inputs marked `(prescribed)` have no producing component.

#### GridTypeMismatch

```
Grid mismatch for variable 'Surface Air Temperature Change': producer component 'A' outputs four_box but consumer component 'B' expects scalar.
```

**Cause:** Two connected components use different spatial grids.

**Fixes:**

1. Change the producer component to output the required grid
2. Change the consumer component to accept the available grid
3. Insert a grid transformation component between them
4. Use automatic aggregation (four-box to scalar aggregation uses the grid weights)

#### UnsupportedGridTransformation

```
Unsupported grid transformation from Hemispheric to FourBox.
```

**Cause:** RSCM cannot automatically transform between these grids without additional physics assumptions.

**Fix:** Write a component that implements the transformation with appropriate physics.

#### GridTransformationNotSupported

```
Grid transformation not supported for variable 'Surface Air Temperature Change': cannot transform from scalar to four_box.
```

**Cause:** A component requires a finer grid resolution than the schema or producer provides. This occurs with [schema-driven auto-aggregation](../grids.md#schema-driven-auto-aggregation) when:

- A component declares a four-box input but the schema declares Scalar
- A component declares a Hemispheric input but the schema declares Scalar
- A component declares a four-box input but the schema declares Hemispheric

Disaggregation (broadcasting coarse data to finer grids) is not supported because it would require inventing spatial structure that doesn't exist.

**Fixes:**

1. **Change the consumer component** to accept the coarser resolution:

    ```python
    # Instead of:
    temp = Input("Surface Air Temperature Change", unit="K", grid="four_box")

    # Use:
    temp = Input("Surface Air Temperature Change", unit="K")  # Scalar
    ```

2. **Change the schema** to provide finer resolution (if a component produces it):

    ```python
    schema.add_variable("Surface Air Temperature Change", "K", GridType.FourBox)
    ```

3. **Create an explicit disaggregation component** with domain-specific assumptions:

    ```python
    class ScalarToFourBox(Component):
        """Disaggregate a scalar temperature to four-box with configurable ratios."""

        scalar_temp = Input("Surface Air Temperature Change", unit="K")
        regional_temp = Output(
            "Regional Surface Air Temperature Change", unit="K", grid="four_box"
        )

        def __init__(self, ocean_ratio=1.05, land_ratio=0.95):
            self.ocean_ratio = ocean_ratio
            self.land_ratio = land_ratio

        def solve(self, t_current, t_next, inputs):
            ocean = inputs.scalar_temp.get() * self.ocean_ratio
            land = inputs.scalar_temp.get() * self.land_ratio
            return self.Outputs(
                regional_temp=FourBoxSlice(
                    northern_ocean=ocean, northern_land=land,
                    southern_ocean=ocean, southern_land=land,
                )
            )
    ```

    The [Spatial Grids tutorial](../notebooks/grid_variables.py) builds and runs a disaggregation component.

#### FeedbackLoop

```
Feedback loop among components [CarbonCycle, TwoLayer] through variables [...].
Break it with one of: with_lag_for("Surface Air Temperature Change", "CarbonCycle"), ...
```

**Cause:** Components form a feedback loop that no lag breaks, where A depends on B and B depends on A.

**Fixes:**

- Lag one edge of the loop with one of the suggested `with_lag_for` calls,
  or with `with_lag(variable)` to lag every reader of a variable
- Restructure component boundaries so dependencies flow in one direction
- Combine tightly coupled components into a single component

### Python-Rust Ownership Issues

**Symptom:** Changes made to objects in Python don't persist or affect Rust.

**Cause:** PyO3 clones Rust objects when passing them to Python. Python operates on copies, not references.

**Fix:** Build the data you need before handing it over, rather than editing it afterwards:

```python
# Wrong: both calls return copies, so the collection is unchanged
ts = collection.get_timeseries_by_name("Surface Air Temperature Change")
ts.set(0, 999.0)
ts.values()[0] = 999.0

# Correct: create the timeseries with the right values, then add it
values[0] = 999.0
collection.add_timeseries("Surface Air Temperature Change", Timeseries(values, time_axis, "K", strategy))
```

`add_timeseries` panics if the name already exists (`pyo3_runtime.PanicException` in Python).
Build a new collection to replace a timeseries.

## PyO3 Binding Issues

### ImportError on Module Load

**Symptom:**

```python
>>> import rscm
ImportError: cannot import name '_lib' from 'rscm'
```

**Fixes:**

1. Ensure you've built the extension: `make build-dev`
2. Check you're using the correct virtual environment
3. Verify `python/rscm/_lib.abi3.so` (or `.pyd` on Windows) exists

### AttributeError for PyO3 Classes

**Symptom:** `AttributeError: type object 'Component' has no attribute 'from_dict'`

**Cause:** The `.pyi` stub files may be out of sync with the Rust bindings.

**Fixes:**

- Check the actual available methods, for example `dir(rscm.core.ModelBuilder)`
- Update `.pyi` files to match Rust bindings after API changes
- Run `make validate-pyi` to find mismatches between the stubs and the compiled module

### Serialisation Errors

**Symptom:**

```
ValueError: missing field `co2_pi`
```

**Cause:** The Python dict doesn't match the Rust struct fields.

**Fixes:**

- Check all required fields are present in the dict
- Verify field names match exactly (Rust uses snake_case)
- Check field types match (an `int` is accepted for an `f64` field, but a string is not)

## Debugging Tips

### Inspecting the Dependency Graph

To see how components connect:

```python
model = builder.build()
print(model.as_dot())  # Graphviz DOT format
```

### Tracing Data Flow

Rust components do not log through Python's `logging`.
Step the model and inspect the state instead:

```python
model.step()
print(model.coupling(format="plain"))
```

See the [Debugging and Inspection](../notebooks/debugging_inspection.py) tutorial for more.

### Checking Available Variables

List all variables in a timeseries collection:

```python
print(collection.names())
```

### Validating Component Definitions

Verify a component's inputs and outputs:

```python
component = MyComponent(param=1.0)
for defn in component.definitions():
    print(f"{defn.role}: {defn.name} [{defn.unit}]")
```

## Common Mistakes

### Unit Mismatches

Components define expected units.
The builder converts prescribed data into the unit the model stores it in,
for example `MtCO2/yr` to `GtC/yr`.
That is the schema's unit, or without a schema the unit the components declare.
With a `VariableSchema`, the model also converts compatible units when components read inputs.
Incompatible units, such as `K` and `W/m^2`, fail when the model is built,
as do units of different species, such as `Mt CH4/yr` and `Mt CO2/yr`,
and unit strings the parser cannot read.
See [Physical Units](../notebooks/units_tutorial.py).

### Time Axis Alignment

All timeseries in a model run should use compatible time axes:

```python
# Ensure emissions and other inputs cover the full simulation period
times = np.arange(1750, 2100, 1)  # Annual from 1750-2100
```

### Forgetting to Rebuild

After any Rust change, always:

```bash
make build-dev
```

Python-only changes don't require rebuilding.
