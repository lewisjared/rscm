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

    RSCM requires Python 3.11+. Check your active version:

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

### RSCMError Types

RSCM uses specific error types to help diagnose issues. Here's what each means and how to fix it:

#### ExtrapolationNotAllowed

```
ExtrapolationNotAllowed: target=1750.0, left of interpolation range=1850.0
```

**Cause:** Requesting data at a time outside the timeseries bounds.

**Fixes:**

- Extend your input data to cover the required time range
- Enable extrapolation on the interpolation strategy:

    ```python
    from rscm import LinearSplineStrategy
    strategy = LinearSplineStrategy(extrapolate=True)
    ```

- Use `PreviousStrategy` or `NextStrategy` for data that should hold constant outside bounds

#### MissingInitialValue

```
MissingInitialValue: variable='Temperature', component='TwoLayerModel'
```

**Cause:** State variables require initial values before the model can run.

**Fix:** Provide initial values when building the model:

```python
model = (
    ModelBuilder()
    .with_component(component)
    .with_initial_values({"Temperature": 0.0, "Heat Content|Deep Ocean": 0.0})
    .build()
)
```

#### VariableNotFound

```
VariableNotFound: name='Emissions|CO2', available=['Emissions|CH4', 'Forcing']
```

**Cause:** A component is trying to read an input variable that doesn't exist in the state.

**Fixes:**

- Check the exact variable name matches what the producing component outputs
- Verify all required components are added to the model
- Check for typos in variable names (names are case-sensitive)

#### GridTypeMismatch

```
GridTypeMismatch: variable='Temperature', producer_grid='FourBox', consumer_grid='Scalar'
```

**Cause:** Two connected components use different spatial grids.

**Fixes:**

1. Change the producer component to output the required grid type
2. Change the consumer component to accept the available grid type
3. Insert a grid transformation component between them
4. Use automatic aggregation (FourBox to Scalar aggregates using area weights)

#### UnsupportedGridTransformation

```
UnsupportedGridTransformation: from='Hemispheric', to='FourBox'
```

**Cause:** RSCM cannot automatically transform between these grid types without additional physics assumptions.

**Fix:** Create a custom component that implements the transformation with appropriate physics.

#### GridTransformationNotSupported

```
GridTransformationNotSupported: variable='Temperature', source_grid='Scalar', target_grid='FourBox'
```

**Cause:** A component requires a finer grid resolution than the schema or producer provides. This occurs with [schema-driven auto-aggregation](../grids.md#schema-driven-auto-aggregation) when:

- A component declares a FourBox input but the schema declares Scalar
- A component declares a Hemispheric input but the schema declares Scalar
- A component declares a FourBox input but the schema declares Hemispheric

Disaggregation (broadcasting coarse data to finer grids) is not supported because it would require inventing spatial structure that doesn't exist.

**Fixes:**

1. **Change the consumer component** to accept the coarser resolution:

    ```python
    # Instead of:
    temp = Input("Temperature", unit="K", grid="FourBox")

    # Use:
    temp = Input("Temperature", unit="K")  # Scalar
    ```

2. **Change the schema** to provide finer resolution (if a component produces it):

    ```python
    schema.add_variable("Temperature", "K", GridType.FourBox)
    ```

3. **Create an explicit disaggregation component** with domain-specific assumptions:

    ```python
    class ScalarToFourBox(Component):
        """Disaggregate scalar to FourBox with configurable ratios."""

        scalar_temp = Input("Temperature|Global", unit="K")
        regional_temp = Output("Temperature", unit="K", grid="FourBox")

        def __init__(self, ocean_ratio=1.05, land_ratio=0.95):
            self.ocean_ratio = ocean_ratio
            self.land_ratio = land_ratio

        def solve(self, t_current, t_next, inputs):
            global_temp = inputs.scalar_temp.at_start()
            return self.Outputs(
                regional_temp=FourBoxSlice(
                    northern_ocean=global_temp * self.ocean_ratio,
                    northern_land=global_temp * self.land_ratio,
                    southern_ocean=global_temp * self.ocean_ratio,
                    southern_land=global_temp * self.land_ratio,
                )
            )
    ```

#### CircularDependency

```
CircularDependency: cycle=['ComponentA', 'ComponentB', 'ComponentA']
```

**Cause:** Components form a dependency loop where A depends on B and B depends on A.

**Fixes:**

- Use state variables to break the cycle (state variables read from the previous timestep)
- Restructure component boundaries so dependencies flow in one direction
- Combine tightly coupled components into a single component

### Python-Rust Ownership Issues

**Symptom:** Changes made to objects in Python don't persist or affect Rust.

**Cause:** PyO3 clones Rust objects when passing them to Python. Python operates on copies, not references.

**Fix:** Use setter methods to update values:

```python
# Wrong - modifies a copy
data = collection.get_data("Temperature")
data[0] = 999.0  # This won't affect the collection

# Correct - uses setter method
collection.set_value("Temperature", 0, 999.0)
```

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
3. Verify the `.so` file exists in `python/rscm/_lib/`

### AttributeError for PyO3 Classes

**Symptom:** `AttributeError: type object 'Component' has no attribute 'from_dict'`

**Cause:** The `.pyi` stub files may be out of sync with the Rust bindings.

**Fixes:**

- Check the actual available methods: `dir(rscm.Component)`
- Update `.pyi` files to match Rust bindings after API changes

### Serialisation Errors

**Symptom:**

```
PyValueError: Failed to deserialize component: missing field 'parameter_name'
```

**Cause:** The Python dict doesn't match the Rust struct fields.

**Fixes:**

- Check all required fields are present in the dict
- Verify field names match exactly (Rust uses snake_case)
- Check field types match (e.g., `float` not `int` for `f64`)

## Debugging Tips

### Inspecting the Dependency Graph

To see how components connect:

```python
model = builder.build()
print(model.dependency_graph())
```

### Tracing Data Flow

Enable verbose output to trace values through components:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Checking Available Variables

List all variables in a timeseries collection:

```python
print(collection.variable_names())
```

### Validating Component Definitions

Verify a component's inputs and outputs:

```python
component = MyComponent(param=1.0)
for defn in component.definitions():
    print(f"{defn.direction}: {defn.name} [{defn.unit}]")
```

## Common Mistakes

### Unit Mismatches

Components define expected units. Ensure your data uses matching units:

```python
# Component expects "GtCO2", not "MtCO2"
# Conversion may be needed before passing to the model
```

### Time Grid Alignment

All timeseries in a model run should use compatible time grids:

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
