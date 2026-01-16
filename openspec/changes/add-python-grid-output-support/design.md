# Design: Python Grid Output Support

## Current State

### Python → Rust (PythonComponent)

```
Python solve() → dict[str, float] → Rust extracts as HashMap<String, FloatValue> → StateValue::Scalar only
```

The `Outputs.to_dict()` method converts `FourBoxSlice`/`HemisphericSlice` to `list[float]`, but the Rust side (`crates/rscm-core/src/python/component.rs:169`) extracts only as `HashMap<String, FloatValue>`, losing grid structure.

### Rust → Python (RustComponent.solve())

```
Rust OutputState (HashMap<String, StateValue>) → converted via to_scalar() → HashMap<String, FloatValue>
```

All grid values are aggregated to scalars before returning to Python (`crates/rscm-core/src/python/component.rs:68-73`).

## Chosen Approach: Native PyO3 Types

Expose `StateValue` as a PyO3 class and use it directly in the Python API.

**Rationale:**
1. Most type-safe option - Python type checkers understand the types
2. Direct Rust type usage eliminates serialization overhead
3. Consistent with how other core types (FourBoxSlice, HemisphericSlice) are already exposed
4. Breaking change is acceptable per user preference

## Detailed Design

### PyStateValue Class

Expose `StateValue` as a Python class with factory methods:

```python
class StateValue:
    """Represents a value that can be scalar or spatially-resolved."""

    @staticmethod
    def scalar(value: float) -> StateValue: ...

    @staticmethod
    def four_box(slice: FourBoxSlice) -> StateValue: ...

    @staticmethod
    def hemispheric(slice: HemisphericSlice) -> StateValue: ...

    def is_scalar(self) -> bool: ...
    def is_four_box(self) -> bool: ...
    def is_hemispheric(self) -> bool: ...

    def as_scalar(self) -> float | None: ...
    def as_four_box(self) -> FourBoxSlice | None: ...
    def as_hemispheric(self) -> HemisphericSlice | None: ...

    def to_scalar(self) -> float: ...  # Aggregates grid values if needed
```

### Rust Implementation

```rust
#[pyclass(name = "StateValue")]
#[derive(Debug, Clone)]
pub struct PyStateValue(pub StateValue);

#[pymethods]
impl PyStateValue {
    #[staticmethod]
    fn scalar(value: FloatValue) -> Self {
        Self(StateValue::Scalar(value))
    }

    #[staticmethod]
    fn four_box(slice: PyFourBoxSlice) -> Self {
        Self(StateValue::FourBox(slice.0))
    }

    #[staticmethod]
    fn hemispheric(slice: PyHemisphericSlice) -> Self {
        Self(StateValue::Hemispheric(slice.0))
    }

    fn is_scalar(&self) -> bool { self.0.is_scalar() }
    fn is_four_box(&self) -> bool { self.0.is_four_box() }
    fn is_hemispheric(&self) -> bool { self.0.is_hemispheric() }

    fn as_scalar(&self) -> Option<FloatValue> { self.0.as_scalar() }
    fn as_four_box(&self) -> Option<PyFourBoxSlice> {
        self.0.as_four_box().map(|s| PyFourBoxSlice(*s))
    }
    fn as_hemispheric(&self) -> Option<PyHemisphericSlice> {
        self.0.as_hemispheric().map(|s| PyHemisphericSlice(*s))
    }

    fn to_scalar(&self) -> FloatValue { self.0.to_scalar() }
}
```

### Python Outputs Class Changes

The generated `Outputs` class changes its `to_dict()` method to return `StateValue` objects:

```python
def to_dict(self) -> dict[str, StateValue]:
    """Convert outputs to a dictionary for Rust interop."""
    result: dict[str, StateValue] = {}
    for field_name, (var_name, grid, _) in self._field_info.items():
        value = getattr(self, field_name)
        if isinstance(value, FourBoxSlice):
            result[var_name] = StateValue.four_box(value)
        elif isinstance(value, HemisphericSlice):
            result[var_name] = StateValue.hemispheric(value)
        else:
            result[var_name] = StateValue.scalar(value)
    return result
```

### PythonComponent.solve() Changes

```rust
fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
    Python::attach(|py| {
        // ... existing input handling ...

        let py_result = /* call Python solve */;
        let py_dict: &PyDict = py_result.downcast()?;

        let mut output_state = OutputState::new();
        for (key, value) in py_dict.iter() {
            let key: String = key.extract()?;
            let state_value: PyStateValue = value.extract()?;
            output_state.insert(key, state_value.0);
        }
        Ok(output_state)
    })
}
```

### RustComponent.solve() Changes

```rust
// In impl_component! macro:
pub fn solve(
    &mut self,
    t_current: Time,
    t_next: Time,
    collection: PyTimeseriesCollection,
) -> PyResult<HashMap<String, PyStateValue>> {
    let input_state = extract_state(&collection.0, self.0.input_names(), t_current);
    let output_state = self.0.solve(t_current, t_next, &input_state)?;

    Ok(output_state
        .into_iter()
        .map(|(key, value)| (key, PyStateValue(value)))
        .collect())
}
```

### Type Stub Updates

```python
# python/rscm/_lib/core/state.pyi

class StateValue:
    """Represents a value that can be scalar or spatially-resolved."""

    @staticmethod
    def scalar(value: float) -> StateValue: ...
    @staticmethod
    def four_box(slice: FourBoxSlice) -> StateValue: ...
    @staticmethod
    def hemispheric(slice: HemisphericSlice) -> StateValue: ...

    def is_scalar(self) -> bool: ...
    def is_four_box(self) -> bool: ...
    def is_hemispheric(self) -> bool: ...

    def as_scalar(self) -> float | None: ...
    def as_four_box(self) -> FourBoxSlice | None: ...
    def as_hemispheric(self) -> HemisphericSlice | None: ...

    def to_scalar(self) -> float: ...

    def __repr__(self) -> str: ...

# python/rscm/_lib/core/__init__.pyi

class RustComponent:
    def solve(
        self,
        t_current: float,
        t_next: float,
        collection: TimeseriesCollection
    ) -> dict[str, StateValue]: ...
```

## Breaking Changes

1. **RustComponent.solve() return type**: Changes from `dict[str, float]` to `dict[str, StateValue]`
   - Existing code accessing values directly needs to call `.to_scalar()` or `.as_scalar()`

2. **Python Outputs.to_dict() return type**: Changes from `dict[str, float]` to `dict[str, StateValue]`
   - Internal change, affects PythonComponent wrapper

3. **Legacy dict-based Python components**: Continue to work with migration path
   - Detect if return value is dict of floats vs dict of StateValue
   - Auto-wrap floats in StateValue::Scalar for backwards compatibility

## Migration Path

For users with existing code calling `RustComponent.solve()`:

```python
# Before
results = component.solve(t_current, t_next, collection)
temp = results["Temperature"]  # float

# After
results = component.solve(t_current, t_next, collection)
temp = results["Temperature"].to_scalar()  # or .as_scalar() for Optional
# Or for grid access:
temp_slice = results["Temperature"].as_four_box()
```
