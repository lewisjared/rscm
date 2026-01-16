# Tasks: Add Python Grid Output Support

## Phase 1: PyStateValue Class

### Task 1.1: Create PyStateValue PyO3 class

**File:** `crates/rscm-core/src/python/state.rs`

**Changes:**
- Add `PyStateValue` struct wrapping `StateValue`
- Implement `#[staticmethod]` factory methods: `scalar`, `four_box`, `hemispheric`
- Implement type check methods: `is_scalar`, `is_four_box`, `is_hemispheric`
- Implement accessor methods: `as_scalar`, `as_four_box`, `as_hemispheric`, `to_scalar`
- Implement `__repr__`

**Verification:** Rust unit tests for PyStateValue construction and accessors

### Task 1.2: Register PyStateValue in Python module

**File:** `crates/rscm-core/src/python/mod.rs`

**Changes:**
- Add `PyStateValue` to the `state` submodule exports
- Ensure it's accessible as `rscm._lib.core.state.StateValue`

**Verification:** Python import test

---

## Phase 2: RustComponent.solve() Changes

### Task 2.1: Update impl_component! macro to return PyStateValue

**File:** `crates/rscm-core/src/python/component.rs`

**Changes:**
- Change return type from `HashMap<String, FloatValue>` to `HashMap<String, PyStateValue>`
- Remove `to_scalar()` conversion
- Wrap each `StateValue` in `PyStateValue`

**Verification:** Rust compile + Python test calling RustComponent.solve()

---

## Phase 3: PythonComponent.solve() Changes

### Task 3.1: Update PythonComponent to extract PyStateValue

**File:** `crates/rscm-core/src/python/component.rs`

**Changes:**
- Update output extraction to handle `PyStateValue` objects
- Extract inner `StateValue` from `PyStateValue`
- Add fallback for raw float values (legacy compatibility)

**Verification:** Python test with component returning StateValue outputs

---

## Phase 4: Python Component Base Class

### Task 4.1: Update Outputs.to_dict() to return StateValue

**File:** `python/rscm/component.py`

**Changes:**
- Import `StateValue` from `rscm._lib.core.state`
- Modify `to_dict()` in `_create_outputs_class()`:
  - Wrap `FourBoxSlice` with `StateValue.four_box()`
  - Wrap `HemisphericSlice` with `StateValue.hemispheric()`
  - Wrap float with `StateValue.scalar()`

**Verification:** Unit test for Outputs.to_dict() return types

---

## Phase 5: Type Stubs

### Task 5.1: Add StateValue type stub

**File:** `python/rscm/_lib/core/state.pyi`

**Changes:**
- Add `StateValue` class with all methods
- Document factory methods and accessors

**Verification:** mypy/pyright type checking

### Task 5.2: Update RustComponent type stub

**File:** `python/rscm/_lib/core/__init__.pyi`

**Changes:**
- Update `RustComponent.solve()` return type to `dict[str, StateValue]`
- Add `StateValue` import if needed

**Verification:** mypy/pyright type checking

### Task 5.3: Export StateValue from core module

**File:** `python/rscm/_lib/core/__init__.pyi`

**Changes:**
- Add `StateValue` to public exports

**Verification:** Python import test

---

## Phase 6: Integration Tests

### Task 6.1: Add Python component grid output test

**File:** `tests/test_component_grid_outputs.py` (new)

**Tests:**
- Python component with FourBox output
- Python component with Hemispheric output
- Python component with mixed scalar and grid outputs
- Verify outputs written to correct timeseries types in model

**Verification:** pytest passes

### Task 6.2: Add Rust component grid output test

**File:** `tests/test_component_grid_outputs.py`

**Tests:**
- RustComponent.solve() returning FourBox StateValue
- RustComponent.solve() returning Hemispheric StateValue
- Accessing grid values via as_four_box()/as_hemispheric()

**Verification:** pytest passes

### Task 6.3: Add StateValue unit tests

**File:** `tests/test_state_value.py` (new)

**Tests:**
- StateValue.scalar() creation and accessors
- StateValue.four_box() creation and accessors
- StateValue.hemispheric() creation and accessors
- to_scalar() aggregation for grid types
- Type check methods return correct values

**Verification:** pytest passes

---

## Dependencies

```
Task 1.1 ─┬─> Task 1.2 ─┬─> Task 2.1 ─> Task 6.2
          │             │
          │             └─> Task 3.1 ─┬─> Task 6.1
          │                           │
          └─────────────────> Task 4.1 ┘

Task 5.1 ─┬─> Task 5.2
          └─> Task 5.3

Task 1.1 ─────────────────────────────> Task 6.3
```

- Phase 1 must complete before Phases 2-4
- Phase 5 (type stubs) can run in parallel after Task 1.1
- Phase 6 tests require corresponding implementation tasks
