## Context

RSCM aims to enable climate scientists to contribute new model components without deep Rust expertise. The current API requires understanding trait objects, serde, typetag, and manual string-based state management. This proposal introduces ergonomic patterns that reduce friction while maintaining zero-cost abstractions.

### Stakeholders

- **Climate scientists**: Primary users writing new components in Python or Rust
- **Framework maintainers**: Need to preserve type safety and performance

## Goals / Non-Goals

### Goals

- Reduce component boilerplate by 50%+ for simple cases
- Provide compile-time (Rust) and definition-time (Python) validation of variable access
- Enable historical data access (last N values) without performance penalty
- Clarify state variable semantics (replacing InputAndOutput)
- Enable coupler to validate and auto-transform grids between components
- Rewrite existing components to use new patterns (no backwards compatibility burden)

### Non-Goals

- Async solve (deferred - see Decision 7)
- Automatic unit conversion (defer to future work)
- Dynamic component discovery/plugin system

## Decisions

### Decision 1: TimeseriesWindow for State Access

**What**: Introduce `TimeseriesWindow<'a>` as a zero-cost view into a timeseries at a specific time index.

**Why**: Provides access to current, previous, and historical values without copying data or allocating. Scientists can compute derivatives, apply smoothing, or access full history as needed.

```rust
pub struct TimeseriesWindow<'a> {
    timeseries: &'a Timeseries<FloatValue>,
    current_index: usize,
    current_time: Time,
}

impl TimeseriesWindow<'_> {
    pub fn current(&self) -> FloatValue;
    pub fn previous(&self) -> Option<FloatValue>;
    pub fn at_offset(&self, offset: isize) -> Option<FloatValue>;
    pub fn last_n(&self, n: usize) -> ArrayView1<FloatValue>;
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue>;
}
```

**Alternatives considered**:

- Pass raw `&Timeseries` - too low-level, no current time context
- Pass `ArrayView` slices - loses interpolation capability
- Clone recent values into Vec - unnecessary allocation

### Decision 2: Typed State Accessors via Macro

**What**: A derive macro generates typed input/output structs from component definitions using struct-level attributes.

**Why**: Eliminates string-based lookups, provides IDE autocomplete, catches typos at compile time.

```rust
#[derive(ComponentIO)]
#[inputs(
    emissions_co2 { name = "Emissions|CO2", unit = "GtC/yr" },
    temperature { name = "Surface Temperature", unit = "K", grid = "FourBox" },
)]
#[states(
    concentration_co2 { name = "Atmospheric Concentration|CO2", unit = "ppm" },
)]
#[outputs(
    uptake { name = "Regional Uptake", unit = "GtC", grid = "FourBox" },
)]
pub struct CarbonCycle {
    pub tau: f64,
    pub conc_pi: f64,
}

// Generated:
struct CarbonCycleInputs<'a> {
    pub emissions_co2: TimeseriesWindow<'a>,            // Scalar
    pub temperature: GridTimeseriesWindow<'a, FourBox>, // FourBox
    pub concentration_co2: TimeseriesWindow<'a>,        // State as input
}

struct CarbonCycleOutputs {
    pub concentration_co2: FloatValue,
    pub uptake: FourBoxSlice,  // Type-safe grid output
}
```

**Design rationale for struct-level attributes**:

Using `#[inputs(...)]` and `#[outputs(...)]` at the struct level rather than marker fields with `#[input]` attributes avoids several DX problems:

1. **No phantom fields**: No need for `_emissions: ()` placeholder fields that store nothing
2. **No serde ceremony**: No `#[serde(skip)]` required on each marker field
3. **No underscore convention**: No need to prefix fields with `_` to suppress unused warnings
4. **Clean constructors**: No explicit initialization of phantom fields in `new()` methods
5. **Clear separation**: Component parameters (like `tau`, `conc_pi`) are clearly distinct from I/O declarations

**Alternatives considered**:

- VarKey registry without macro - still requires manual struct definition
- Trait-based access patterns - more complex, less ergonomic
- Keep string-based with better errors - doesn't provide autocomplete
- Phantom field approach with `#[input]` on fields - requires `()` fields, `#[serde(skip)]`, and underscore prefixes

### Decision 3: RequirementType::State Replacing InputAndOutput

**What**: New `RequirementType::State` with explicit initial value semantics.

**Why**: Current `InputAndOutput` is confusing - it's unclear whether a variable is truly bidirectional or just needs initialization. State variables have clear semantics: they read previous timestep value and write new value.

```rust
pub enum RequirementType {
    Input,       // Read from external source or other component
    Output,      // Write new value each timestep
    State,       // Read previous, write new (requires initial value)
    EmptyLink,   // Internal graph connectivity (unchanged)
}
```

Initial values can be:

- Provided via `ModelBuilder::with_initial_values()`
- Defaulted in component definition
- Loaded from configuration

**Alternatives considered**:

- Keep InputAndOutput - confusing, documented but not self-evident
- Split into two requirements - breaks single-source-of-truth for state variables

### Decision 4: Python Typed Inputs via Dataclass Generation

**What**: Generate Python dataclasses from component definitions at class creation time.

**Why**: Provides IDE autocomplete, type hints, and clear structure for Python scientists.

```python
class CarbonCycle(Component):
    inputs = [
        Input("Emissions|CO2", "GtC/yr", grid=Scalar),
        Input("Surface Temperature", "K", grid=FourBox),
        State("Atmospheric Concentration|CO2", "ppm", grid=Scalar),
    ]
    outputs = [
        Output("Regional Uptake", "GtC", grid=FourBox),
    ]

    def solve(self, t: float, dt: float, inputs: "CarbonCycle.Inputs") -> "CarbonCycle.Outputs":
        # inputs.emissions_co2.current - scalar value
        # inputs.temperature.current - array of 4 values
        # inputs.temperature.region(FourBoxRegion.NorthernOcean) - single region
        pass
```

**Alternatives considered**:

- TypedDict - no attribute access, still dict-like
- Pydantic models - additional dependency, heavier weight
- Raw dict with runtime validation - no IDE support

### Decision 5: Spatial Grid Requirements in Definitions

**What**: Add optional `grid` parameter to `RequirementDefinition` specifying the expected spatial structure.

**Why**: Enables the coupler to:

- Validate grid compatibility at build time
- Auto-insert aggregation transforms (FourBox -> Scalar)
- Fail fast when incompatible grids connect (Scalar -> FourBox)

```rust
pub struct RequirementDefinition {
    pub name: String,
    pub unit: String,
    pub requirement_type: RequirementType,
    pub grid: GridType,  // New field
}

pub enum GridType {
    Scalar,
    FourBox,
    Hemispheric,
    Any,  // Accepts any grid, component handles internally
}
```

**Grid compatibility rules**:

| Producer | Consumer | Result |
|----------|----------|--------|
| FourBox | Scalar | Auto-aggregate using grid weights |
| Hemispheric | Scalar | Auto-aggregate |
| FourBox | Hemispheric | Auto-transform (average ocean+land per hemisphere) |
| Scalar | FourBox | Error: cannot disaggregate without physical assumptions |
| Scalar | Any | Pass through |
| FourBox | Any | Pass through |

### Decision 6: Coupler Grid Validation and Auto-Transform

**What**: The model builder validates grid compatibility and inserts transform nodes in the component graph.

**Why**: Scientists shouldn't need to manually handle grid transformations for common cases (aggregation).

```rust
impl ModelBuilder {
    fn build(&self) -> RSCMResult<Model> {
        // For each edge in the component graph:
        // 1. Get producer's output grid
        // 2. Get consumer's input grid
        // 3. If compatible but different, insert GridTransform node
        // 4. If incompatible, return error with clear message
    }
}
```

**Error example**:

```
Error: Grid mismatch connecting components
  - Component 'GlobalEmissions' outputs 'Emissions|CO2' as Scalar
  - Component 'RegionalCarbon' requires 'Emissions|CO2' as FourBox
  - Cannot disaggregate Scalar to FourBox without explicit transform
  Hint: Add a disaggregation component or change RegionalCarbon to accept Scalar
```

### Decision 7: Defer Async Solve

**What**: Do not implement async solve in this proposal.

**Why**: Climate model components are CPU-bound (ODE solving, matrix operations). Async would add complexity with minimal benefit:

| Factor           | Impact                                                                        |
| ---------------- | ----------------------------------------------------------------------------- |
| Rust lifetimes   | `TimeseriesWindow<'a>` across await points requires `Pin` or `'static` bounds |
| PyO3 integration | Requires pyo3-asyncio, GIL release complexity                                 |
| Testing          | Async tests harder to write and reason about                                  |
| Actual use case  | Most components don't do I/O during solve                                     |

**Alternative for parallelism**: Use Rayon for parallel grid operations within components, or thread-pool for independent components at same BFS level.

### Decision 8: Separate rscm-macros Crate

**What**: Place proc-macro in dedicated `rscm-macros` crate.

**Why**:

- Proc-macros must be in their own crate (Rust requirement)
- Keeps `rscm-core` focused on runtime types
- Clear separation of compile-time vs runtime code

```
rscm/
├── rscm-macros/        # New: proc-macro crate
│   ├── Cargo.toml
│   └── src/lib.rs      # #[derive(ComponentIO)]
├── rscm-core/          # Re-exports macro, defines traits
└── rscm-components/    # Uses macro
```

### Decision 9: Typed Output Slices

**What**: Provide zero-cost wrapper types for output values that enforce type-safe region access instead of raw arrays with magic indices.

**Why**: Writing `output[2] = 1.5` is error-prone - which region is index 2? Typed slices use region enums for clarity and compile-time safety.

```rust
// Zero-cost wrapper - same memory layout as [FloatValue; 4]
#[repr(transparent)]
pub struct FourBoxSlice([FloatValue; 4]);

impl FourBoxSlice {
    pub fn new() -> Self {
        Self([FloatValue::NAN; 4])
    }

    // Builder pattern for ergonomic construction
    pub fn with(mut self, region: FourBoxRegion, value: FloatValue) -> Self {
        self.0[region as usize] = value;
        self
    }

    pub fn set(&mut self, region: FourBoxRegion, value: FloatValue) {
        self.0[region as usize] = value;
    }

    pub fn get(&self, region: FourBoxRegion) -> FloatValue {
        self.0[region as usize]
    }
}

// Similarly for other grids
pub struct HemisphericSlice([FloatValue; 2]);
pub struct ScalarSlice(FloatValue);  // Trivial but consistent API
```

**Usage in component solve**:

```rust
fn solve(&self, inputs: CarbonCycleInputs) -> CarbonCycleOutputs {
    // Type-safe, self-documenting
    CarbonCycleOutputs {
        concentration_co2: 285.0,
        uptake: FourBoxSlice::new()
            .with(FourBoxRegion::NorthernOcean, 1.2)
            .with(FourBoxRegion::NorthernLand, 0.8)
            .with(FourBoxRegion::SouthernOcean, 1.5)
            .with(FourBoxRegion::SouthernLand, 0.5),
    }
}
```

**Python equivalent**:

```python
def solve(self, inputs):
    return self.Outputs(
        concentration_co2=285.0,
        uptake=FourBoxSlice(
            northern_ocean=1.2,
            northern_land=0.8,
            southern_ocean=1.5,
            southern_land=0.5,
        ),
    )
```

**Alternatives considered**:

- Raw arrays with constants (`NORTHERN_OCEAN = 0`) - still allows wrong index, no IDE help
- Named struct fields - more verbose, doesn't scale to arbitrary grids
- Dict/HashMap - runtime overhead, no compile-time checking

## Risks / Trade-offs

### Risk: Macro Complexity

Derive macros add compile-time complexity and can produce confusing error messages.

**Mitigation**:

- Keep macro simple, generate straightforward code
- Provide good error spans pointing to the source
- Allow manual trait implementation as escape hatch

### Risk: Grid Transform Performance

Auto-inserted grid transforms add overhead.

**Mitigation**:

- Transforms are simple weighted averages (fast)
- Only inserted when grids differ
- Profile and optimize if bottleneck identified

### Trade-off: Macro vs Builder Pattern

Macros require proc-macro crate and increase compile times. Builder pattern is more verbose but debuggable.

**Decision**: Use macro for common case, support manual implementation for advanced cases.

## Migration Plan

1. Implement `TimeseriesWindow` and new `RequirementType::State`
2. Create `rscm-macros` crate with `#[derive(ComponentIO)]` using struct-level attributes
3. Rewrite `CO2ERF` component using new macro (simplest component)
4. Rewrite `CarbonCycleComponent` using new macro (has state variables)
5. Remove old `InputState.get_latest()` API
6. Update all tests
7. Update Python bindings
8. Update documentation
