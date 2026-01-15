# Design: Variable Registration System

## Context

MAGICC modules use a DATASTORE pattern where variables are centrally registered with comprehensive metadata. This enables:

- Temporal alignment between components (emissions are mid-year, concentrations are start-of-year)
- Automatic validation of unit compatibility
- Variable discovery and introspection

Preindustrial reference values are scenario/configuration-dependent and belong with the timeseries data, not the variable definition.

Currently, RSCM components declare requirements as ad-hoc strings with no validation. Components can connect variables with mismatched units or time conventions without any warning.

### Stakeholders

- Component authors: Need clear API for declaring variables
- Model builders: Need validation that connections are compatible
- Python users: Need to define custom variables at runtime
- Users: Need to discover what variables a model supports

## Goals / Non-Goals

### Goals

1. Provide a central registry for variable definitions with metadata
2. Require all components to use registered variables (enforce consistency)
3. Support both compile-time (Rust) and runtime (Python) variable registration
4. Enable validation of unit and time convention compatibility at model build time
5. Support preindustrial reference values as timeseries metadata
6. Keep the API simple - components should not need to write boilerplate

### Non-Goals

1. Automatic unit conversion (future work - requires proper units library)
2. Enforcing strict unit systems (units remain strings for now)

## Decisions

### Decision 1: Hybrid Registration (Static + Runtime)

Variables can be registered in two ways:

**Rust (compile-time)**: Using `inventory` crate for zero-cost static registration
```rust
define_variable!(
    CO2_CONCENTRATION,
    name = "Atmospheric Concentration|CO2",
    unit = "ppm",
    time_convention = TimeConvention::StartOfYear,
    description = "Atmospheric CO2 concentration",
);
```

**Python (runtime)**: Using registry API
```python
co2_conc = rscm.VariableDefinition(
    name="Atmospheric Concentration|CO2",
    unit="ppm",
    time_convention=rscm.TimeConvention.StartOfYear,
    description="Atmospheric CO2 concentration",
)
rscm.register_variable(co2_conc)
```

**Rationale**: Rust components benefit from compile-time safety. Python users need flexibility for custom scenarios and experimentation.

**Implementation**: `VariableRegistry` wraps:
- Static variables from `inventory` (immutable, loaded at startup)
- Runtime variables in a `RwLock<HashMap>` (mutable, thread-safe)

### Decision 2: TimeConvention Enum

```rust
pub enum TimeConvention {
    /// Value applies at start of year (Jan 1)
    StartOfYear,
    /// Value applies at mid-year (Jul 1)
    MidYear,
    /// Instantaneous value (no temporal averaging)
    Instantaneous,
}
```

**Rationale**: MAGICC explicitly documents these three conventions. Emissions are mid-year, concentrations/temperatures are start-of-year.

### Decision 3: Preindustrial Values as Timeseries Metadata

Preindustrial values are configuration-dependent, not intrinsic to variable definitions. They belong with the timeseries data:

```rust
pub struct TimeseriesItem {
    pub data: TimeseriesData,
    pub name: String,
    pub variable_type: VariableType,
    pub preindustrial: Option<PreindustrialValue>,  // NEW
}

pub enum PreindustrialValue {
    /// Global scalar value
    Scalar(f64),
    /// Four-box regional values [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
    FourBox([f64; 4]),
    /// Hemispheric values [Northern, Southern]
    Hemispheric([f64; 2]),
}
```

**Access via InputState**:
```rust
impl InputState {
    pub fn get_preindustrial(&self, name: &str) -> Option<&PreindustrialValue>;
    pub fn get_preindustrial_scalar(&self, name: &str) -> Option<f64>;
}
```

**Rationale**:
- Preindustrial values vary by scenario (historical vs counterfactual)
- Not all variables have preindustrial values (e.g., derived quantities)
- Data loaders can set preindustrial when loading scenario files
- Keeps VariableDefinition focused on intrinsic metadata

### Decision 4: RequirementDefinition with Variable Name

`RequirementDefinition` stores variable name, with metadata resolved from registry:

```rust
pub struct RequirementDefinition {
    pub variable_name: String,
    pub requirement_type: RequirementType,
    pub grid_type: GridType,
}

// Usage
RequirementDefinition::input("Atmospheric Concentration|CO2")
RequirementDefinition::four_box_output("Surface Temperature")
```

**Rationale**: Name-based lookup enables both Rust and Python variables. Simpler serialization than storing references.

### Decision 5: Validation at Model Build Time

`ModelBuilder::build()` validates:
1. All referenced variables exist in the registry
2. All output variables have unique names
3. Connected input/output pairs have compatible units (string equality)
4. Connected pairs have compatible time conventions (error for mismatches)
5. Connected pairs have compatible grid types (already implemented)

**Rationale**: Catch errors before simulation runs. Validation is cheap compared to multi-year climate runs.

### Decision 6: Time Convention Mismatches are Errors

Time convention mismatches fail the build. This enforces temporal consistency across the model.

**Rationale**: Silent mismatches lead to subtle bugs. Better to fail loudly and force explicit handling.

## Data Structures

```rust
// Variable definition (intrinsic metadata only)
pub struct VariableDefinition {
    pub name: String,
    pub unit: String,
    pub time_convention: TimeConvention,
    pub description: String,
}

// Preindustrial values (timeseries metadata)
pub enum PreindustrialValue {
    Scalar(f64),
    FourBox([f64; 4]),
    Hemispheric([f64; 2]),
}

impl PreindustrialValue {
    pub fn as_scalar(&self) -> Option<f64>;
    pub fn as_four_box(&self) -> Option<[f64; 4]>;
    pub fn to_scalar(&self) -> f64;  // Aggregates using area weights
}

// Timeseries with optional preindustrial
pub struct TimeseriesItem {
    pub data: TimeseriesData,
    pub name: String,
    pub variable_type: VariableType,
    pub preindustrial: Option<PreindustrialValue>,
}

// Registry
pub struct VariableRegistry {
    static_vars: Vec<&'static VariableDefinition>,
    runtime_vars: RwLock<HashMap<String, Arc<VariableDefinition>>>,
}
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Name collisions between static/runtime | Runtime registration checks for duplicates |
| Thread safety for runtime registry | Use `RwLock` for concurrent access |
| Serialization complexity | Store variable name in serialized form |
| Forgetting to set preindustrial | Components that need it should check and error clearly |

## Migration Plan

1. Add `inventory` to Cargo.toml dependencies
2. Implement `VariableDefinition`, `TimeConvention` types
3. Implement `PreindustrialValue` enum
4. Add `preindustrial` field to `TimeseriesItem`
5. Add `get_preindustrial()` methods to `InputState`
6. Implement `VariableRegistry` with static + runtime support
7. Create `define_variable!` macro using inventory
8. Define standard variables in `rscm-core/src/variables/` module
9. Refactor `RequirementDefinition` to use variable names
10. Update all existing components to use registered variables
11. Implement validation in `ModelBuilder::build()`
12. Add Python bindings for registration and introspection
