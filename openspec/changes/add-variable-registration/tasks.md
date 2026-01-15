# Tasks: Variable Registration System

## 1. Core Types

- [x] 1.1 Add `inventory` crate to Cargo.toml
- [x] 1.2 Create `rscm-core/src/variable.rs` module
- [x] 1.3 Implement `TimeConvention` enum (StartOfYear, MidYear, Instantaneous)
- [x] 1.4 Implement `VariableDefinition` struct (name, unit, time_convention, description)
- [x] 1.5 Add unit tests for core types

## 2. Preindustrial Value (Timeseries Metadata)

- [x] 2.1 Implement `PreindustrialValue` enum (Scalar, FourBox, Hemispheric)
- [x] 2.2 Implement convenience methods (as_scalar, as_four_box, to_scalar)
- [x] 2.3 Add `preindustrial: Option<PreindustrialValue>` field to TimeseriesItem
- [x] 2.4 Add `get_preindustrial()` and `get_preindustrial_scalar()` methods to InputState
- [x] 2.5 Update TimeseriesCollection methods to accept optional preindustrial
- [x] 2.6 Add unit tests for preindustrial functionality

## 3. Variable Registry

- [x] 3.1 Implement `VariableRegistry` struct with static + runtime storage
- [x] 3.2 Implement `get()` method for variable lookup
- [x] 3.3 Implement `register()` method for runtime registration (with duplicate check)
- [x] 3.4 Implement `list()` method for enumeration
- [x] 3.5 Create global registry instance (lazy_static or OnceCell)
- [x] 3.6 Add unit tests for registry operations

## 4. Static Registration (Rust)

- [x] 4.1 Create `define_variable!` macro using inventory
- [x] 4.2 Implement inventory collection for static variables
- [x] 4.3 Wire static variables into global registry at startup
- [x] 4.4 Add unit tests for macro and static registration

## 5. Refactor RequirementDefinition

- [x] 5.1 Rename `name` field to `variable_name` (keep `unit` field - components declare expected units)
- [x] 5.2 Keep existing constructors (API unchanged: `scalar_input(name, unit)`, etc.)
- [x] 5.3 Add `time_convention()` accessor method (registry lookup - intrinsic to variable)
- [x] 5.4 Update serialization to use `variable_name` field name
- [x] 5.5 Add unit tests for RequirementDefinition

## 6. Model Builder Validation

- [x] 6.1 Add variable existence validation (check registry)
- [x] 6.2 Add unit compatibility validation (compare between connected components, not registry)
- [x] 6.3 Add time convention compatibility validation (from registry, error on mismatch)
- [x] 6.4 Create RSCMError variants: UnregisteredVariableError, UnitMismatchError, TimeConventionMismatchError
- [x] 6.5 Add integration tests for validation scenarios

## 7. Standard Variable Definitions

- [x] 7.1 Create `rscm-core/src/standard_variables.rs` module
- [x] 7.2 Define CO2-related variables (concentration, emissions, forcing)
- [x] 7.3 Define CH4-related variables
- [x] 7.4 Define N2O-related variables
- [x] 7.5 Define temperature variables (surface, ocean heat uptake)
- [x] 7.6 Document variable naming conventions

## 8. Migrate Existing Components

- [ ] 8.1 Update TestComponent to use registered variables (N/A - example component with different naming)
- [x] 8.2 Update CO2ERF component
- [x] 8.3 Update CarbonCycle component (partial - uses VAR_CO2_CONCENTRATION)
- [ ] 8.4 Update FourBoxOceanHeatUptake component (N/A - uses different variable names)
- [x] 8.5 Verify all Rust tests pass after migration

## 9. Python Bindings

- [x] 9.1 Export `TimeConvention` enum to Python
- [x] 9.2 Export `PreindustrialValue` enum to Python (with convenience methods)
- [x] 9.3 Export `VariableDefinition` class (constructible from Python)
- [x] 9.4 Implement `register_variable()` function
- [x] 9.5 Implement `get_variable()` function
- [x] 9.6 Implement `list_variables()` function
- [ ] 9.7 Update timeseries creation methods to accept preindustrial
- [ ] 9.8 Update type stubs (.pyi files)
- [ ] 9.9 Add Python tests for registration, lookup, and preindustrial

## 10. Documentation

- [x] 10.1 Add rustdoc for all public types
- [ ] 10.2 Update CLAUDE.md with variable registration guidance
- [ ] 10.3 Add example showing Rust variable definition and usage
- [ ] 10.4 Add example showing Python variable registration
- [ ] 10.5 Add example showing preindustrial value usage
