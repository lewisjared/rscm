# Tasks: Variable Registration System

## 1. Core Types

- [ ] 1.1 Add `inventory` crate to Cargo.toml
- [ ] 1.2 Create `rscm-core/src/variable.rs` module
- [ ] 1.3 Implement `TimeConvention` enum (StartOfYear, MidYear, Instantaneous)
- [ ] 1.4 Implement `VariableDefinition` struct (name, unit, time_convention, description)
- [ ] 1.5 Add unit tests for core types

## 2. Preindustrial Value (Timeseries Metadata)

- [ ] 2.1 Implement `PreindustrialValue` enum (Scalar, FourBox, Hemispheric)
- [ ] 2.2 Implement convenience methods (as_scalar, as_four_box, to_scalar)
- [ ] 2.3 Add `preindustrial: Option<PreindustrialValue>` field to TimeseriesItem
- [ ] 2.4 Add `get_preindustrial()` and `get_preindustrial_scalar()` methods to InputState
- [ ] 2.5 Update TimeseriesCollection methods to accept optional preindustrial
- [ ] 2.6 Add unit tests for preindustrial functionality

## 3. Variable Registry

- [ ] 3.1 Implement `VariableRegistry` struct with static + runtime storage
- [ ] 3.2 Implement `get()` method for variable lookup
- [ ] 3.3 Implement `register()` method for runtime registration (with duplicate check)
- [ ] 3.4 Implement `list()` method for enumeration
- [ ] 3.5 Create global registry instance (lazy_static or OnceCell)
- [ ] 3.6 Add unit tests for registry operations

## 4. Static Registration (Rust)

- [ ] 4.1 Create `define_variable!` macro using inventory
- [ ] 4.2 Implement inventory collection for static variables
- [ ] 4.3 Wire static variables into global registry at startup
- [ ] 4.4 Add unit tests for macro and static registration

## 5. Refactor RequirementDefinition

- [ ] 5.1 Rename `name` field to `variable_name` (keep `unit` field - components declare expected units)
- [ ] 5.2 Keep existing constructors (API unchanged: `scalar_input(name, unit)`, etc.)
- [ ] 5.3 Add `time_convention()` accessor method (registry lookup - intrinsic to variable)
- [ ] 5.4 Update serialization to use `variable_name` field name
- [ ] 5.5 Add unit tests for RequirementDefinition

## 6. Model Builder Validation

- [ ] 6.1 Add variable existence validation (check registry)
- [ ] 6.2 Add unit compatibility validation (compare between connected components, not registry)
- [ ] 6.3 Add time convention compatibility validation (from registry, error on mismatch)
- [ ] 6.4 Create RSCMError variants: UnregisteredVariableError, UnitMismatchError, TimeConventionMismatchError
- [ ] 6.5 Add integration tests for validation scenarios

## 7. Standard Variable Definitions

- [ ] 7.1 Create `rscm-core/src/variables/` directory structure
- [ ] 7.2 Define CO2-related variables (concentration, emissions, forcing)
- [ ] 7.3 Define CH4-related variables
- [ ] 7.4 Define N2O-related variables
- [ ] 7.5 Define temperature variables (surface, ocean layers)
- [ ] 7.6 Document variable naming conventions

## 8. Migrate Existing Components

- [ ] 8.1 Update TestComponent to use registered variables
- [ ] 8.2 Update CO2ERF component
- [ ] 8.3 Update CarbonCycle component
- [ ] 8.4 Update FourBoxOceanHeatUptake component
- [ ] 8.5 Verify all Rust tests pass after migration

## 9. Python Bindings

- [ ] 9.1 Export `TimeConvention` enum to Python
- [ ] 9.2 Export `PreindustrialValue` enum to Python (with convenience methods)
- [ ] 9.3 Export `VariableDefinition` class (constructible from Python)
- [ ] 9.4 Implement `register_variable()` function
- [ ] 9.5 Implement `get_variable()` function
- [ ] 9.6 Implement `list_variables()` function
- [ ] 9.7 Update timeseries creation methods to accept preindustrial
- [ ] 9.8 Update type stubs (.pyi files)
- [ ] 9.9 Add Python tests for registration, lookup, and preindustrial

## 10. Documentation

- [ ] 10.1 Add rustdoc for all public types
- [ ] 10.2 Update CLAUDE.md with variable registration guidance
- [ ] 10.3 Add example showing Rust variable definition and usage
- [ ] 10.4 Add example showing Python variable registration
- [ ] 10.5 Add example showing preindustrial value usage
