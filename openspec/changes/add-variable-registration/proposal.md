# Change: Add Variable Registration System

## Why

MAGICC modules use a DATASTORE pattern where variables carry rich metadata: units, time conventions (start-of-year vs mid-year), preindustrial references, and descriptions. Currently, RSCM components declare inputs/outputs as ad-hoc strings with no validation. This makes it impossible to:

1. Validate that connected components use compatible units and time conventions
2. Automatically handle unit conversions between components
3. Discover what variables a model produces or consumes
4. Set preindustrial reference values required by forcing calculations

## What Changes

- **ADDED** `VariableDefinition` struct with comprehensive metadata (units, time convention, description, preindustrial value)
- **ADDED** `VariableRegistry` for registering and looking up variable definitions
- **ADDED** `TimeConvention` enum (StartOfYear, MidYear, Instantaneous) for temporal alignment
- **ADDED** Validation during model building to check unit and time convention compatibility
- **ADDED** Introspection API for discovering registered variables
- **BREAKING** `RequirementDefinition` now requires a registered variable reference (removes ad-hoc string-based declarations)

## Impact

- Affected specs: variable-registration (new capability)
- Affected code:
  - `rscm-core/src/variable.rs` (new)
  - `rscm-core/src/component.rs` (refactored)
  - `rscm-core/src/model.rs` (validation added)
  - All existing components must migrate to use registered variables
