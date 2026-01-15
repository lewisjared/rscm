# Change: Improve Component Developer Experience

## Why

Creating new RSCM components requires significant boilerplate and uses string-based state access that is error-prone and lacks IDE support. Scientists contributing new climate physics components face:

1. **Boilerplate overhead**: ~40 lines to define a simple component (struct, parameters, `from_parameters`, `#[typetag::serde]`, `definitions()`, `solve()`)
2. **String-based state access**: `input_state.get_latest("Atmospheric Concentration|CO2")` has no compile-time checking; typos cause runtime panics
3. **No historical access**: Components needing smoothing, derivatives, or moving averages cannot access past values through `InputState`
4. **Confusing InputAndOutput**: The `RequirementType::InputAndOutput` pattern is unclear and requires separate initial value handling
5. **No spatial grid declarations**: Components don't declare their grid requirements, making the coupler unable to validate or auto-transform grids
6. **Python/Rust parity gap**: Python components receive raw dicts with no type hints or validation

## What Changes

### Rust API

- Add `#[derive(ComponentIO)]` macro (in separate `rscm-macros` crate) to reduce boilerplate
- Introduce `TimeseriesWindow` type providing zero-cost access to current, historical, and interpolated values
- Add `RequirementType::State` to replace `InputAndOutput` with explicit semantics
- Add spatial grid requirements to `RequirementDefinition` (e.g., `grid = FourBox`)
- Generate typed input/output structs per component (via macro) with field access instead of string lookups
- Rewrite existing components (`CarbonCycleComponent`, `CO2ERF`) to use new patterns

### Python API

- Generate typed dataclasses for component inputs/outputs from definitions
- Expose `TimeseriesWindow` as numpy-backed views for zero-copy historical access
- Add validation with clear error messages for missing/mistyped variables

### Coupler Improvements

- Validate grid compatibility between connected components at build time
- Auto-insert grid aggregation transforms when output grid is finer than input grid
- Fail fast with clear errors when grid transformation is impossible (e.g., Scalar → FourBox)
- Enable parallel execution of independent components within same BFS level (future)

### Not In Scope

- **Async solve**: Deferred. Climate components are CPU-bound; async adds complexity (lifetimes, PyO3 integration) for marginal benefit. Parallelism better achieved via Rayon/SIMD.
- **Unit conversion**: Deferred to future work. Currently units are validated as string equality.

## Impact

### Affected Specs

- New capability: `component-dx` (component developer experience)

### Affected Code

- `rscm-core/src/component.rs` - Add `RequirementType::State`, grid requirements
- `rscm-core/src/state.rs` - Add `TimeseriesWindow` type, remove old InputState API
- `rscm-core/src/model.rs` - Add grid validation and auto-transform in coupler
- `rscm-macros/` - New crate for `#[derive(Component)]` proc-macro
- `rscm-components/src/components/*.rs` - Rewrite all existing components
- `rscm-core/src/python/component.rs` - Generate Python dataclasses from definitions
- Python package - Add base classes and validation utilities
