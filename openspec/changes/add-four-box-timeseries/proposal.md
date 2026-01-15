# Change: Add Four-Box Timeseries with Grid Support

## Why

MAGICC uses a four-box regional structure (Northern Ocean, Northern Land, Southern Ocean, Southern Land) to represent spatially-resolved climate variables. RSCM currently only supports scalar timeseries (time dimension only), which limits its ability to represent regional variations and couple components that operate at different spatial resolutions.

To support MAGICC-like models and enable more sophisticated climate component coupling, RSCM needs first-class support for spatially-structured timeseries data with efficient grid transformations.

## What Changes

- Add a `SpatialGrid` trait and implementations for common grid structures (scalar, four-box, arbitrary regions)
- Add `GridTimeseries<T, G>` type that extends timeseries with spatial dimensions using a grid type `G`
- Implement four-box grid (`FourBoxGrid`) for MAGICC-standard regions (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
- Add grid transformation/aggregation capabilities for coupling components at different spatial resolutions
- Update `InputState` and `OutputState` to support both scalar and grid-based values
- Add serialization support for grid-based timeseries (JSON/TOML)
- Ensure computational efficiency through zero-cost abstractions where possible

## Impact

### Affected Specs

- New capability: `grid-timeseries` (spatial timeseries support)

### Affected Code

- `rscm-core/src/timeseries.rs` - Add GridTimeseries type
- `rscm-core/src/spatial.rs` - New module for SpatialGrid trait and implementations
- `rscm-core/src/state.rs` - Update InputState/OutputState for grid data
- `rscm-core/src/component.rs` - May need to support grid-aware requirement definitions
- `rscm-core/src/lib.rs` - Export new spatial module
- Python bindings in `rscm/src/python/` - Expose grid timeseries to Python
- Tests throughout for grid operations

### Migration Strategy

- Fully backwards compatible: existing scalar timeseries remain unchanged
- Scalar values are a special case of grid (single-point grid)
- Components can gradually adopt grid support without breaking existing code
