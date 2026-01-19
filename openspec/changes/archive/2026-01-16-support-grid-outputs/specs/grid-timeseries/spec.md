# grid-timeseries Specification

## MODIFIED Requirements

### Requirement: GridTimeseries Type

The system SHALL provide a `GridTimeseries<T, G>` type that represents time-varying values over a spatial grid.

**Modifications from original spec:**

- Add `set_all(&mut self, time_index: usize, values: &[T])` method to set all regional values at a time index
- Add `set_from_slice(&mut self, time_index: usize, slice: &FourBoxSlice)` for FourBox grids
- Add `set_from_slice(&mut self, time_index: usize, slice: &HemisphericSlice)` for Hemispheric grids
- These methods enable components to write grid outputs without explicit per-region iteration

#### Scenario: Set all regional values at a time index

- **WHEN** calling `grid_timeseries.set_all(time_index, values)`
- **THEN** it MUST set values for all regions at the specified time
- **AND** update the `latest` index if the time_index is newer
- **AND** panic if `values.len()` does not equal `grid.size()`

#### Scenario: Set FourBox values from slice

- **WHEN** calling `grid_timeseries.set_from_slice(time_index, slice)` with a `FourBoxSlice`
- **THEN** it MUST set all four region values from the slice at the specified time

#### Scenario: Set Hemispheric values from slice

- **WHEN** calling `grid_timeseries.set_from_slice(time_index, slice)` with a `HemisphericSlice`
- **THEN** it MUST set both hemisphere values from the slice at the specified time
