# Grid Timeseries Capability

## ADDED Requirements

### Requirement: SpatialGrid Trait

The system SHALL provide a trait `SpatialGrid` that defines the spatial structure and transformation operations for grid-based timeseries.

#### Scenario: Grid defines spatial regions

- **WHEN** a grid type implements `SpatialGrid`
- **THEN** it MUST specify the number of spatial regions
- **AND** provide names for each region
- **AND** provide a unique grid type name
- **AND** support aggregation of all regions to a single global value
- **AND** define explicit transformations to supported grid types (returning Result)
- **AND** error on transformations to unsupported grid types

#### Scenario: Standard four-box grid

- **WHEN** using `FourBoxGrid::magicc_standard()`
- **THEN** the grid MUST have exactly 4 regions
- **AND** regions MUST be named "Northern Ocean", "Northern Land", "Southern Ocean", "Southern Land"
- **AND** regions MUST be accessible via constants (NORTHERN_OCEAN, NORTHERN_LAND, SOUTHERN_OCEAN, SOUTHERN_LAND)

### Requirement: GridTimeseries Type

The system SHALL provide a `GridTimeseries<T, G>` type that represents time-varying values over a spatial grid.

#### Scenario: Create grid timeseries from values

- **WHEN** creating a `GridTimeseries` with values, time axis, and grid
- **THEN** the values MUST have shape (time_steps, grid_regions)
- **AND** the number of columns MUST equal `grid.size()`
- **AND** the number of rows MUST equal `time_axis.len()`

#### Scenario: Access values at specific time and region

- **WHEN** calling `grid_timeseries.at(time_index, region_index)`
- **THEN** it MUST return the value at the specified time and spatial region
- **AND** return None if indices are out of bounds

#### Scenario: Interpolate to specific time across all regions

- **WHEN** calling `grid_timeseries.at_time(time)`
- **THEN** it MUST interpolate values for all spatial regions at the given time
- **AND** return an array with length equal to `grid.size()`
- **AND** use the configured interpolation strategy

### Requirement: Grid Aggregation

The system SHALL support aggregating grid-based timeseries to global values.

#### Scenario: Aggregate four-box to global timeseries

- **WHEN** calling `grid_timeseries.aggregate_global()`
- **THEN** it MUST return a scalar timeseries (GridTimeseries<T, ScalarGrid>)
- **AND** each timestep MUST be aggregated using `grid.aggregate_global()`
- **AND** the aggregation MUST use grid-specific weights (e.g., area fractions)

#### Scenario: Get global value at specific time

- **WHEN** calling `grid_timeseries.at_time_global(time)`
- **THEN** it MUST interpolate to the specified time
- **AND** aggregate all regions to a single global value

### Requirement: Grid Transformation

The system SHALL support transforming timeseries between different grid types.

#### Scenario: Transform four-box to scalar grid

- **WHEN** calling `four_box_timeseries.transform_to(ScalarGrid)`
- **THEN** it MUST aggregate all four regions to a single global value for each timestep
- **AND** return a `GridTimeseries<T, ScalarGrid>`

#### Scenario: Transform four-box to hemispheric grid

- **WHEN** calling `four_box_timeseries.transform_to(HemisphericGrid::new())`
- **THEN** it MUST aggregate Northern Ocean and Northern Land to Northern Hemisphere
- **AND** aggregate Southern Ocean and Southern Land to Southern Hemisphere
- **AND** use appropriate weights for aggregation
- **AND** return a `GridTimeseries<T, HemisphericGrid>`

#### Scenario: Error on unsupported grid transformation

- **WHEN** transforming to a grid type without a defined transformation
- **THEN** it MUST return an error indicating the unsupported transformation
- **AND** the error MUST include both source and target grid type names
- **AND** the transformation MUST NOT silently aggregate or broadcast

### Requirement: Region Access

The system SHALL provide convenient access to individual regional timeseries.

#### Scenario: Extract single region as scalar timeseries

- **WHEN** calling `grid_timeseries.region(region_index)`
- **THEN** it MUST return a `GridTimeseries<T, ScalarGrid>` for that region only
- **AND** the returned timeseries MUST have the same time axis
- **AND** values MUST match the specified region's values at all timesteps

#### Scenario: Set value for specific region and time

- **WHEN** calling `grid_timeseries.set(time_index, region_index, value)`
- **THEN** it MUST update the value at the specified time and region
- **AND** NOT affect values at other times or regions
- **AND** update the latest valid timestep if applicable

### Requirement: Serialization Support

The system SHALL support serialization and deserialization of grid timeseries to JSON and TOML formats.

#### Scenario: Serialize four-box timeseries to JSON

- **WHEN** serializing a `GridTimeseries<f64, FourBoxGrid>` to JSON
- **THEN** the JSON MUST include grid metadata (type, region names, weights)
- **AND** include values as a 2D array (time x space)
- **AND** include time axis information
- **AND** include units and interpolation strategy

#### Scenario: Deserialize four-box timeseries from JSON

- **WHEN** deserializing JSON to `GridTimeseries<f64, FourBoxGrid>`
- **THEN** it MUST reconstruct the grid with correct region names and weights
- **AND** reconstruct values with correct shape
- **AND** reconstruct time axis, units, and interpolation strategy
- **AND** handle NaN values correctly in the data

### Requirement: Backwards Compatibility

The system SHALL maintain backwards compatibility with existing scalar `Timeseries` usage.

#### Scenario: Type alias for scalar timeseries

- **WHEN** using `Timeseries<T>`
- **THEN** it MUST be equivalent to `GridTimeseries<T, ScalarGrid>`
- **AND** all existing Timeseries methods MUST work unchanged

#### Scenario: Existing component code unchanged

- **WHEN** a component uses `Timeseries<FloatValue>` in state
- **THEN** it MUST continue to work without modification
- **AND** no changes to component implementations required unless adopting grid support

### Requirement: Standard Grid Implementations

The system SHALL provide standard grid implementations for common use cases.

#### Scenario: Scalar grid for single global values

- **WHEN** using `ScalarGrid`
- **THEN** it MUST have exactly 1 region
- **AND** region MUST be named "Global"
- **AND** aggregation MUST return the single value unchanged

#### Scenario: Hemispheric grid for north/south split

- **WHEN** using `HemisphericGrid`
- **THEN** it MUST have exactly 2 regions
- **AND** regions MUST be named "Northern Hemisphere" and "Southern Hemisphere"
- **AND** aggregation MUST use hemisphere-specific weights

#### Scenario: Four-box grid weights are configurable

- **WHEN** creating a `FourBoxGrid` with custom weights
- **THEN** the weights MUST be used for aggregation operations
- **AND** weights SHOULD represent physical quantities (e.g., area fractions)
- **AND** weights SHOULD sum to 1.0 for proper averaging

### Requirement: Interpolation Strategy Support

The system SHALL support different interpolation strategies for grid timeseries.

#### Scenario: Linear interpolation across time for all regions

- **WHEN** using `LinearSplineStrategy` with grid timeseries
- **THEN** interpolation MUST be applied independently to each spatial region
- **AND** all regions MUST use the same interpolation strategy

#### Scenario: Previous value strategy for all regions

- **WHEN** using `PreviousStrategy` with grid timeseries
- **THEN** each region MUST return its most recent non-NaN value
- **AND** extrapolation behavior MUST be controlled by the strategy configuration

### Requirement: Grid Timeseries Collection Integration

The system SHALL support grid timeseries in the TimeseriesCollection.

#### Scenario: Add grid timeseries to collection

- **WHEN** adding a `GridTimeseries` to `TimeseriesCollection`
- **THEN** it MUST be stored with grid metadata
- **AND** retrievable by name
- **AND** distinguishable from scalar timeseries

#### Scenario: Component state with mixed scalar and grid timeseries

- **WHEN** a component's `InputState` contains both scalar and grid timeseries
- **THEN** the component MUST be able to query the grid type
- **AND** access scalar values as before for backwards compatibility
- **AND** access grid values with region information
