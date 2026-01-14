# Design: Grid-Based Timeseries Architecture

## Context

RSCM needs to support spatially-resolved timeseries data for climate modelling. MAGICC uses a four-box regional structure, but the design should be extensible to arbitrary spatial grids. Components may operate at different spatial resolutions and need to exchange data through grid transformations.

**Constraints:**

- Must be computationally efficient (climate models run many timesteps)
- Must maintain backwards compatibility with existing scalar timeseries
- Must support serialization (JSON/TOML)
- Must work with existing Component trait and state management
- Grid should be a first-class concept to enable future extensions

**Stakeholders:**

- Climate scientists implementing MAGICC-like models
- Component developers who may need regional data
- Users who need to couple components at different spatial resolutions

## Goals / Non-Goals

**Goals:**

- Support MAGICC standard four-box grid (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
- Enable grid transformations (aggregation, disaggregation) for component coupling
- Make grid a first-class, extensible concept
- Zero-cost abstractions where possible (compile-time grid knowledge)
- Maintain ergonomic API similar to existing Timeseries

**Non-Goals:**

- Full 2D/3D gridded data (lat/lon) - focus on regional box models for now
- Automatic spatial interpolation between arbitrary grids (only predefined aggregations)
- Runtime-configurable grid structures (use compile-time types for efficiency)

## Decisions

### Decision 1: Generic GridTimeseries with SpatialGrid Trait

**Approach:** Create a `GridTimeseries<T, G>` type that is generic over both the value type `T` and a grid type `G: SpatialGrid`. The grid trait defines the spatial structure and transformation operations.

```rust
pub trait SpatialGrid: Clone + Debug + Serialize + DeserializeOwned {
    /// Unique name for this grid type (e.g., "FourBox", "Scalar")
    fn grid_name(&self) -> &'static str;

    /// Number of spatial regions in this grid
    fn size(&self) -> usize;

    /// Names of regions (e.g., ["Northern Ocean", "Northern Land", ...])
    fn region_names(&self) -> &[String];

    /// Aggregate all regions to a single global value
    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue;

    /// Transform values from this grid to another grid type
    /// Returns error if transformation is not explicitly defined
    fn transform_to<G: SpatialGrid>(&self, values: &[FloatValue], target: &G) -> RSCMResult<Vec<FloatValue>>;
}

pub struct GridTimeseries<T, G>
where
    T: Float,
    G: SpatialGrid,
{
    grid: G,
    // Values stored as Array2<T>: shape (time, space)
    values: Array2<T>,
    time_axis: Arc<TimeAxis>,
    units: String,
    interpolation_strategy: InterpolationStrategy,
    latest: usize,
}
```

**Rationale:**

- Grid is a first-class type parameter, enabling compile-time optimization
- Can support arbitrary grid types beyond four-box (scalar, hemispheric, custom)
- Grid transformation is explicit via the trait, making coupling clear
- Backwards compatible: `Timeseries<T>` can be aliased to `GridTimeseries<T, ScalarGrid>`

**Alternatives considered:**

1. **Composition (four separate Timeseries objects):** Simple but loses spatial relationships, harder to implement aggregation efficiently, more complex API
2. **Multi-dimensional Timeseries (ndarray internally):** More general but less type-safe, grid structure implicit, harder to optimize
3. **Separate FourBoxTimeseries type:** Simpler but not extensible, duplicates code for each grid type

### Decision 2: Standard Grid Implementations

Provide three built-in grid types:

1. **ScalarGrid:** Single global value (backwards compatibility with existing Timeseries)
2. **FourBoxGrid:** MAGICC standard (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
3. **HemisphericGrid:** Simple Northern/Southern split

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarGrid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FourBoxGrid {
    // Weights for aggregation to global (e.g., area fractions)
    weights: [FloatValue; 4],
}

impl FourBoxGrid {
    pub fn magicc_standard() -> Self {
        Self {
            // Equal weighting for now (can be refined with actual area fractions)
            weights: [0.25, 0.25, 0.25, 0.25],
        }
    }

    pub const NORTHERN_OCEAN: usize = 0;
    pub const NORTHERN_LAND: usize = 1;
    pub const SOUTHERN_OCEAN: usize = 2;
    pub const SOUTHERN_LAND: usize = 3;
}
```

**Rationale:**

- Covers common use cases (scalar for backwards compat, four-box for MAGICC, hemispheric as middle ground)
- Weights enable physical-based aggregation (e.g., area-weighted means)
- Constants provide clear, named access to regions

### Decision 3: Grid Transformation Strategy

Grid transformations are explicit and bidirectional where defined:

```rust
impl SpatialGrid for FourBoxGrid {
    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue {
        // Weighted average using grid weights
        values.iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    fn transform_to<G: SpatialGrid>(&self, values: &[FloatValue], target: &G) -> RSCMResult<Vec<FloatValue>> {
        // Implement transformations for known target types
        // Error on unsupported transformations - be explicit about grid coupling
        match TypeId::of::<G>() {
            id if id == TypeId::of::<ScalarGrid>() => {
                Ok(vec![self.aggregate_global(values)])
            }
            id if id == TypeId::of::<HemisphericGrid>() => {
                Ok(vec![
                    (values[0] * self.weights[0] + values[1] * self.weights[1])
                        / (self.weights[0] + self.weights[1]), // Northern
                    (values[2] * self.weights[2] + values[3] * self.weights[3])
                        / (self.weights[2] + self.weights[3]), // Southern
                ])
            }
            _ => {
                // Error on unsupported transformation
                Err(RSCMError::UnsupportedGridTransformation {
                    from: self.grid_name(),
                    to: target.grid_name(),
                })
            }
        }
    }
}
```

**Rationale:**

- Explicit transformations prevent silent data loss
- Type-based dispatch enables efficient, known transformations
- Erroring on unsupported transformations ensures developers explicitly define all grid coupling
- Component developers must consciously handle spatial resolution changes
- Forces proper design of grid transformation logic rather than hiding issues

**Alternatives considered:**

1. **Automatic interpolation:** Complex, requires spatial metadata, computationally expensive
2. **Fallback to global broadcast:** Silently loses spatial information, can hide bugs (REJECTED)
3. **Only support identical grids:** Too restrictive for component coupling

### Decision 4: State System Integration

Update `InputState` and `OutputState` to support grid-aware values:

```rust
// Option A: Type-erased approach (runtime flexibility)
pub enum StateValue {
    Scalar(FloatValue),
    Grid(Box<dyn GridValues>),
}

// Option B: Generic approach (compile-time safety)
pub struct InputState<'a> {
    scalar_state: Vec<&'a TimeseriesItem>,
    grid_state: Vec<&'a GridTimeseriesItem>,
}

// DECISION: Start with Option A (type-erased) for flexibility
// Components can query the spatial structure at runtime
// Can optimize to Option B later if performance critical
```

**Rationale:**

- Type-erased approach allows heterogeneous grids in the same model
- Components can check grid type at runtime and transform as needed
- More flexible for experimentation and future extensions
- Can optimize later with benchmarks if performance issues arise

## Risks / Trade-offs

### Risk: Performance overhead from dynamic dispatch

- **Mitigation:** Use monomorphization where possible, benchmark critical paths, provide compile-time grid type option if needed

### Risk: Complex API for component developers

- **Mitigation:** Provide helper methods for common operations (get_global, get_region), clear documentation with examples, scalar API unchanged

### Risk: Grid transformation information loss

- **Mitigation:** Explicit transformation methods make loss visible, logging/warnings for fallback transformations, document transformation semantics

### Trade-off: Type safety vs. flexibility

- **Decision:** Favor flexibility initially (type-erased state) since use cases are still being explored
- Can add typed variants later for performance-critical components

## Migration Plan

### Phase 1: Core types (this proposal)

1. Implement `SpatialGrid` trait and standard grid types (Scalar, FourBox, Hemispheric)
2. Implement `GridTimeseries<T, G>` with full API (set, get, interpolate, transform)
3. Add serialization support
4. Type alias: `type Timeseries<T> = GridTimeseries<T, ScalarGrid>` (backwards compatible)

### Phase 2: State system integration (this proposal)

1. Update `InputState`/`OutputState` to handle grid values (type-erased approach)
2. Update `Component` trait examples to show grid usage patterns
3. Add helper methods for common grid operations

### Phase 3: Component adoption (future)

1. Create example four-box components (e.g., four-box ocean heat uptake)
2. Migrate existing components that would benefit from spatial structure
3. Document coupling patterns between components with different grids

### Rollback Strategy

- If design proves problematic, can deprecate `GridTimeseries` and keep original `Timeseries`
- Alias approach means no breaking changes to existing code
- New components using grids would need refactoring, but limited since feature is new

## Open Questions

1. **Exact weights for four-box aggregation:** Should we use actual ocean/land area fractions?
   - **Answer:** Start with equal weights (0.25 each), make configurable, document how to set physical weights

2. **Support for time-varying grids:** Should grid structure be allowed to change over time?
   - **Answer:** No for now (adds significant complexity), can revisit if needed

3. **Python API design:** How should grid timeseries be exposed to Python (numpy arrays, scmdata)?
   - **Answer:** Defer to Python binding implementation, likely 2D numpy arrays with grid metadata

4. **Named region access:** Should we provide named accessors (e.g., `.northern_ocean()`) or indices?
   - **Answer:** Both - constants for indices, helper methods for named access if ergonomic

## Component Integration Patterns

### Pattern 1: Scalar Component (Backwards Compatible)

Existing components that work with global values continue unchanged:

```rust
#[typetag::serde]
impl Component for CO2ERFComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new("Atmospheric Concentration|CO2", "ppm", RequirementType::Input),
            RequirementDefinition::new("ERF|CO2", "W/m²", RequirementType::Output),
        ]
    }

    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        // Works with scalar Timeseries (alias for GridTimeseries<T, ScalarGrid>)
        let co2 = input_state.get_latest("Atmospheric Concentration|CO2");
        let erf = calculate_erf(co2, self.parameters);

        let mut output = HashMap::new();
        output.insert("ERF|CO2".to_string(), erf);
        Ok(output)
    }
}
```

**Key points:**

- No changes needed for components that use scalar values
- `Timeseries<T>` is now an alias for `GridTimeseries<T, ScalarGrid>`
- Aggregation happens automatically if fed grid data

### Pattern 2: Grid-Native Component

Components that naturally operate at regional resolution:

```rust
#[typetag::serde]
impl Component for FourBoxOceanHeatUptakeComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new(
                "ERF|Aggregated",
                "W/m²",
                RequirementType::Input
            ),
            RequirementDefinition::new(
                "Ocean Heat Uptake",
                "W/m²",
                RequirementType::Output
            ).with_grid_type(GridType::FourBox), // Metadata about expected grid
        ]
    }

    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        // Get grid value - returns GridTimeseries<FloatValue, G>
        let erf = input_state.get_grid::<FourBoxGrid>("ERF|Aggregated")?;

        // Access regional values
        let erf_values = erf.at_time_values(t_current)?; // Returns [FloatValue; 4]

        // Compute regional heat uptake
        let mut heat_uptake = [0.0; 4];
        for i in 0..4 {
            heat_uptake[i] = self.compute_regional_uptake(erf_values[i], i, t_current);
        }

        // Return grid output
        let mut output = HashMap::new();
        output.insert_grid("Ocean Heat Uptake".to_string(), heat_uptake, FourBoxGrid::magicc_standard());
        Ok(output)
    }
}
```

**Key points:**

- Explicitly works with grid data using `InputState::get_grid<G>()`
- Returns grid values via `OutputState::insert_grid()`
- Can use grid-specific indexing (constants like `FourBoxGrid::NORTHERN_OCEAN`)

### Pattern 3: Mixed Grid Component (Coupling Different Resolutions)

Component that accepts scalar input but produces grid output (disaggregation):

```rust
#[typetag::serde]
impl Component for GlobalToRegionalDisaggregatorComponent {
    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        // Get scalar input
        let global_erf = input_state.get_latest("ERF|Global");

        // Disaggregate using predetermined fractions
        let regional_erf = [
            global_erf * self.northern_ocean_fraction,
            global_erf * self.northern_land_fraction,
            global_erf * self.southern_ocean_fraction,
            global_erf * self.southern_land_fraction,
        ];

        // Output as four-box grid
        let mut output = HashMap::new();
        output.insert_grid("ERF|Regional".to_string(), regional_erf, FourBoxGrid::magicc_standard());
        Ok(output)
    }
}
```

### Pattern 4: Grid Transformation in Component Coupling

When components at different resolutions need to communicate:

```rust
// Scenario: Four-box ocean component feeds into hemispheric atmosphere component

// Ocean component outputs four-box heat uptake
let ocean_output = four_box_ocean.solve(t_current, t_next, &state)?;
// Returns: GridTimeseries<FloatValue, FourBoxGrid>

// Atmosphere component needs hemispheric input
// Model builder handles transformation automatically:
model_builder
    .add_component(four_box_ocean_component)
    .add_component(hemispheric_atmosphere_component)
    .with_transformation("Ocean Heat Uptake", HemisphericGrid::new())?;

// Or manual transformation in component:
let heat_uptake_four_box = input_state.get_grid::<FourBoxGrid>("Ocean Heat Uptake")?;
let heat_uptake_hemispheric = heat_uptake_four_box.transform_to(HemisphericGrid::new())?;
```

**Key points:**

- Transformations can happen automatically in model builder (if configured)
- Or explicitly in component code for more control
- Unsupported transformations error at runtime with clear messages
- Component developers see exactly where resolution changes

### Pattern 5: Aggregate-then-Compute vs. Compute-then-Aggregate

Two approaches for handling grid coupling:

**Approach A: Aggregate input, compute scalar**

```rust
// Get four-box input, aggregate to scalar, compute
let regional_temp = input_state.get_grid::<FourBoxGrid>("Surface Temperature")?;
let global_temp = regional_temp.aggregate_global_at_time(t_current)?;
let response = self.compute_scalar_response(global_temp);
```

**Approach B: Compute regional, aggregate output**

```rust
// Get scalar input, compute regionally, return grid
let global_forcing = input_state.get_latest("Forcing");
let regional_response = [
    self.compute_regional_response(global_forcing, FourBoxGrid::NORTHERN_OCEAN),
    self.compute_regional_response(global_forcing, FourBoxGrid::NORTHERN_LAND),
    self.compute_regional_response(global_forcing, FourBoxGrid::SOUTHERN_OCEAN),
    self.compute_regional_response(global_forcing, FourBoxGrid::SOUTHERN_LAND),
];
```

**Which to use:**

- Use A when the physics is fundamentally global (e.g., global mean feedback)
- Use B when regional differences matter (e.g., land-ocean heat capacity differences)

## Supported Regions and Transformations

### Standard Grid Types

#### 1. ScalarGrid (Global)

- **Regions:** 1 (Global)
- **Use case:** Backwards compatibility, components that work with global means
- **Region names:** `["Global"]`
- **Typical variables:** Global mean temperature, global ERF, total emissions

#### 2. FourBoxGrid (MAGICC Standard)

- **Regions:** 4 (hemispheric ocean-land split)
- **Use case:** MAGICC-equivalent models, basic spatial resolution
- **Region indices:**
  - `FourBoxGrid::NORTHERN_OCEAN` (0)
  - `FourBoxGrid::NORTHERN_LAND` (1)
  - `FourBoxGrid::SOUTHERN_OCEAN` (2)
  - `FourBoxGrid::SOUTHERN_LAND` (3)
- **Region names:** `["Northern Ocean", "Northern Land", "Southern Ocean", "Southern Land"]`
- **Typical variables:** Regional temperatures, regional heat uptake, regional ERF
- **Default weights:** Equal (0.25 each), configurable with `FourBoxGrid::with_weights([w1, w2, w3, w4])`
- **Physical weights (example):** Based on actual surface area fractions
  - Northern Ocean: 0.25 (assuming 50% of NH is ocean)
  - Northern Land: 0.25
  - Southern Ocean: 0.40 (Southern Hemisphere is more ocean-dominated)
  - Southern Land: 0.10

#### 3. HemisphericGrid

- **Regions:** 2 (north-south split)
- **Use case:** Intermediate resolution, simple latitudinal gradients
- **Region indices:**
  - `HemisphericGrid::NORTHERN` (0)
  - `HemisphericGrid::SOUTHERN` (1)
- **Region names:** `["Northern Hemisphere", "Southern Hemisphere"]`
- **Typical variables:** Hemispheric temperature, hemispheric forcing
- **Default weights:** Equal (0.5 each), configurable

### Transformation Matrix

Table showing which transformations are explicitly supported:

| From \ To         | ScalarGrid | HemisphericGrid | FourBoxGrid |
|-------------------|------------|-----------------|-------------|
| **ScalarGrid**    | Identity   | Broadcast*      | Broadcast*  |
| **HemisphericGrid**| Aggregate  | Identity        | ERROR**     |
| **FourBoxGrid**   | Aggregate  | Aggregate       | Identity    |

**Legend:**

- **Identity:** No transformation needed (same grid type)
- **Aggregate:** Weighted average of regions → coarser resolution
- **Broadcast:** Copy scalar value to all regions (information duplication)
- **ERROR:** No physically meaningful transformation defined

**Notes:**

- `*` Broadcast transformations: Use with caution. The scalar value is copied to all regions, which may not be physically accurate. Only use when the variable is truly uniform spatially.
- `**` Hemispheric → FourBox: Not supported because we cannot infer ocean/land split from hemisphere data. Component must explicitly disaggregate if needed.

### Detailed Transformation Semantics

#### FourBox → Scalar (Aggregation)

```rust
global_value =
    northern_ocean * weight_NO +
    northern_land * weight_NL +
    southern_ocean * weight_SO +
    southern_land * weight_SL
```

where weights sum to 1.0 (typically based on area fractions)

#### FourBox → Hemispheric (Aggregation)

```rust
northern_hemisphere = (northern_ocean * weight_NO + northern_land * weight_NL)
                    / (weight_NO + weight_NL)
southern_hemisphere = (southern_ocean * weight_SO + southern_land * weight_SL)
                    / (weight_SO + weight_SL)
```

#### Hemispheric → Scalar (Aggregation)

```rust
global_value = northern * weight_N + southern * weight_S
```

where weights sum to 1.0 (typically 0.5 each unless using actual hemisphere areas)

#### Scalar → FourBox (Broadcast)

```rust
[global_value, global_value, global_value, global_value]
```

**Warning:** This assumes the variable is spatially uniform. Use only for:

- Atmospheric CO₂ concentrations (well-mixed)
- Global forcing agents that apply uniformly
- Initialization values before spatial patterns develop

**Do NOT use for:**

- Temperature (strong latitudinal gradients)
- Regional emissions (spatially heterogeneous by definition)
- Ocean properties (land-ocean differences)

#### Scalar → Hemispheric (Broadcast)

```rust
[global_value, global_value]
```

Same caveats as Scalar → FourBox

### Adding Custom Transformations

For unsupported transformations (e.g., Hemispheric → FourBox), components must implement explicit logic:

```rust
impl Component for MyDisaggregatorComponent {
    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        // Get hemispheric input
        let hemispheric = input_state.get_grid::<HemisphericGrid>("Temperature")?;
        let [northern, southern] = hemispheric.at_time_values(t_current)?;

        // Custom disaggregation logic based on physics
        // Example: Use ocean/land fraction and empirical temperature patterns
        let four_box = [
            northern * 1.1,  // Northern Ocean (slightly warmer than mean)
            northern * 0.9,  // Northern Land (slightly cooler)
            southern * 1.05, // Southern Ocean
            southern * 0.85, // Southern Land (much cooler in SH)
        ];

        let mut output = HashMap::new();
        output.insert_grid("Temperature|FourBox".to_string(), four_box, FourBoxGrid::magicc_standard());
        Ok(output)
    }
}
```

This makes disaggregation explicit and documented, rather than hidden in automatic transformations.

### Transformation Error Messages

When attempting an unsupported transformation, users see:

```
Error: Unsupported grid transformation
  From: HemisphericGrid (2 regions)
  To: FourBoxGrid (4 regions)

  This transformation is not defined because spatial disaggregation from
  hemispheric to ocean/land boxes requires additional physical assumptions.

  To resolve:
  1. Create a custom component that explicitly disaggregates based on your model's physics
  2. Use an intermediate transformation (e.g., Hemispheric → Scalar → FourBox with broadcast)
  3. Consider if your component should work at a different resolution

  See docs: https://lewisjared.github.io/rscm/grid-transformations.html
```

This guides users toward explicit, physically-motivated solutions.
