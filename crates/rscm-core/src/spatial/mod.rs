//! Spatial grid types for representing spatially-resolved climate data
//!
//! This module provides the [`SpatialGrid`] trait and implementations for common grid structures
//! used in climate models, including:
//!
//! - [`ScalarGrid`]: Single global value (backwards compatible with scalar timeseries)
//! - [`FourBoxGrid`]: MAGICC standard four-box structure (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
//! - [`HemisphericGrid`]: Simple Northern/Southern hemisphere split
//!
//! # Grid Types and Use Cases
//!
//! ## ScalarGrid
//!
//! Single global value. Use for:
//! - Backwards compatibility with existing scalar timeseries
//! - Variables that are truly spatially uniform (e.g., well-mixed atmospheric CO₂)
//! - Components that work with global means
//!
//! ## FourBoxGrid
//!
//! MAGICC standard four-box structure with regions:
//! - Northern Ocean ([`FourBoxRegion::NorthernOcean`])
//! - Northern Land ([`FourBoxRegion::NorthernLand`])
//! - Southern Ocean ([`FourBoxRegion::SouthernOcean`])
//! - Southern Land ([`FourBoxRegion::SouthernLand`])
//!
//! Use for:
//! - MAGICC-equivalent models
//! - Basic spatial resolution capturing ocean-land and north-south differences
//! - Variables with different dynamics in ocean vs. land regions
//!
//! ## HemisphericGrid
//!
//! Simple north-south split with regions:
//! - Northern Hemisphere ([`HemisphericRegion::Northern`])
//! - Southern Hemisphere ([`HemisphericRegion::Southern`])
//!
//! Use for:
//! - Intermediate spatial resolution
//! - Variables with strong latitudinal gradients
//! - When ocean-land distinction is less important
//!
//! # Grid Transformations
//!
//! Grid transformations allow components at different spatial resolutions to communicate.
//! The following transformation matrix shows which transformations are supported:
//!
//! | From \ To         | ScalarGrid | HemisphericGrid | FourBoxGrid |
//! |-------------------|------------|-----------------|-------------|
//! | **ScalarGrid**    | Identity   | Broadcast*      | Broadcast*  |
//! | **HemisphericGrid**| Aggregate  | Identity        | ERROR**     |
//! | **FourBoxGrid**   | Aggregate  | Aggregate       | Identity    |
//!
//! **Legend:**
//! - **Identity**: No transformation needed (same grid type)
//! - **Aggregate**: Weighted average of regions -> coarser resolution
//! - **Broadcast**: Copy scalar value to all regions (use with caution!)
//! - **ERROR**: No physically meaningful transformation defined
//!
//! ## Aggregation Transformations
//!
//! Aggregation uses weighted averages based on grid weights (typically area fractions):
//!
//! ### FourBox -> Scalar
//!
//! ```text
//! global = NO*w_NO + NL*w_NL + SO*w_SO + SL*w_SL
//! ```
//!
//! where weights sum to 1.0.
//!
//! ### FourBox -> Hemispheric
//!
//! ```text
//! northern = (NO*w_NO + NL*w_NL) / (w_NO + w_NL)
//! southern = (SO*w_SO + SL*w_SL) / (w_SO + w_SL)
//! ```
//!
//! ### Hemispheric -> Scalar
//!
//! ```text
//! global = northern*w_N + southern*w_S
//! ```
//!
//! ## Broadcast Transformations
//!
//! Broadcast transformations copy a scalar value to all regions. **Use with extreme caution!**
//!
//! ### Safe Uses
//!
//! - Well-mixed atmospheric gases (CO₂, CH₄) that are spatially uniform
//! - Global forcing agents that apply uniformly
//! - Initialisation values before spatial patterns develop
//!
//! ### Unsafe Uses (DO NOT USE)
//!
//! - Temperature (strong latitudinal gradients)
//! - Regional emissions (spatially heterogeneous by definition)
//! - Ocean properties (land-ocean differences)
//! - Any variable with known spatial structure
//!
//! ## Unsupported Transformations
//!
//! Some transformations (e.g., Hemispheric -> FourBox) are not supported because they
//! require spatial disaggregation that cannot be determined uniquely. For example,
//! splitting hemispheric data into ocean/land boxes requires assumptions about
//! land-ocean temperature differences.
//!
//! If you need such a transformation, implement a custom component that explicitly
//! performs the disaggregation based on your model's physics (see Custom Disaggregation below).
//!
//! # Component Integration Patterns
//!
//! ## Pattern 1: Scalar Component (Backwards Compatible)
//!
//! Existing components that work with global values continue unchanged:
//!
//! ```rust,ignore
//! # use rscm_core::component::{Component, RequirementDefinition, RequirementType, InputState, OutputState};
//! # use rscm_core::timeseries::Time;
//! # use rscm_core::errors::RSCMResult;
//! # use std::collections::HashMap;
//! # #[derive(Debug)]
//! # struct CO2ERFComponent;
//! #[typetag::serde]
//! impl Component for CO2ERFComponent {
//!     fn definitions(&self) -> Vec<RequirementDefinition> {
//!         vec![
//!             RequirementDefinition::new("Atmospheric Concentration|CO2", "ppm", RequirementType::Input),
//!             RequirementDefinition::new("ERF|CO2", "W/m^2", RequirementType::Output),
//!         ]
//!     }
//!
//!     fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
//!         // Works with scalar values - InputState handles grid aggregation automatically
//!         let co2 = input_state.get_scalar_window("Atmospheric Concentration|CO2").at_start();
//!         // ... compute ERF ...
//!         let mut output = HashMap::new();
//!         output.insert("ERF|CO2".to_string(), StateValue::Scalar(co2 * 5.35_f64.ln()));
//!         Ok(output)
//!     }
//! }
//! ```
//!
//! ## Pattern 2: Grid-Native Component
//!
//! Components that naturally operate at regional resolution can access grid data directly
//! using the typed window API:
//!
//! ```rust,ignore
//! # use rscm_core::component::{Component, InputState, OutputState};
//! # use rscm_core::timeseries::Time;
//! # use rscm_core::errors::RSCMResult;
//! # use rscm_core::state::{StateValue, FourBoxSlice};
//! # use std::collections::HashMap;
//! # use rscm_core::spatial::FourBoxGrid;
//! fn solve_grid_component(input_state: &InputState, t_current: Time) -> RSCMResult<OutputState> {
//!     // Get grid values using the typed window API
//!     let window = input_state.get_four_box_window("ERF|FourBox");
//!     let erf_regions = window.all();
//!
//!     // Compute regional heat uptake
//!     let heat_uptake: Vec<f64> = erf_regions.iter()
//!         .map(|erf| erf * 0.9) // Example: 90% absorbed by ocean
//!         .collect();
//!
//!     // Return FourBox output with regional values
//!     let mut output = HashMap::new();
//!     output.insert(
//!         "Ocean Heat Uptake|FourBox".to_string(),
//!         StateValue::FourBox(FourBoxSlice::from_array([
//!             heat_uptake[0], heat_uptake[1], heat_uptake[2], heat_uptake[3]
//!         ])),
//!     );
//!     Ok(output)
//! }
//! ```
//!
//! ## Pattern 3: Aggregate-then-Compute vs. Compute-then-Aggregate
//!
//! When handling grid data, you can either:
//!
//! **Approach A: Aggregate input, compute scalar**
//!
//! ```rust
//! # use rscm_core::component::InputState;
//! # fn example(input_state: &InputState, t_current: f64) {
//! // Get aggregated global value automatically
//! let global_temp = input_state.get_global("Surface Temperature").unwrap();
//! let response = global_temp * 0.5; // Compute from global mean
//! # }
//! ```
//!
//! **Approach B: Compute regional, aggregate output**
//!
//! ```rust,ignore
//! # use rscm_core::component::InputState;
//! # use rscm_core::spatial::{FourBoxGrid, SpatialGrid};
//! # fn example(input_state: &InputState) {
//! // Get grid values, compute regionally, then aggregate
//! let window = input_state.get_four_box_window("Surface Temperature|FourBox");
//! let temps = window.all();
//! let regional_responses: Vec<f64> = temps.iter()
//!     .map(|t| t * 0.5) // Different response per region
//!     .collect();
//! let grid = FourBoxGrid::magicc_standard();
//! let global_response = grid.aggregate_global(&regional_responses);
//! # }
//! ```
//!
//! **Which to use:**
//! - Use A when the physics is fundamentally global (e.g., global mean feedback)
//! - Use B when regional differences matter (e.g., land-ocean heat capacity differences)
//!
//! # Custom Disaggregation Components
//!
//! For unsupported transformations (e.g., Hemispheric -> FourBox), implement a custom
//! disaggregation component that explicitly encodes the physics:
//!
//! ```rust,ignore
//! # use rscm_core::component::{Component, RequirementDefinition, RequirementType, InputState, OutputState, GridType};
//! # use rscm_core::timeseries::Time;
//! # use rscm_core::errors::RSCMResult;
//! # use rscm_core::state::{StateValue, FourBoxSlice};
//! # use std::collections::HashMap;
//! # #[derive(Debug)]
//! # struct HemisphericToFourBoxDisaggregator { ocean_land_ratio: f64 }
//! #[typetag::serde]
//! impl Component for HemisphericToFourBoxDisaggregator {
//!     fn definitions(&self) -> Vec<RequirementDefinition> {
//!         vec![
//!             RequirementDefinition::with_grid("Temperature|Hemispheric", "degC", RequirementType::Input, GridType::Hemispheric),
//!             RequirementDefinition::with_grid("Temperature|FourBox", "degC", RequirementType::Output, GridType::FourBox),
//!         ]
//!     }
//!
//!     fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
//!         // Get hemispheric input using typed window API
//!         let window = input_state.get_hemispheric_window("Temperature|Hemispheric");
//!         let [northern, southern] = window.all();
//!
//!         // Custom disaggregation based on physical reasoning
//!         // Example: Ocean regions slightly warmer, land regions cooler
//!         let four_box_values = FourBoxSlice::from_array([
//!             northern * 1.1,  // Northern Ocean
//!             northern * 0.9,  // Northern Land
//!             southern * 1.05, // Southern Ocean
//!             southern * 0.85, // Southern Land (Southern Hemisphere land is colder)
//!         ]);
//!
//!         let mut output = HashMap::new();
//!         output.insert("Temperature|FourBox".to_string(), StateValue::FourBox(four_box_values));
//!         Ok(output)
//!     }
//! }
//! ```
//!
//! This makes the disaggregation explicit and documented rather than hidden in
//! automatic transformations.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use rscm_core::spatial::{FourBoxGrid, SpatialGrid};
//!
//! let grid = FourBoxGrid::magicc_standard();
//! assert_eq!(grid.size(), 4);
//! assert_eq!(grid.grid_name(), "FourBox");
//!
//! // Aggregate regional values to global
//! let regional_temps = vec![15.0, 14.0, 10.0, 9.0]; // degC
//! let global_temp = grid.aggregate_global(&regional_temps);
//! assert_eq!(global_temp, 12.0); // Equal weights = simple average
//! ```
//!
//! ## Grid Transformation
//!
//! ```rust
//! use rscm_core::spatial::{FourBoxGrid, HemisphericGrid, SpatialGrid};
//!
//! let four_box = FourBoxGrid::magicc_standard();
//! let hemispheric = HemisphericGrid::equal_weights();
//!
//! // Transform four-box to hemispheric
//! let four_box_temps = vec![15.0, 14.0, 10.0, 9.0]; // [NO, NL, SO, SL]
//! let hemispheric_temps = four_box.transform_to(&four_box_temps, &hemispheric).unwrap();
//!
//! assert_eq!(hemispheric_temps.len(), 2);
//! assert_eq!(hemispheric_temps[0], 14.5); // (15.0 + 14.0) / 2
//! assert_eq!(hemispheric_temps[1], 9.5);  // (10.0 + 9.0) / 2
//! ```

use crate::errors::RSCMResult;
use crate::timeseries::FloatValue;

pub mod four_box;
pub mod hemispheric;
pub mod scalar;

pub use four_box::{FourBoxGrid, FourBoxRegion};
pub use hemispheric::{HemisphericGrid, HemisphericRegion};
pub use scalar::{ScalarGrid, ScalarRegion};

/// Trait for spatial grid structures used in climate models
///
/// A spatial grid defines how climate variables are discretized spatially.
/// For example, a four-box grid divides the world into Northern Ocean, Northern Land,
/// Southern Ocean, and Southern Land regions.
///
/// The trait provides methods for:
/// - Querying grid structure (size, region names)
/// - Aggregating regional values to global values
/// - Transforming between different grid types
///
/// Note: This trait is not object-safe due to the generic `transform_to` method.
/// Use the concrete grid types directly rather than trait objects.
pub trait SpatialGrid: Clone + std::fmt::Debug + Send + Sync {
    /// Unique name for this grid type
    ///
    /// Used for error messages and debugging
    fn grid_name(&self) -> &'static str;

    /// Number of spatial regions in this grid
    fn size(&self) -> usize;

    /// Names of regions in this grid
    ///
    /// For example, a four-box grid returns:
    /// `["Northern Ocean", "Northern Land", "Southern Ocean", "Southern Land"]`
    fn region_names(&self) -> &[String];

    /// Aggregate all regional values to a single global value
    ///
    /// Uses grid-specific weights (typically area fractions) to compute
    /// a weighted average of all regions.
    ///
    /// # Arguments
    ///
    /// * `values` - Regional values to aggregate (must have length equal to `self.size()`)
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` does not match `self.size()`
    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue;

    /// Transform values from this grid to another grid type
    ///
    /// This method performs explicit grid transformations where defined.
    /// Unsupported transformations return an error to prevent silent data loss.
    ///
    /// # Arguments
    ///
    /// * `values` - Regional values in this grid's coordinate system
    /// * `target` - Target grid to transform to
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<FloatValue>)` - Transformed values in target grid's coordinate system
    /// * `Err(RSCMError::UnsupportedGridTransformation)` - If transformation is not defined
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rscm_core::spatial::{FourBoxGrid, ScalarGrid, SpatialGrid};
    ///
    /// let four_box = FourBoxGrid::magicc_standard();
    /// let scalar = ScalarGrid;
    ///
    /// let regional = vec![15.0, 14.0, 10.0, 9.0];
    /// let global = four_box.transform_to(&regional, &scalar).unwrap();
    /// assert_eq!(global.len(), 1);
    /// assert_eq!(global[0], 12.0); // Average of regional values
    /// ```
    fn transform_to<G: SpatialGrid>(
        &self,
        values: &[FloatValue],
        target: &G,
    ) -> RSCMResult<Vec<FloatValue>>;
}
