use crate::errors::RSCMResult;
pub use crate::state::{
    FourBoxSlice, GridTimeseriesWindow, HemisphericSlice, InputState, OutputState, TimeseriesWindow,
};
use crate::timeseries::Time;
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Type of requirement (input, output, state, or internal link)
///
/// # Variants
///
/// - `Input`: Read-only variable from external source or other component
/// - `Output`: Write-only variable produced by this component each timestep
/// - `State`: Variable that reads its previous value and writes a new value each timestep.
///   State variables require an initial value to be provided at model build time.
/// - `EmptyLink`: Internal graph connectivity (not for user code)
#[pyclass(eq, eq_int)]
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub enum RequirementType {
    /// Read from external source or other component
    Input,
    /// Write new value each timestep
    Output,
    /// Read previous value and write new value (requires initial value)
    State,
    /// Internal graph connectivity (not for user code)
    EmptyLink,
}

#[pymethods]
impl RequirementType {
    /// Get the name of this requirement type variant
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            RequirementType::Input => "Input",
            RequirementType::Output => "Output",
            RequirementType::State => "State",
            RequirementType::EmptyLink => "EmptyLink",
        }
    }
}

/// Spatial grid type for a variable
///
/// Specifies what spatial resolution a variable operates at.
/// This enables type-safe coupling validation between components.
#[pyclass(eq, eq_int)]
#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash, Serialize, Deserialize, Default)]
pub enum GridType {
    /// Scalar (global average or non-spatial) - default for backwards compatibility
    #[default]
    Scalar,
    /// Four-box model (NorthernOcean, NorthernLand, SouthernOcean, SouthernLand)
    FourBox,
    /// Two-region hemispheric (Northern, Southern)
    Hemispheric,
}

#[pymethods]
impl GridType {
    /// Get the name of this grid type variant
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            GridType::Scalar => "Scalar",
            GridType::FourBox => "FourBox",
            GridType::Hemispheric => "Hemispheric",
        }
    }
}

impl GridType {
    /// Returns true if `self` is strictly coarser (lower resolution) than `other`.
    ///
    /// The grid hierarchy from finest to coarsest is:
    /// - FourBox (4 regions) > Hemispheric (2 regions) > Scalar (1 value)
    ///
    /// A grid is coarser if it has fewer spatial divisions. Same grids are not
    /// considered coarser than each other.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_core::component::GridType;
    ///
    /// assert!(GridType::Scalar.is_coarser_than(GridType::FourBox));
    /// assert!(GridType::Scalar.is_coarser_than(GridType::Hemispheric));
    /// assert!(GridType::Hemispheric.is_coarser_than(GridType::FourBox));
    /// assert!(!GridType::FourBox.is_coarser_than(GridType::Scalar));
    /// assert!(!GridType::Scalar.is_coarser_than(GridType::Scalar));
    /// ```
    pub fn is_coarser_than(&self, other: GridType) -> bool {
        match (self, other) {
            // Scalar is coarser than everything except itself
            (GridType::Scalar, GridType::FourBox) => true,
            (GridType::Scalar, GridType::Hemispheric) => true,
            // Hemispheric is coarser than FourBox only
            (GridType::Hemispheric, GridType::FourBox) => true,
            // All other cases: same grid or finer
            _ => false,
        }
    }

    /// Returns true if `self` can be aggregated to `target`.
    ///
    /// Aggregation transforms a finer grid to a coarser grid via weighted averaging.
    /// This is valid when `target` is coarser than or equal to `self`.
    ///
    /// Returns false if `target` is finer than `self` (disaggregation/broadcast),
    /// as that would require inventing spatial structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_core::component::GridType;
    ///
    /// // Aggregation from fine to coarse is valid
    /// assert!(GridType::FourBox.can_aggregate_to(GridType::Scalar));
    /// assert!(GridType::FourBox.can_aggregate_to(GridType::Hemispheric));
    /// assert!(GridType::Hemispheric.can_aggregate_to(GridType::Scalar));
    ///
    /// // Same grid is valid (no-op)
    /// assert!(GridType::FourBox.can_aggregate_to(GridType::FourBox));
    /// assert!(GridType::Scalar.can_aggregate_to(GridType::Scalar));
    ///
    /// // Disaggregation (coarse to fine) is NOT valid
    /// assert!(!GridType::Scalar.can_aggregate_to(GridType::FourBox));
    /// assert!(!GridType::Scalar.can_aggregate_to(GridType::Hemispheric));
    /// assert!(!GridType::Hemispheric.can_aggregate_to(GridType::FourBox));
    /// ```
    pub fn can_aggregate_to(&self, target: GridType) -> bool {
        // Can aggregate if target is coarser or same as self
        // i.e., NOT if target is finer (which would be target.is_coarser_than(self) == false
        // but we need target NOT coarser... wait, let me think through this more carefully)
        //
        // If target == self: valid (no-op)
        // If target is coarser than self: valid (aggregation)
        // If self is coarser than target: invalid (disaggregation)
        *self == target || target.is_coarser_than(*self)
    }
}

impl std::fmt::Display for GridType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GridType::Scalar => write!(f, "Scalar"),
            GridType::FourBox => write!(f, "FourBox"),
            GridType::Hemispheric => write!(f, "Hemispheric"),
        }
    }
}

/// Definition of a component's input or output requirement
///
/// Each requirement specifies:
/// - `name`: Variable identifier (e.g., "Atmospheric Concentration|CO2")
/// - `unit`: Physical units (e.g., "ppm", "W / m^2")
/// - `requirement_type`: Whether this is an input, output, or both
/// - `grid_type`: Spatial resolution (Scalar, FourBox, Hemispheric)
#[pyclass]
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub struct RequirementDefinition {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub unit: String,
    #[pyo3(get, set)]
    pub requirement_type: RequirementType,
    #[pyo3(get, set)]
    pub grid_type: GridType,
}

impl RequirementDefinition {
    /// Create a new scalar requirement (default, backwards compatible)
    pub fn new(name: &str, unit: &str, requirement_type: RequirementType) -> Self {
        Self {
            name: name.to_string(),
            unit: unit.to_string(),
            requirement_type,
            grid_type: GridType::Scalar,
        }
    }

    /// Create a new requirement with an explicit grid type
    pub fn with_grid(
        name: &str,
        unit: &str,
        requirement_type: RequirementType,
        grid_type: GridType,
    ) -> Self {
        Self {
            name: name.to_string(),
            unit: unit.to_string(),
            requirement_type,
            grid_type,
        }
    }

    /// Create a scalar input requirement
    pub fn scalar_input(name: &str, unit: &str) -> Self {
        Self::new(name, unit, RequirementType::Input)
    }

    /// Create a scalar output requirement
    pub fn scalar_output(name: &str, unit: &str) -> Self {
        Self::new(name, unit, RequirementType::Output)
    }

    /// Create a scalar state requirement (reads previous, writes new value)
    pub fn scalar_state(name: &str, unit: &str) -> Self {
        Self::new(name, unit, RequirementType::State)
    }

    /// Create a four-box input requirement
    pub fn four_box_input(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, RequirementType::Input, GridType::FourBox)
    }

    /// Create a four-box output requirement
    pub fn four_box_output(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, RequirementType::Output, GridType::FourBox)
    }

    /// Create a four-box state requirement (reads previous, writes new value)
    pub fn four_box_state(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, RequirementType::State, GridType::FourBox)
    }

    /// Create a hemispheric input requirement
    pub fn hemispheric_input(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, RequirementType::Input, GridType::Hemispheric)
    }

    /// Create a hemispheric output requirement
    pub fn hemispheric_output(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, RequirementType::Output, GridType::Hemispheric)
    }

    /// Create a hemispheric state requirement (reads previous, writes new value)
    pub fn hemispheric_state(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, RequirementType::State, GridType::Hemispheric)
    }

    /// Check if this requirement is spatially resolved (non-scalar)
    pub fn is_spatial(&self) -> bool {
        self.grid_type != GridType::Scalar
    }
}

/// Metadata about a variable (input, output, or state)
///
/// Used for documentation generation and introspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableMetadata {
    /// Rust field name (e.g., "concentration_co2")
    pub rust_name: String,
    /// Variable name in the model (e.g., "Atmospheric Concentration|CO2")
    pub variable_name: String,
    /// Physical units (e.g., "ppm")
    pub unit: String,
    /// Spatial grid type
    pub grid: GridType,
    /// Description from doc comments (if available)
    pub description: String,
}

/// Metadata about a component for documentation generation
///
/// This struct contains all the information needed to generate
/// documentation pages for a component, including I/O definitions,
/// tags, and categorisation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component struct name (e.g., "CarbonCycle")
    pub name: String,
    /// Tags for filtering (e.g., ["carbon-cycle", "simple", "stable"])
    pub tags: Vec<String>,
    /// Primary category for grouping (e.g., "Carbon Cycle")
    pub category: Option<String>,
    /// Input variable definitions
    pub inputs: Vec<VariableMetadata>,
    /// Output variable definitions
    pub outputs: Vec<VariableMetadata>,
    /// State variable definitions
    pub states: Vec<VariableMetadata>,
}

/// Component of a reduced complexity climate model
///
/// Each component encapsulates some set of physics that can be solved for a given time step.
/// Generally these components can be modelled as a set of Ordinary Differential Equations (ODEs)
/// with an input state that can be solved as an initial value problem over a given time domain.
///
/// The resulting state of a component can then be used by other components as part of a `Model`
/// or solved alone during calibration.
///
/// Each component contains:
/// * parameters: Time invariant constants used to parameterize the components physics
/// * inputs: State information required to solve the model. This come from either other
///   components as part of a coupled system or from exogenous data.
/// * outputs: Information that is solved by the component
///
/// Structs implementing the `Component` trait should be serializable and deserializable
/// and use the `#[typetag::serde]` macro when implementing the trait to enable
/// serialisation/deserialisation when using `Component` as an object trait
/// (i.e. where `dyn Component` is used; see `models.rs`).
#[typetag::serde(tag = "type")]
pub trait Component: Debug + Send + Sync {
    fn definitions(&self) -> Vec<RequirementDefinition>;

    /// Variables that are required to solve this component
    ///
    /// Returns all `Input` and `State` requirements.
    fn inputs(&self) -> Vec<RequirementDefinition> {
        self.definitions()
            .iter()
            .filter(|d| {
                matches!(
                    d.requirement_type,
                    RequirementType::Input | RequirementType::State
                )
            })
            .cloned()
            .collect()
    }
    fn input_names(&self) -> Vec<String> {
        self.inputs().into_iter().map(|d| d.name).collect()
    }

    /// Variables that are solved by this component
    ///
    /// The names of the solved variables must be unique for a given model.
    /// i.e. No two components within a model can produce the same variable names.
    /// These names can contain '|' to namespace variables to avoid collisions,
    /// for example, 'Emissions|CO2' and 'Atmospheric Concentrations|CO2'
    ///
    /// Returns all `Output` and `State` requirements.
    fn outputs(&self) -> Vec<RequirementDefinition> {
        self.definitions()
            .iter()
            .filter(|d| {
                matches!(
                    d.requirement_type,
                    RequirementType::Output | RequirementType::State
                )
            })
            .cloned()
            .collect()
    }
    fn output_names(&self) -> Vec<String> {
        self.outputs().into_iter().map(|d| d.name).collect()
    }

    /// Solve the component until `t_next`
    ///
    /// The result should contain values for the current time step for all output variables
    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example_components::{TestComponent, TestComponentParameters};
    use crate::timeseries::Timeseries;
    use crate::timeseries_collection::{TimeseriesItem, VariableType};
    use ndarray::array;

    #[test]
    fn solve() {
        let component = TestComponent::from_parameters(TestComponentParameters {
            conversion_factor: 2.0,
        });

        let emissions_co2 = TimeseriesItem {
            data: crate::timeseries_collection::TimeseriesData::Scalar(Timeseries::from_values(
                array![1.1, 1.3],
                array![2020.0, 2021.0],
            )),
            name: "Emissions|CO2".to_string(),
            variable_type: VariableType::Exogenous,
        };

        // current_time=2020.0 corresponds to index 0 in the timeseries
        let input_state = InputState::build(vec![&emissions_co2], 2020.0);

        // current() returns the value at the index corresponding to current_time (index 0)
        assert_eq!(
            input_state.get_scalar_window("Emissions|CO2").at_start(),
            1.1
        );

        let output_state = component.solve(2020.0, 2021.0, &input_state).unwrap();
        assert_eq!(
            output_state.get("Concentrations|CO2").unwrap(),
            &crate::state::StateValue::Scalar(1.1 * 2.0)
        );
    }

    #[test]
    fn test_requirement_definition_new_is_scalar() {
        let req = RequirementDefinition::new("Emissions|CO2", "GtC / yr", RequirementType::Input);
        assert_eq!(req.grid_type, GridType::Scalar);
        assert!(!req.is_spatial());
    }

    #[test]
    fn test_requirement_definition_with_grid() {
        let req = RequirementDefinition::with_grid(
            "Temperature",
            "K",
            RequirementType::Output,
            GridType::FourBox,
        );
        assert_eq!(req.grid_type, GridType::FourBox);
        assert!(req.is_spatial());
    }

    #[test]
    fn test_requirement_definition_convenience_constructors() {
        let scalar_in = RequirementDefinition::scalar_input("Emissions|CO2", "GtC / yr");
        assert_eq!(scalar_in.grid_type, GridType::Scalar);
        assert_eq!(scalar_in.requirement_type, RequirementType::Input);

        let scalar_out = RequirementDefinition::scalar_output("Concentrations|CO2", "ppm");
        assert_eq!(scalar_out.grid_type, GridType::Scalar);
        assert_eq!(scalar_out.requirement_type, RequirementType::Output);

        let four_box_in = RequirementDefinition::four_box_input("Temperature", "K");
        assert_eq!(four_box_in.grid_type, GridType::FourBox);
        assert_eq!(four_box_in.requirement_type, RequirementType::Input);

        let four_box_out = RequirementDefinition::four_box_output("Surface Temperature", "K");
        assert_eq!(four_box_out.grid_type, GridType::FourBox);
        assert_eq!(four_box_out.requirement_type, RequirementType::Output);

        let hemi_in = RequirementDefinition::hemispheric_input("Precipitation", "mm / yr");
        assert_eq!(hemi_in.grid_type, GridType::Hemispheric);
        assert_eq!(hemi_in.requirement_type, RequirementType::Input);
    }

    #[test]
    fn test_grid_type_display() {
        assert_eq!(format!("{}", GridType::Scalar), "Scalar");
        assert_eq!(format!("{}", GridType::FourBox), "FourBox");
        assert_eq!(format!("{}", GridType::Hemispheric), "Hemispheric");
    }

    #[test]
    fn test_grid_type_default() {
        let default: GridType = Default::default();
        assert_eq!(default, GridType::Scalar);
    }

    #[test]
    fn test_is_coarser_than_scalar_coarser_than_all_finer_grids() {
        // Scalar is coarser than FourBox and Hemispheric
        assert!(GridType::Scalar.is_coarser_than(GridType::FourBox));
        assert!(GridType::Scalar.is_coarser_than(GridType::Hemispheric));
    }

    #[test]
    fn test_is_coarser_than_hemispheric_coarser_than_fourbox() {
        // Hemispheric (2 regions) is coarser than FourBox (4 regions)
        assert!(GridType::Hemispheric.is_coarser_than(GridType::FourBox));
    }

    #[test]
    fn test_is_coarser_than_fourbox_not_coarser_than_anything() {
        // FourBox is the finest grid, not coarser than anything
        assert!(!GridType::FourBox.is_coarser_than(GridType::Scalar));
        assert!(!GridType::FourBox.is_coarser_than(GridType::Hemispheric));
        assert!(!GridType::FourBox.is_coarser_than(GridType::FourBox));
    }

    #[test]
    fn test_is_coarser_than_same_grid_not_coarser() {
        // Same grids are not considered coarser than each other
        assert!(!GridType::Scalar.is_coarser_than(GridType::Scalar));
        assert!(!GridType::Hemispheric.is_coarser_than(GridType::Hemispheric));
        assert!(!GridType::FourBox.is_coarser_than(GridType::FourBox));
    }

    #[test]
    fn test_is_coarser_than_hemispheric_not_coarser_than_scalar() {
        // Hemispheric is finer than Scalar, so not coarser
        assert!(!GridType::Hemispheric.is_coarser_than(GridType::Scalar));
    }

    #[test]
    fn test_can_aggregate_to_same_grid_always_valid() {
        // Aggregation to same grid is always valid (no-op)
        assert!(GridType::Scalar.can_aggregate_to(GridType::Scalar));
        assert!(GridType::Hemispheric.can_aggregate_to(GridType::Hemispheric));
        assert!(GridType::FourBox.can_aggregate_to(GridType::FourBox));
    }

    #[test]
    fn test_can_aggregate_to_coarser_valid() {
        // Aggregation from finer to coarser is valid
        assert!(GridType::FourBox.can_aggregate_to(GridType::Scalar));
        assert!(GridType::FourBox.can_aggregate_to(GridType::Hemispheric));
        assert!(GridType::Hemispheric.can_aggregate_to(GridType::Scalar));
    }

    #[test]
    fn test_can_aggregate_to_finer_invalid_disaggregation() {
        // Disaggregation (coarse to fine) is NOT valid
        assert!(!GridType::Scalar.can_aggregate_to(GridType::FourBox));
        assert!(!GridType::Scalar.can_aggregate_to(GridType::Hemispheric));
        assert!(!GridType::Hemispheric.can_aggregate_to(GridType::FourBox));
    }
}
