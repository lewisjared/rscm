use crate::errors::RSCMResult;
pub use crate::state::{
    FourBoxSlice, GridTimeseriesWindow, HemisphericSlice, HemisphericWindow, InputState,
    OutputState, ScalarWindow, TimeseriesWindow, VariableSource,
};
use crate::timeseries::Time;
use crate::units::Unit;
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;

/// How a component uses a variable it declares.
///
/// A state variable needs an initial value when the model is built.
#[pyclass(eq, eq_int)]
#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariableRole {
    /// Read each step, from prescribed data or another component.
    Input,
    /// Written each step.
    Output,
    /// Read from the previous step and written each step.
    State,
}

#[pymethods]
impl VariableRole {
    /// Get the name of this role
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            VariableRole::Input => "Input",
            VariableRole::Output => "Output",
            VariableRole::State => "State",
        }
    }
}

/// The spatial layout of a variable's values.
///
/// Coupling validation compares the grids that components declare.
#[pyclass(eq, eq_int, hash, frozen)]
#[derive(
    Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum GridType {
    /// One global value.
    #[default]
    Scalar,
    /// The four-box grid: northern ocean, northern land, southern ocean, southern land.
    FourBox,
    /// Northern and southern hemispheres.
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
    /// Whether this is the scalar grid.
    pub fn is_scalar(&self) -> bool {
        *self == GridType::Scalar
    }

    /// Returns true if `self` is strictly coarser than `other`.
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
        f.write_str(match self {
            GridType::Scalar => "scalar",
            GridType::FourBox => "four_box",
            GridType::Hemispheric => "hemispheric",
        })
    }
}

/// Whether a variable's value holds at an instant or over a step.
///
/// The producer declares it, and consumers inherit it.
/// When nobody declares it, the unit decides: see [`TimeseriesType::from_unit`].
#[pyclass(eq, eq_int)]
#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeseriesType {
    /// Holds at an instant, such as a concentration or temperature.
    Point,
    /// Holds over a whole step, such as emissions, a flux or a lifetime.
    Average,
}

#[pymethods]
impl TimeseriesType {
    /// Get the name of this timeseries type variant
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            TimeseriesType::Point => "Point",
            TimeseriesType::Average => "Average",
        }
    }
}

impl TimeseriesType {
    /// The type a unit implies: `Average` for a unit per time, such as `GtC/yr`, else `Point`.
    ///
    /// `W/m^2`, `yr` and units that do not parse give `Point`.
    pub fn from_unit(unit: &str) -> Self {
        match Unit::parse(unit) {
            Ok(parsed) if parsed.is_per_time() => TimeseriesType::Average,
            _ => TimeseriesType::Point,
        }
    }
}

impl std::fmt::Display for TimeseriesType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TimeseriesType::Point => "point",
            TimeseriesType::Average => "average",
        })
    }
}

/// A component's declaration of one variable it uses.
///
/// - `name`: the variable name, such as "Atmospheric Concentrations|CO2"
/// - `unit`: the variable's unit, such as "ppm" or "W/m^2"
/// - `role`: whether the component reads it, writes it or carries it as state
/// - `grid`: the grid the component reads or writes it on
/// - `timeseries_type`: point or average, `None` to infer it from the unit
#[pyclass]
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub struct VariableDefinition {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub unit: String,
    #[pyo3(get, set)]
    pub role: VariableRole,
    #[pyo3(get, set)]
    pub grid: GridType,
    #[pyo3(get, set)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeseries_type: Option<TimeseriesType>,
}

impl VariableDefinition {
    /// Define a scalar variable.
    pub fn new(name: &str, unit: &str, role: VariableRole) -> Self {
        Self::with_grid(name, unit, role, GridType::Scalar)
    }

    /// Define a variable on `grid`.
    pub fn with_grid(name: &str, unit: &str, role: VariableRole, grid: GridType) -> Self {
        Self {
            name: name.to_string(),
            unit: unit.to_string(),
            role,
            grid,
            timeseries_type: None,
        }
    }

    /// Declare whether the variable is a point or an average, overriding the unit default.
    pub fn with_timeseries_type(mut self, timeseries_type: TimeseriesType) -> Self {
        self.timeseries_type = Some(timeseries_type);
        self
    }

    /// The declared timeseries type, else the default from the unit.
    pub fn resolved_timeseries_type(&self) -> TimeseriesType {
        self.timeseries_type
            .unwrap_or_else(|| TimeseriesType::from_unit(&self.unit))
    }

    /// Define a scalar input.
    pub fn scalar_input(name: &str, unit: &str) -> Self {
        Self::new(name, unit, VariableRole::Input)
    }

    /// Define a scalar output.
    pub fn scalar_output(name: &str, unit: &str) -> Self {
        Self::new(name, unit, VariableRole::Output)
    }

    /// Define a scalar state variable.
    pub fn scalar_state(name: &str, unit: &str) -> Self {
        Self::new(name, unit, VariableRole::State)
    }

    /// Define a four-box input.
    pub fn four_box_input(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, VariableRole::Input, GridType::FourBox)
    }

    /// Define a four-box output.
    pub fn four_box_output(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, VariableRole::Output, GridType::FourBox)
    }

    /// Define a four-box state variable.
    pub fn four_box_state(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, VariableRole::State, GridType::FourBox)
    }

    /// Define a hemispheric input.
    pub fn hemispheric_input(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, VariableRole::Input, GridType::Hemispheric)
    }

    /// Define a hemispheric output.
    pub fn hemispheric_output(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, VariableRole::Output, GridType::Hemispheric)
    }

    /// Define a hemispheric state variable.
    pub fn hemispheric_state(name: &str, unit: &str) -> Self {
        Self::with_grid(name, unit, VariableRole::State, GridType::Hemispheric)
    }

    /// Whether the variable is on a grid finer than scalar.
    pub fn is_spatial(&self) -> bool {
        self.grid != GridType::Scalar
    }
}

/// A variable as one component declares it, for the coupling report and the generated docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredVariable {
    /// The variable name, such as "Atmospheric Concentrations|CO2".
    pub name: String,
    pub unit: String,
    #[serde(default, skip_serializing_if = "GridType::is_scalar")]
    pub grid: GridType,
    pub timeseries_type: TimeseriesType,
    /// Where the component reads the variable from, `None` for an output or outside a built model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<VariableSource>,
    /// The field that holds the variable in the generated inputs or outputs struct.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub rust_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

impl DeclaredVariable {
    /// Describe a definition, with the timeseries type resolved from the unit when undeclared.
    pub fn from_definition(definition: &VariableDefinition) -> Self {
        Self {
            name: definition.name.clone(),
            unit: definition.unit.clone(),
            grid: definition.grid,
            timeseries_type: definition.resolved_timeseries_type(),
            source: None,
            rust_name: String::new(),
            description: String::new(),
        }
    }
}

/// Metadata about a component for documentation generation
///
/// This struct contains all the information needed to generate
/// documentation pages for a component, including its variables,
/// tags, and categorisation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component struct name (e.g., "CarbonCycle")
    pub name: String,
    /// Tags for filtering (e.g., ["carbon-cycle", "simple", "stable"])
    pub tags: Vec<String>,
    /// Primary category for grouping (e.g., "Carbon Cycle")
    pub category: Option<String>,
    pub inputs: Vec<DeclaredVariable>,
    pub outputs: Vec<DeclaredVariable>,
    pub states: Vec<DeclaredVariable>,
}

/// Trait for component internal state with automatic serialisation support.
///
/// Internal state is private to a component and persists across solve() calls.
/// Unlike coupled state (VariableRole::State), internal state is not shared
/// between components.
///
/// # Implementors
///
/// - Use `()` for stateless components (most components)
/// - Define a custom struct for components that need substepping history
///
/// # Serialisation
///
/// Uses `#[typetag::serde]` for automatic type registry, enabling polymorphic
/// serialisation of `Box<dyn ComponentState>` without manual type tagging.
#[typetag::serde(tag = "state_type")]
pub trait ComponentState: Debug + Send + Sync + Any {
    /// Downcast to concrete type for component use
    fn as_any(&self) -> &dyn Any;
    /// Downcast to concrete type for mutable access
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Marker implementation for stateless components.
///
/// Most components don't need internal state and can use `()` as their state type.
#[typetag::serde(name = "unit")]
impl ComponentState for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Component of a reduced complexity climate model
///
/// Each component encapsulates some set of physics that can be solved for a given step.
/// Generally these components can be modelled as a set of Ordinary Differential Equations (ODEs)
/// with an input state that can be solved as an initial value problem over a given time domain.
///
/// The resulting state of a component can then be used by other components as part of a `Model`
/// or solved alone during calibration.
///
/// Each component contains:
/// * parameters: Time invariant constants used to parameterise the components physics
/// * inputs: State information required to solve the model. This come from either other
///   components as part of a coupled system or from prescribed data.
/// * outputs: Information that is solved by the component
///
/// Structs implementing the `Component` trait should be serialisable and deserialisable
/// and use the `#[typetag::serde]` macro when implementing the trait to enable
/// serialisation/deserialisation when using `Component` as an object trait
/// (i.e. where `dyn Component` is used; see `models.rs`).
#[typetag::serde(tag = "type")]
pub trait Component: Debug + Send + Sync {
    fn definitions(&self) -> Vec<VariableDefinition>;

    /// Variables that are required to solve this component
    ///
    /// Returns all `Input` and `State` definitions.
    fn inputs(&self) -> Vec<VariableDefinition> {
        self.definitions()
            .iter()
            .filter(|d| matches!(d.role, VariableRole::Input | VariableRole::State))
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
    /// Returns all `Output` and `State` definitions.
    fn outputs(&self) -> Vec<VariableDefinition> {
        self.definitions()
            .iter()
            .filter(|d| matches!(d.role, VariableRole::Output | VariableRole::State))
            .cloned()
            .collect()
    }
    fn output_names(&self) -> Vec<String> {
        self.outputs().into_iter().map(|d| d.name).collect()
    }

    /// Solve the component until `t_next`
    ///
    /// The result should contain values for the current step for all output variables
    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState>;

    /// Solve with internal state support.
    ///
    /// Override this method for components that need persistent internal state.
    /// The default implementation ignores the state and calls `solve()`.
    ///
    /// # Arguments
    ///
    /// * `t_current` - Start bound of the step
    /// * `t_next` - End bound of the step
    /// * `input_state` - Input state from model
    /// * `internal_state` - Mutable reference to component's internal state
    fn solve_with_state(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        // Default: ignore state, call existing solve()
        let _ = internal_state;
        self.solve(t_current, t_next, input_state)
    }

    /// Create the initial state for this component.
    ///
    /// Default returns `Box::new(())` for stateless components.
    /// Override for components that need internal state.
    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example_components::{TestComponent, TestComponentParameters};
    use crate::timeseries::Timeseries;
    use crate::timeseries_collection::TimeseriesItem;
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
        };

        // Emissions are an average, which a prescribed input reads at the step's start (index 0)
        let transforms = crate::state::TransformContext {
            read_transforms: std::collections::HashMap::from([(
                "Emissions|CO2".to_string(),
                crate::state::ReadTransformInfo::default()
                    .with_timeseries_type(TimeseriesType::Average),
            )]),
        };
        let input_state =
            InputState::build_with_transforms(vec![&emissions_co2], 2020.0, transforms);
        assert_eq!(input_state.get_scalar_window("Emissions|CO2").get(), 1.1);

        let output_state = component.solve(2020.0, 2021.0, &input_state).unwrap();
        assert_eq!(
            output_state.get("Atmospheric Concentrations|CO2").unwrap(),
            &crate::state::StateValue::Scalar(1.1 * 2.0)
        );
    }

    #[test]
    fn test_variable_definition_new_is_scalar() {
        let req = VariableDefinition::new("Emissions|CO2", "GtC/yr", VariableRole::Input);
        assert_eq!(req.grid, GridType::Scalar);
        assert!(!req.is_spatial());
    }

    #[test]
    fn timeseries_type_from_unit() {
        for unit in ["GtC / yr", "kt / yr", "Gt N / yr", "Mt CO2 / yr"] {
            assert_eq!(
                TimeseriesType::from_unit(unit),
                TimeseriesType::Average,
                "{unit}"
            );
        }
        for unit in [
            "W / m^2",
            "K",
            "ppm",
            "yr",
            "dimensionless",
            "1",
            "not a unit !",
        ] {
            assert_eq!(
                TimeseriesType::from_unit(unit),
                TimeseriesType::Point,
                "{unit}"
            );
        }
    }

    #[test]
    fn resolved_timeseries_type_prefers_declaration() {
        let lifetime = VariableDefinition::scalar_output("Lifetime|CH4", "yr");
        assert_eq!(lifetime.timeseries_type, None);
        assert_eq!(lifetime.resolved_timeseries_type(), TimeseriesType::Point);
        let lifetime = lifetime.with_timeseries_type(TimeseriesType::Average);
        assert_eq!(lifetime.resolved_timeseries_type(), TimeseriesType::Average);
    }

    #[test]
    fn test_variable_definition_with_grid() {
        let req = VariableDefinition::with_grid(
            "Temperature",
            "K",
            VariableRole::Output,
            GridType::FourBox,
        );
        assert_eq!(req.grid, GridType::FourBox);
        assert!(req.is_spatial());
    }

    #[test]
    fn test_variable_definition_convenience_constructors() {
        let scalar_in = VariableDefinition::scalar_input("Emissions|CO2", "GtC/yr");
        assert_eq!(scalar_in.grid, GridType::Scalar);
        assert_eq!(scalar_in.role, VariableRole::Input);

        let scalar_out = VariableDefinition::scalar_output("Atmospheric Concentrations|CO2", "ppm");
        assert_eq!(scalar_out.grid, GridType::Scalar);
        assert_eq!(scalar_out.role, VariableRole::Output);

        let four_box_in = VariableDefinition::four_box_input("Temperature", "K");
        assert_eq!(four_box_in.grid, GridType::FourBox);
        assert_eq!(four_box_in.role, VariableRole::Input);

        let four_box_out =
            VariableDefinition::four_box_output("Surface Air Temperature Change", "K");
        assert_eq!(four_box_out.grid, GridType::FourBox);
        assert_eq!(four_box_out.role, VariableRole::Output);

        let hemi_in = VariableDefinition::hemispheric_input("Precipitation", "mm/yr");
        assert_eq!(hemi_in.grid, GridType::Hemispheric);
        assert_eq!(hemi_in.role, VariableRole::Input);
    }

    #[test]
    fn test_grid_display() {
        assert_eq!(format!("{}", GridType::Scalar), "scalar");
        assert_eq!(format!("{}", GridType::FourBox), "four_box");
        assert_eq!(format!("{}", GridType::Hemispheric), "hemispheric");
    }

    #[test]
    fn test_grid_default() {
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
    fn test_is_coarser_than_hemispheric_coarser_than_four_box() {
        // Hemispheric (2 regions) is coarser than FourBox (4 regions)
        assert!(GridType::Hemispheric.is_coarser_than(GridType::FourBox));
    }

    #[test]
    fn test_is_coarser_than_four_box_not_coarser_than_anything() {
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

    #[test]
    fn test_component_state_unit_type_serialisation() {
        // Test that the unit type () can be serialised as ComponentState
        let state: Box<dyn ComponentState> = Box::new(());
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("\"state_type\":\"unit\""));

        // Test deserialisation
        let deserialised: Box<dyn ComponentState> = serde_json::from_str(&json).unwrap();
        assert!(deserialised.as_any().is::<()>());
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct TestComponentState {
        value: f64,
    }

    #[typetag::serde]
    impl ComponentState for TestComponentState {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_component_state_custom_type_serialisation() {
        // Test that custom state types can be serialised
        let state: Box<dyn ComponentState> = Box::new(TestComponentState { value: 42.0 });
        let json = serde_json::to_string(&state).unwrap();

        // Deserialize and verify
        let deserialised: Box<dyn ComponentState> = serde_json::from_str(&json).unwrap();
        let concrete = deserialised
            .as_any()
            .downcast_ref::<TestComponentState>()
            .unwrap();
        assert_eq!(concrete.value, 42.0);
    }
}
