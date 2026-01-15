#![allow(dead_code)]

use crate::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    TimeseriesWindow,
};
use crate::errors::RSCMResult;
use crate::timeseries::{FloatValue, Time};
use crate::ComponentIO;
use serde::{Deserialize, Serialize};

// ============================================================================
// TestComponent - demonstrates the ComponentIO derive macro pattern
// ============================================================================

/// Parameters for the derived test component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TestComponentParameters {
    pub conversion_factor: FloatValue,
}

/// Example component using the ComponentIO derive macro with standard variables
///
/// This demonstrates the recommended pattern for new components:
/// 1. Use `#[derive(ComponentIO)]` with struct-level attributes
/// 2. Declare inputs/outputs using registered standard variable names and units
/// 3. Use the generated `{Name}Inputs` struct with `from_input_state()`
///
/// **Important**: Variable names and units should match those in `standard_variables.rs`.
/// This ensures consistency across components and enables registry-based validation.
///
/// The macro generates:
/// - `TestComponentInputs<'a>` with `emissions_co2: TimeseriesWindow<'a>`
/// - `TestComponentOutputs` with `concentration_co2: FloatValue`
/// - `TestComponent::generated_definitions()` for the Component trait
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    emissions_co2 { name = "Emissions|CO2", unit = "GtC / yr" },
)]
#[outputs(
    concentration_co2 { name = "Atmospheric Concentration|CO2", unit = "ppm" },
)]
pub(crate) struct TestComponent {
    /// Component parameters (not marked as input/output)
    pub parameters: TestComponentParameters,
}

impl TestComponent {
    pub fn from_parameters(parameters: TestComponentParameters) -> Self {
        Self { parameters }
    }

    /// Core physics calculation - extracted for testability
    pub fn calculate_concentration(&self, emissions: FloatValue) -> FloatValue {
        emissions * self.parameters.conversion_factor
    }
}

#[typetag::serde]
impl Component for TestComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        // Use the generated typed inputs struct
        let inputs = TestComponentInputs::from_input_state(input_state);

        // Access emissions using typed window - provides current(), previous(), etc.
        let emissions = inputs.emissions_co2.current();

        // Calculate the output
        let concentration = self.calculate_concentration(emissions);

        // Create output using the generated outputs struct
        let outputs = TestComponentOutputs {
            concentration_co2: concentration,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
    use crate::model::ModelBuilder;
    use crate::spatial::ScalarGrid;
    use crate::standard_variables::{VAR_CO2_CONCENTRATION, VAR_CO2_EMISSIONS};
    use crate::timeseries::{TimeAxis, Timeseries};
    use crate::variable::TimeConvention;
    use numpy::array;
    use numpy::ndarray::Axis;
    use std::sync::Arc;

    fn create_emissions_timeseries() -> Timeseries<FloatValue> {
        let values = array![0.0, 10.0].insert_axis(Axis(1));
        Timeseries::new(
            values,
            Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
            ScalarGrid,
            VAR_CO2_EMISSIONS.unit.to_string(),
            InterpolationStrategy::from(PreviousStrategy::new(true)),
        )
    }

    #[test]
    fn test_derived_component_uses_standard_variables() {
        let component = TestComponent::from_parameters(TestComponentParameters {
            conversion_factor: 0.5,
        });

        let defs = component.definitions();
        assert_eq!(defs.len(), 2);

        // Check input uses standard CO2 emissions variable
        let input_def = &defs[0];
        assert_eq!(input_def.variable_name, VAR_CO2_EMISSIONS.name);
        assert_eq!(input_def.unit, VAR_CO2_EMISSIONS.unit);
        assert_eq!(input_def.requirement_type, RequirementType::Input);
        assert_eq!(input_def.grid_type, GridType::Scalar);
        // Time convention is available from registry
        assert_eq!(input_def.time_convention(), Some(TimeConvention::MidYear));

        // Check output uses standard CO2 concentration variable
        let output_def = &defs[1];
        assert_eq!(output_def.variable_name, VAR_CO2_CONCENTRATION.name);
        assert_eq!(output_def.unit, VAR_CO2_CONCENTRATION.unit);
        assert_eq!(output_def.requirement_type, RequirementType::Output);
        // Time convention is available from registry
        assert_eq!(
            output_def.time_convention(),
            Some(TimeConvention::StartOfYear)
        );
    }

    #[test]
    fn test_derived_component_in_model() {
        use numpy::ndarray::Array;

        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let mut model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters {
                    conversion_factor: 0.5,
                },
            )))
            .with_exogenous_variable(VAR_CO2_EMISSIONS.name, create_emissions_timeseries())
            .build()
            .unwrap();

        model.run();
        assert!(model.finished());
    }

    #[test]
    fn test_derived_component_outputs_conversion() {
        let outputs = TestComponentOutputs {
            concentration_co2: 42.5,
        };

        let state: OutputState = outputs.into();
        assert_eq!(state.get(VAR_CO2_CONCENTRATION.name), Some(&42.5));
    }

    #[test]
    fn test_model_produces_standard_variable_output() {
        use numpy::ndarray::Array;

        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let mut model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters {
                    conversion_factor: 0.5,
                },
            )))
            .with_exogenous_variable(VAR_CO2_EMISSIONS.name, create_emissions_timeseries())
            .build()
            .unwrap();

        model.run();
        assert!(model.finished());

        // Output uses standard variable name
        let ts_collection = model.timeseries();
        assert!(ts_collection
            .get_by_name(VAR_CO2_CONCENTRATION.name)
            .is_some());
    }
}
