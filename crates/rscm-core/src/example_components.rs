use crate::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    TimeseriesWindow,
};
use crate::errors::RSCMResult;
use crate::state::StateValue;
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

/// Example component using the ComponentIO derive macro
///
/// This demonstrates the recommended pattern for new components:
/// 1. Use `#[derive(ComponentIO)]` with struct-level attributes
/// 2. Declare inputs with `#[inputs(field { name = "...", unit = "..." })]`
/// 3. Declare outputs with `#[outputs(field { name = "...", unit = "..." })]`
/// 4. Use the generated `{Name}Inputs` struct with `from_input_state()`
///
/// The macro generates:
/// - `TestComponentInputs<'a>` with `emissions_co2: TimeseriesWindow<'a>`
/// - `TestComponentOutputs` with `concentration_co2: FloatValue`
/// - `TestComponent::generated_definitions()` for the Component trait
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    emissions_co2 { name = "Emissions|CO2", unit = "GtCO2" },
)]
#[outputs(
    concentration_co2 { name = "Concentrations|CO2", unit = "ppm" },
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

// ============================================================================
// TaggedTestComponent - demonstrates the #[component()] attribute
// ============================================================================

/// Example component with tags and category
///
/// This demonstrates the `#[component()]` attribute for documentation generation.
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["test", "example", "simple"], category = "Testing")]
#[inputs(
    input_var { name = "Test Input", unit = "units" },
)]
#[outputs(
    output_var { name = "Test Output", unit = "units" },
)]
pub(crate) struct TaggedTestComponent {
    pub multiplier: FloatValue,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
    use crate::model::ModelBuilder;
    use crate::spatial::ScalarGrid;
    use crate::timeseries::{TimeAxis, Timeseries};
    use numpy::array;
    use numpy::ndarray::Axis;
    use std::sync::Arc;

    fn create_emissions_timeseries() -> Timeseries<FloatValue> {
        let values = array![0.0, 10.0].insert_axis(Axis(1));
        Timeseries::new(
            values,
            Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
            ScalarGrid,
            "GtCO2".to_string(),
            InterpolationStrategy::from(PreviousStrategy::new(true)),
        )
    }

    #[test]
    fn test_derived_component_definitions() {
        let component = TestComponent::from_parameters(TestComponentParameters {
            conversion_factor: 0.5,
        });

        let defs = component.definitions();
        assert_eq!(defs.len(), 2);

        // Check input definition
        let input_def = &defs[0];
        assert_eq!(input_def.name, "Emissions|CO2");
        assert_eq!(input_def.unit, "GtCO2");
        assert_eq!(input_def.requirement_type, RequirementType::Input);
        assert_eq!(input_def.grid_type, GridType::Scalar);

        // Check output definition
        let output_def = &defs[1];
        assert_eq!(output_def.name, "Concentrations|CO2");
        assert_eq!(output_def.unit, "ppm");
        assert_eq!(output_def.requirement_type, RequirementType::Output);
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
            .with_exogenous_variable("Emissions|CO2", create_emissions_timeseries())
            .build()
            .unwrap();

        model.run();
        assert!(model.finished());
    }

    #[test]
    fn test_derived_component_outputs_conversion() {
        use crate::state::StateValue;

        let outputs = TestComponentOutputs {
            concentration_co2: 42.5,
        };

        let state: OutputState = outputs.into();
        assert_eq!(
            state.get("Concentrations|CO2"),
            Some(&StateValue::Scalar(42.5))
        );
    }

    #[test]
    fn test_model_with_multiple_derived_components() {
        use numpy::ndarray::Array;

        // Create a model with the derived component
        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let mut model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters {
                    conversion_factor: 0.5,
                },
            )))
            .with_exogenous_variable("Emissions|CO2", create_emissions_timeseries())
            .build()
            .unwrap();

        model.run();
        assert!(model.finished());

        // Verify the output timeseries was produced
        let ts_collection = model.timeseries();
        assert!(ts_collection.get_by_name("Concentrations|CO2").is_some());
    }

    #[test]
    fn test_component_metadata_without_tags() {
        let metadata = TestComponent::component_metadata();

        assert_eq!(metadata.name, "TestComponent");
        assert!(metadata.tags.is_empty());
        assert!(metadata.category.is_none());
        assert_eq!(metadata.inputs.len(), 1);
        assert_eq!(metadata.outputs.len(), 1);
        assert!(metadata.states.is_empty());

        // Check input metadata
        let input = &metadata.inputs[0];
        assert_eq!(input.rust_name, "emissions_co2");
        assert_eq!(input.variable_name, "Emissions|CO2");
        assert_eq!(input.unit, "GtCO2");
        assert_eq!(input.grid, GridType::Scalar);

        // Check output metadata
        let output = &metadata.outputs[0];
        assert_eq!(output.rust_name, "concentration_co2");
        assert_eq!(output.variable_name, "Concentrations|CO2");
        assert_eq!(output.unit, "ppm");
        assert_eq!(output.grid, GridType::Scalar);
    }

    #[test]
    fn test_component_metadata_with_tags() {
        let metadata = TaggedTestComponent::component_metadata();

        assert_eq!(metadata.name, "TaggedTestComponent");
        assert_eq!(metadata.tags, vec!["test", "example", "simple"]);
        assert_eq!(metadata.category, Some("Testing".to_string()));
        assert_eq!(metadata.inputs.len(), 1);
        assert_eq!(metadata.outputs.len(), 1);
        assert!(metadata.states.is_empty());

        // Check input metadata
        let input = &metadata.inputs[0];
        assert_eq!(input.rust_name, "input_var");
        assert_eq!(input.variable_name, "Test Input");
        assert_eq!(input.unit, "units");

        // Check output metadata
        let output = &metadata.outputs[0];
        assert_eq!(output.rust_name, "output_var");
        assert_eq!(output.variable_name, "Test Output");
        assert_eq!(output.unit, "units");
    }
}
