//! Tests for unit validation and compatibility checking during model building.
//!
//! These tests verify that:
//! - Compatible units with different strings are accepted (e.g., "W/m^2" and "W / m^2")
//! - Dimensionally compatible units are accepted (e.g., "GtC/yr" and "MtCO2/yr")
//! - Dimensionally incompatible units are rejected (e.g., "W/m^2" and "GtC/yr")
//! - Unit conversion factors are correctly calculated

use crate::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use crate::errors::{RscmError, RscmResult};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::{Model, ModelBuilder};
use crate::schema::VariableSchema;
use crate::spatial::{FourBoxGrid, ScalarGrid};
use crate::timeseries::{FloatValue, GridTimeseries, Time, TimeAxis, Timeseries};
use numpy::array;
use numpy::ndarray::{Array, Axis};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A simple component that reads emissions and outputs concentration.
/// Used to test unit validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmissionsToConcentration {
    input_unit: String,
    output_unit: String,
}

#[typetag::serde]
impl Component for EmissionsToConcentration {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new("Emissions|CO2", &self.input_unit, VariableRole::Input),
            VariableDefinition::new(
                "Atmospheric Concentrations|CO2",
                &self.output_unit,
                VariableRole::Output,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RscmResult<OutputState> {
        Ok(OutputState::new())
    }
}

/// A component that chains from concentration to forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConcentrationToForcing {
    input_unit: String,
    output_unit: String,
}

#[typetag::serde]
impl Component for ConcentrationToForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new(
                "Atmospheric Concentrations|CO2",
                &self.input_unit,
                VariableRole::Input,
            ),
            VariableDefinition::new(
                "Effective Radiative Forcing|CO2",
                &self.output_unit,
                VariableRole::Output,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RscmResult<OutputState> {
        Ok(OutputState::new())
    }
}

fn get_emissions(unit: &str) -> Timeseries<FloatValue> {
    let values = array![0.0, 10.0].insert_axis(Axis(1));
    Timeseries::new(
        values,
        Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
        ScalarGrid,
        unit.to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

#[test]
fn test_identical_units_accepted() {
    // Components using identical unit strings should work
    let comp1 = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(),
        output_unit: "ppm".to_string(),
    };
    let comp2 = ConcentrationToForcing {
        input_unit: "ppm".to_string(), // Matches comp1's output exactly
        output_unit: "W/m^2".to_string(),
    };

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(comp1))
        .with_component(Arc::new(comp2))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtC/yr"))
        .build();

    assert!(
        result.is_ok(),
        "Identical units should be accepted: {:?}",
        result.err()
    );
}

#[test]
fn test_whitespace_normalised_units_accepted() {
    // Units that differ only in whitespace should be accepted
    let comp1 = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(),
        output_unit: "ppm".to_string(),
    };
    let comp2 = ConcentrationToForcing {
        input_unit: "ppm".to_string(),
        output_unit: "W/m^2".to_string(), // Standard form
    };
    // Using a schema with different whitespace
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtC / yr") // Different whitespace
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Effective Radiative Forcing|CO2", "W / m ^ 2"); // Different whitespace

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(comp1))
        .with_component(Arc::new(comp2))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtC/yr"))
        .build();

    assert!(
        result.is_ok(),
        "Whitespace-normalised units should be accepted: {:?}",
        result.err()
    );
}

#[test]
fn test_compatible_units_different_magnitudes_between_components() {
    // Components using dimensionally compatible units with different magnitudes
    // e.g., one outputs GtC/yr, another reads MtC/yr (factor of 1000)
    let comp1 = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(),
        output_unit: "ppm".to_string(),
    };

    // This component reads the same variable but in different units
    // Currently this should fail validation as we need schema for unit conversion
    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(comp1))
        .with_prescribed_variable("Emissions|CO2", get_emissions("MtC/yr")) // Different magnitude!
        .build();

    // Without schema, components must use identical units
    // The prescribed timeseries unit doesn't participate in component validation
    // This should succeed because prescribed variable unit matching isn't checked
    assert!(result.is_ok());
}

#[test]
fn test_schema_accepts_compatible_units() {
    // Schema with one unit, component with compatible but different unit
    // e.g., schema: GtC/yr, component: MtC/yr (both are mass/time)
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtC/yr")
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Effective Radiative Forcing|CO2", "W/m^2");

    let comp1 = EmissionsToConcentration {
        input_unit: "MtC/yr".to_string(), // Different magnitude but compatible
        output_unit: "ppm".to_string(),
    };
    let comp2 = ConcentrationToForcing {
        input_unit: "ppm".to_string(),
        output_unit: "W/m^2".to_string(),
    };

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(comp1))
        .with_component(Arc::new(comp2))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtC/yr"))
        .build();

    // This should now succeed because MtC/yr is dimensionally compatible with GtC/yr
    assert!(
        result.is_ok(),
        "Dimensionally compatible units should be accepted: {:?}",
        result.err()
    );
}

#[test]
fn test_schema_rejects_dimensionally_incompatible_units() {
    // Schema expects W/m^2 but component outputs GtC/yr (different dimensions)
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtC/yr")
        .variable("Atmospheric Concentrations|CO2", "W/m^2"); // Wrong dimension! Should be "ppm"

    let comp = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(),
        output_unit: "ppm".to_string(), // dimensionless
    };

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(comp))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtC/yr"))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Incompatible units"),
        "Should report incompatible units: {}",
        err
    );
    assert!(
        err.contains("Atmospheric Concentrations|CO2"),
        "Should mention the variable: {}",
        err
    );
}

#[test]
fn test_carbon_co2_conversion_accepted() {
    // GtC and GtCO2 are compatible (both mass) with molecular weight conversion
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2/yr") // Schema uses CO2
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Effective Radiative Forcing|CO2", "W/m^2");

    let comp1 = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(), // Component uses C (compatible)
        output_unit: "ppm".to_string(),
    };
    let comp2 = ConcentrationToForcing {
        input_unit: "ppm".to_string(),
        output_unit: "W/m^2".to_string(),
    };

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(comp1))
        .with_component(Arc::new(comp2))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtCO2/yr"))
        .build();

    // GtC/yr and GtCO2/yr are dimensionally compatible (both mass/time)
    // The conversion factor is 12/44 (C to CO2)
    assert!(
        result.is_ok(),
        "Carbon-CO2 unit conversion should be accepted: {:?}",
        result.err()
    );
}

#[test]
fn test_unparseable_units_warn_but_allow_string_comparison() {
    // If a unit cannot be parsed, we should still allow string comparison
    // This maintains backward compatibility
    let comp1 = EmissionsToConcentration {
        input_unit: "strange_unit".to_string(),
        output_unit: "ppm".to_string(),
    };

    // Without schema, string comparison should work if both use same string
    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(comp1))
        .with_prescribed_variable("Emissions|CO2", get_emissions("strange_unit"))
        .build();

    // Identical strings match without parsing.
    assert!(result.is_ok());
}

#[test]
fn test_between_component_units_must_match_without_schema() {
    // When two components share a variable, their units must be compatible
    let comp1 = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(),
        output_unit: "ppm".to_string(),
    };
    let comp2 = ConcentrationToForcing {
        input_unit: "W/m^2".to_string(), // WRONG: incompatible with ppm
        output_unit: "W/m^2".to_string(),
    };

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(comp1))
        .with_component(Arc::new(comp2))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtC/yr"))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Incompatible units"),
        "Should report incompatible units: {}",
        err
    );
}

#[test]
fn test_between_component_units_compatible_but_different() {
    // When two components share a variable with compatible but different units
    let comp1 = EmissionsToConcentration {
        input_unit: "GtC/yr".to_string(),
        output_unit: "GtC".to_string(), // Output mass
    };

    // A component that reads mass but in different units
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct MassReader {
        input_unit: String,
    }

    #[typetag::serde]
    impl Component for MassReader {
        fn definitions(&self) -> Vec<VariableDefinition> {
            vec![VariableDefinition::new(
                "Atmospheric Concentrations|CO2",
                &self.input_unit,
                VariableRole::Input,
            )]
        }

        fn solve(
            &self,
            _t_current: Time,
            _t_next: Time,
            _input_state: &InputState,
        ) -> RscmResult<OutputState> {
            Ok(OutputState::new())
        }
    }

    let comp2 = MassReader {
        input_unit: "MtC".to_string(), // Compatible but different magnitude
    };

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(comp1))
        .with_component(Arc::new(comp2))
        .with_prescribed_variable("Emissions|CO2", get_emissions("GtC/yr"))
        .build();

    // This should succeed because GtC and MtC are dimensionally compatible
    assert!(
        result.is_ok(),
        "Compatible units between components should be accepted: {:?}",
        result.err()
    );
}

// ============================================================================
// Runtime Unit Conversion Tests
// ============================================================================

/// A component that reads an input value and outputs it multiplied by a factor.
/// Used to verify that unit conversions are actually applied at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScaleAndOutput {
    /// The unit for the input variable
    input_unit: String,
    /// Internal scaling factor to apply
    scale: f64,
}

#[typetag::serde]
impl Component for ScaleAndOutput {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::new("Input", &self.input_unit, VariableRole::Input),
            VariableDefinition::new("Output", "1", VariableRole::Output),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        // Get the input value (after any unit conversion has been applied)
        let input_window = input_state.get_scalar_window("Input");
        let input_value = input_window.get();

        // Apply our internal scale and output
        let mut output = OutputState::new();
        output.insert("Output".to_string(), (input_value * self.scale).into());
        Ok(output)
    }
}

#[test]
fn test_unit_conversion_applied_at_runtime_gtc_to_mtc() {
    // Test that unit conversion factors are actually applied at runtime.
    // Schema stores data in GtC/yr, component requests MtC/yr (factor of 1000).
    // If conversion works, the component should receive values 1000x larger.

    let schema = VariableSchema::new()
        .variable("Input", "GtC/yr") // Schema stores in GtC/yr
        .variable("Output", "1");

    let comp = ScaleAndOutput {
        input_unit: "MtC/yr".to_string(), // Component expects MtC/yr
        scale: 1.0,                       // Just pass through the converted value
    };

    // Create prescribed data with known values
    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    let values = array![10.0, 20.0, 30.0].insert_axis(Axis(1)); // 10, 20, 30 GtC/yr
    let emissions = Timeseries::new(
        values,
        time_axis.clone(),
        ScalarGrid,
        "GtC/yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut model = ModelBuilder::new()
        .with_time_axis((*time_axis).clone())
        .with_schema(schema)
        .with_component(Arc::new(comp))
        .with_prescribed_variable("Input", emissions)
        .build()
        .expect("Model should build successfully");

    // Check that unit conversion was registered
    let conversions = model.unit_conversions();
    assert!(
        !conversions.is_empty(),
        "Unit conversions should be registered"
    );

    // Find the conversion for our input
    let factor = conversions
        .get(&("Input".to_string(), "ScaleAndOutput".to_string()))
        .expect("Should have conversion for Input/ScaleAndOutput");

    // GtC -> MtC conversion factor is 1000 (1 Gt = 1000 Mt)
    assert!(
        (*factor - 1000.0).abs() < 0.001,
        "Conversion factor should be 1000, got {}",
        factor
    );

    // Run the model
    model.run();

    // Check the output values
    // Input was 10 GtC/yr at t=2020, component requests MtC/yr
    // So component should receive 10,000 MtC/yr
    // Since scale=1.0, output should be 10,000

    let output_ts = model.timeseries().get_data("Output").unwrap();
    let output_value = output_ts.as_scalar().unwrap().at_scalar(1).unwrap(); // After first step

    assert!(
        (output_value - 10000.0).abs() < 0.001,
        "Output should be 10000 (10 GtC/yr * 1000 = 10000 MtC/yr), got {}",
        output_value
    );
}

#[test]
fn test_unit_conversion_applied_at_runtime_co2_to_c() {
    // Test CO2 to C conversion (factor of 12/44 ≈ 0.2727).
    // Schema stores in GtCO2/yr, component requests GtC/yr.

    let schema = VariableSchema::new()
        .variable("Input", "GtCO2/yr") // Schema stores in CO2
        .variable("Output", "1");

    let comp = ScaleAndOutput {
        input_unit: "GtC/yr".to_string(), // Component expects C
        scale: 1.0,
    };

    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    let values = array![44.0, 88.0, 132.0].insert_axis(Axis(1)); // 44, 88, 132 GtCO2/yr
    let emissions = Timeseries::new(
        values,
        time_axis.clone(),
        ScalarGrid,
        "GtCO2/yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut model = ModelBuilder::new()
        .with_time_axis((*time_axis).clone())
        .with_schema(schema)
        .with_component(Arc::new(comp))
        .with_prescribed_variable("Input", emissions)
        .build()
        .expect("Model should build successfully");

    // Run the model
    model.run();

    // Check the output
    // Input was 44 GtCO2/yr, converted to GtC/yr with factor 12/44
    // Expected: 44 * (12/44) = 12 GtC/yr
    let output_ts = model.timeseries().get_data("Output").unwrap();
    let output_value = output_ts.as_scalar().unwrap().at_scalar(1).unwrap();

    assert!(
        (output_value - 12.0).abs() < 0.001,
        "Output should be 12 (44 GtCO2/yr * 12/44 = 12 GtC/yr), got {}",
        output_value
    );
}

#[test]
fn test_unit_conversion_combined_with_prefix_and_molecular_weight() {
    // Test combined conversion: GtC/yr (schema) to MtCO2/yr (component)
    // Factor: 1000 (Gt to Mt) * 44/12 (C to CO2) = 3666.67

    let schema = VariableSchema::new()
        .variable("Input", "GtC/yr")
        .variable("Output", "1");

    let comp = ScaleAndOutput {
        input_unit: "MtCO2/yr".to_string(), // Component expects Mt of CO2
        scale: 1.0,
    };

    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    let values = array![1.0, 2.0, 3.0].insert_axis(Axis(1)); // 1, 2, 3 GtC/yr
    let emissions = Timeseries::new(
        values,
        time_axis.clone(),
        ScalarGrid,
        "GtC/yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut model = ModelBuilder::new()
        .with_time_axis((*time_axis).clone())
        .with_schema(schema)
        .with_component(Arc::new(comp))
        .with_prescribed_variable("Input", emissions)
        .build()
        .expect("Model should build successfully");

    // Verify the conversion factor
    let factor = model
        .unit_conversions()
        .get(&("Input".to_string(), "ScaleAndOutput".to_string()))
        .copied()
        .unwrap_or(1.0);

    // Expected: 1000 * 44/12 = 3666.67
    let expected_factor = 1000.0 * 44.0 / 12.0;
    assert!(
        (factor - expected_factor).abs() < 0.01,
        "Conversion factor should be {}, got {}",
        expected_factor,
        factor
    );

    // Run the model
    model.run();

    // Check the output
    // Input was 1 GtC/yr, converted to MtCO2/yr
    // Expected: 1 * 1000 * 44/12 = 3666.67 MtCO2/yr
    let output_ts = model.timeseries().get_data("Output").unwrap();
    let output_value = output_ts.as_scalar().unwrap().at_scalar(1).unwrap();

    assert!(
        (output_value - expected_factor).abs() < 0.01,
        "Output should be {} (1 GtC/yr * factor), got {}",
        expected_factor,
        output_value
    );
}

#[test]
fn test_no_conversion_when_units_match() {
    // When units match exactly, no conversion should be needed
    let schema = VariableSchema::new()
        .variable("Input", "GtC/yr")
        .variable("Output", "1");

    let comp = ScaleAndOutput {
        input_unit: "GtC/yr".to_string(), // Same as schema
        scale: 1.0,
    };

    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)));
    let values = array![100.0, 200.0, 300.0].insert_axis(Axis(1));
    let emissions = Timeseries::new(
        values,
        time_axis.clone(),
        ScalarGrid,
        "GtC/yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut model = ModelBuilder::new()
        .with_time_axis((*time_axis).clone())
        .with_schema(schema)
        .with_component(Arc::new(comp))
        .with_prescribed_variable("Input", emissions)
        .build()
        .expect("Model should build successfully");

    // Should have no conversions registered (or factor of 1.0)
    let conversions = model.unit_conversions();
    let factor = conversions
        .get(&("Input".to_string(), "ScaleAndOutput".to_string()))
        .copied()
        .unwrap_or(1.0);

    assert!(
        (factor - 1.0).abs() < 0.0001
            || !conversions.contains_key(&("Input".to_string(), "ScaleAndOutput".to_string())),
        "No conversion should be registered for matching units"
    );

    // Run the model
    model.run();

    // Output should equal input (no conversion)
    let output_ts = model.timeseries().get_data("Output").unwrap();
    let output_value = output_ts.as_scalar().unwrap().at_scalar(1).unwrap();

    assert!(
        (output_value - 100.0).abs() < 0.001,
        "Output should be 100 (unchanged), got {}",
        output_value
    );
}

fn emissions_model(component_unit: &str, data: Timeseries<FloatValue>) -> RscmResult<Model> {
    ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(EmissionsToConcentration {
            input_unit: component_unit.to_string(),
            output_unit: "ppm".to_string(),
        }))
        .with_prescribed_variable("Emissions|CO2", data)
        .build()
}

fn stored_emissions(model: &Model) -> &Timeseries<FloatValue> {
    model
        .timeseries()
        .get_data("Emissions|CO2")
        .and_then(|data| data.as_scalar())
        .unwrap()
}

#[test]
fn test_prescribed_data_converted_into_declared_unit() {
    let model = emissions_model("GtC / yr", get_emissions("MtCO2 / yr")).unwrap();

    let stored = stored_emissions(&model);
    assert_eq!(stored.unit(), "GtC / yr");
    let expected = 10.0 * 12.0 / 44.0 / 1000.0;
    let value = stored.at_scalar(0).unwrap();
    assert!(
        (value - expected).abs() < 1e-6 * expected,
        "10 MtCO2/yr should be stored as {expected} GtC/yr, got {value}"
    );
}

#[test]
fn test_prescribed_data_in_equivalent_unit_unchanged() {
    let model = emissions_model("GtC/yr", get_emissions("GtC / yr")).unwrap();

    assert_eq!(stored_emissions(&model).at_scalar(0).unwrap(), 10.0);
}

#[test]
fn test_prescribed_data_with_incompatible_unit_rejected() {
    let err = emissions_model("GtC / yr", get_emissions("W / m^2")).unwrap_err();

    match err {
        RscmError::IncompatibleUnits { variable, .. } => assert_eq!(variable, "Emissions|CO2"),
        other => panic!("expected IncompatibleUnits, got {other:?}"),
    }
}

#[test]
fn test_incompatible_species_error_shows_normalised_units() {
    let err = emissions_model("Mt N/yr", get_emissions("kt N2O/yr")).unwrap_err();

    let message = err.to_string();
    for unit in ["Mt N/yr", "kt N2O/yr"] {
        let normalised = crate::units::Unit::parse(unit).unwrap().normalised();
        assert!(message.contains(&format!("'{normalised}'")), "{message}");
    }
}

#[test]
fn test_prescribed_data_with_unparseable_unit_rejected() {
    let err = emissions_model("GtC / yr", get_emissions("")).unwrap_err();

    match err {
        RscmError::UnitParseError { variable, .. } => assert_eq!(variable, "Emissions|CO2"),
        other => panic!("expected UnitParseError, got {other:?}"),
    }
}

#[test]
fn test_prescribed_data_converted_into_schema_unit() {
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2 / yr")
        .variable("Atmospheric Concentrations|CO2", "ppm");

    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(EmissionsToConcentration {
            input_unit: "GtC / yr".to_string(),
            output_unit: "ppm".to_string(),
        }))
        .with_prescribed_variable("Emissions|CO2", get_emissions("MtCO2 / yr"))
        .build()
        .unwrap();

    let stored = stored_emissions(&model);
    assert_eq!(stored.unit(), "GtCO2 / yr");
    assert!((stored.at_scalar(0).unwrap() - 0.01).abs() < 1e-12);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxReader;

#[typetag::serde]
impl Component for FourBoxReader {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::with_grid(
            "Surface Air Temperature Change",
            "K",
            VariableRole::Input,
            GridType::FourBox,
        )]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RscmResult<OutputState> {
        Ok(OutputState::new())
    }
}

fn four_box_model(unit: &str) -> RscmResult<Model> {
    let time_axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)));
    let values = Array::from_elem((time_axis.len(), 4), 1.0);
    let data = GridTimeseries::new(
        values,
        time_axis.clone(),
        FourBoxGrid::magicc_standard(),
        unit.to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );
    ModelBuilder::new()
        .with_time_axis((*time_axis).clone())
        .with_component(Arc::new(FourBoxReader))
        .with_prescribed_variable("Surface Air Temperature Change", data)
        .build()
}

#[test]
fn test_four_box_prescribed_data_converted() {
    let model = four_box_model("mK").unwrap();

    let stored = model
        .timeseries()
        .get_data("Surface Air Temperature Change")
        .and_then(|data| data.as_four_box())
        .unwrap();
    assert_eq!(stored.unit(), "K");
    assert_eq!(stored.at_time_index(0).unwrap(), vec![0.001; 4]);
}

#[test]
fn test_four_box_prescribed_data_with_incompatible_unit_rejected() {
    let err = four_box_model("W / m^2").unwrap_err();

    assert!(
        matches!(err, RscmError::IncompatibleUnits { ref variable, .. } if variable == "Surface Air Temperature Change"),
        "expected IncompatibleUnits, got {err:?}"
    );
}
