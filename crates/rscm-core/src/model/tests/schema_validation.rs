//! Schema validation tests.

use super::grid_validation::{FourBoxConsumer, FourBoxProducer};
use crate::component::GridType;
use crate::example_components::{TestComponent, TestComponentParameters};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::ModelBuilder;
use crate::schema::{AggregateOp, VariableSchema};
use crate::spatial::ScalarGrid;
use crate::timeseries::{FloatValue, TimeAxis, Timeseries};
use numpy::array;
use numpy::ndarray::{Array, Axis};
use std::sync::Arc;

fn get_emissions() -> Timeseries<FloatValue> {
    let values = array![0.0, 10.0].insert_axis(Axis(1));
    Timeseries::new(
        values,
        Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
        ScalarGrid,
        "GtC / yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

#[test]
fn test_model_with_valid_schema() {
    // Schema that matches component requirements
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2")
        .variable("Concentrations|CO2", "ppm");

    let _model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();
}

#[test]
fn test_schema_rejects_undefined_output() {
    // Schema missing the output variable
    let schema = VariableSchema::new().variable("Emissions|CO2", "GtCO2");
    // Missing "Concentrations|CO2"

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Concentrations|CO2"),
        "Error should mention missing variable: {}",
        msg
    );
    assert!(
        msg.contains("not defined in the schema"),
        "Error should indicate schema issue: {}",
        msg
    );
}

#[test]
fn test_schema_rejects_undefined_input() {
    // Schema missing the input variable (and no component produces it)
    let schema = VariableSchema::new().variable("Concentrations|CO2", "ppm");
    // Missing "Emissions|CO2"

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Emissions|CO2"),
        "Error should mention missing variable: {}",
        msg
    );
}

#[test]
fn test_schema_rejects_incompatible_units() {
    // Schema with dimensionally incompatible unit for output variable
    // The component outputs "ppm" (dimensionless) but schema expects "GtC" (mass)
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2")
        .variable("Concentrations|CO2", "GtC"); // Wrong dimension - should be "ppm"

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    // Now correctly identifies incompatible dimensions rather than string mismatch
    assert!(
        msg.contains("Incompatible units"),
        "Error should indicate incompatible units: {}",
        msg
    );
    assert!(
        msg.contains("Concentrations|CO2"),
        "Error should mention the variable: {}",
        msg
    );
    // Should explain the dimension mismatch
    assert!(
        msg.contains("dimension"),
        "Error should mention dimensions: {}",
        msg
    );
}

#[test]
fn test_schema_rejects_disaggregation_on_read() {
    // Schema has Scalar temperature, but FourBoxConsumer wants FourBox input
    // This is a disaggregation (broadcast) attempt and should fail
    let schema = VariableSchema::new()
        .variable_with_grid("Temperature", "K", GridType::Scalar)
        .variable("GlobalTemperature", "K");

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(FourBoxProducer)) // Writes FourBox, aggregated to Scalar (OK)
        .with_component(Arc::new(FourBoxConsumer)) // Reads FourBox from Scalar schema (ERROR)
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Grid transformation not supported"),
        "Error should indicate unsupported transformation: {}",
        msg
    );
    assert!(
        msg.contains("Scalar") && msg.contains("FourBox"),
        "Error should mention the grid types: {}",
        msg
    );
}

#[test]
fn test_schema_with_aggregate_validates() {
    // Schema with an aggregate definition
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2")
        .variable("Concentrations|CO2", "ppm")
        .aggregate("Total Concentrations", "ppm", AggregateOp::Sum)
        .from("Concentrations|CO2")
        .build();

    let _model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();
}

#[test]
fn test_schema_creates_nan_for_unwritten_variables() {
    // Schema has a variable that no component writes to
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2")
        .variable("Concentrations|CO2", "ppm")
        .variable("Concentrations|CH4", "ppb"); // No component writes this

    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    // The CH4 variable should exist but be all NaN
    let ch4 = model
        .timeseries()
        .get_data("Concentrations|CH4")
        .and_then(|d| d.as_scalar());
    assert!(
        ch4.is_some(),
        "CH4 timeseries should exist even though no component writes it"
    );
    let ch4 = ch4.unwrap();
    assert!(
        ch4.values().iter().all(|v| v.is_nan()),
        "All CH4 values should be NaN since no component writes to it"
    );
}

#[test]
fn test_schema_invalid_aggregate_fails() {
    // Schema with invalid aggregate (circular dependency)
    let mut schema = VariableSchema::new();
    schema.aggregates.insert(
        "A".to_string(),
        crate::schema::AggregateDefinition {
            name: "A".to_string(),
            unit: "units".to_string(),
            grid_type: GridType::Scalar,
            operation: AggregateOp::Sum,
            contributors: vec!["B".to_string()],
        },
    );
    schema.aggregates.insert(
        "B".to_string(),
        crate::schema::AggregateDefinition {
            name: "B".to_string(),
            unit: "units".to_string(),
            grid_type: GridType::Scalar,
            operation: AggregateOp::Sum,
            contributors: vec!["A".to_string()],
        },
    );

    let result = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Circular dependency"),
        "Error should indicate circular dependency: {}",
        msg
    );
}

#[test]
fn test_model_without_schema_still_works() {
    // Ensure models without schema still work as before
    let _model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();
}
