//! Aggregate execution tests.

use crate::component::{Component, InputState, OutputState, TimeseriesType, VariableDefinition};
use crate::errors::RSCMResult;
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::ModelBuilder;
use crate::schema::{AggregateOp, VariableSchema};
use crate::spatial::{ScalarGrid, ScalarRegion};
use crate::state::StateValue;
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use crate::VariableSource;
use numpy::array;
use numpy::ndarray::{Array, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChainForcing {
    input: Option<String>,
    output: String,
}

#[typetag::serde]
impl Component for ChainForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut defs = vec![VariableDefinition::scalar_output(&self.output, "W/m^2")];
        if let Some(input) = &self.input {
            defs.push(VariableDefinition::scalar_input(input, "W/m^2"));
        }
        defs
    }

    fn solve(&self, _t: Time, _next: Time, inputs: &InputState) -> RSCMResult<OutputState> {
        let upstream = self
            .input
            .as_ref()
            .map(|name| inputs.get_scalar_window(name).get())
            .unwrap_or(0.0);
        Ok(OutputState::from([(
            self.output.clone(),
            StateValue::Scalar(upstream + 1.0),
        )]))
    }
}

#[test]
fn test_aggregate_waits_for_unequal_length_branches() {
    let schema = VariableSchema::new()
        .variable("root", "W/m^2")
        .variable("middle", "W/m^2")
        .variable("long", "W/m^2")
        .variable("short", "W/m^2")
        .aggregate("total", "W/m^2", AggregateOp::Sum)
        .from("short")
        .from("long")
        .build();
    let mut builder = ModelBuilder::new();
    builder
        .with_schema(schema)
        .with_time_axis(TimeAxis::from_values(array![0.0, 1.0, 2.0]));
    for (input, output) in [
        (None, "root"),
        (Some("root"), "middle"),
        (Some("middle"), "long"),
        (None, "short"),
    ] {
        builder.with_component(Arc::new(ChainForcing {
            input: input.map(String::from),
            output: output.into(),
        }));
    }
    let mut model = builder.build().unwrap();
    model.run();
    let values = model
        .timeseries()
        .get_data("total")
        .unwrap()
        .as_scalar()
        .unwrap();
    assert_eq!(values.at(1, ScalarRegion::Global).unwrap(), 4.0);
    assert_eq!(values.at(2, ScalarRegion::Global).unwrap(), 4.0);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct FailingForcing;

#[typetag::serde]
impl Component for FailingForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::scalar_output("failed", "W/m^2")]
    }

    fn solve(&self, _t: Time, _next: Time, _inputs: &InputState) -> RSCMResult<OutputState> {
        Err(crate::errors::RSCMError::Error(
            "forcing unavailable".into(),
        ))
    }
}

#[test]
fn test_run_reports_component_error_without_advancing_time() {
    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(array![1750.0, 1751.0]))
        .with_component(Arc::new(FailingForcing))
        .build()
        .unwrap();
    let error = model.try_run().unwrap_err().to_string();
    assert!(error.contains("FailingForcing"));
    assert!(error.contains("1750..1751"));
    assert!(error.contains("forcing unavailable"));
    assert_eq!(model.time_index(), 0);
}

/// A simple component that produces Effective Radiative Forcing|CO2
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CO2ERFComponent {
    forcing_per_ppm: f64,
}

#[typetag::serde]
impl Component for CO2ERFComponent {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::scalar_input("Atmospheric Concentrations|CO2", "ppm"),
            VariableDefinition::scalar_output("Effective Radiative Forcing|CO2", "W/m^2"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let conc = input_state
            .get_scalar_window("Atmospheric Concentrations|CO2")
            .get();
        let mut output = OutputState::new();
        output.insert(
            "Effective Radiative Forcing|CO2".to_string(),
            StateValue::Scalar(conc * self.forcing_per_ppm),
        );
        Ok(output)
    }
}

/// A simple component that produces Effective Radiative Forcing|CH4
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CH4ERFComponent {
    forcing_per_ppb: f64,
}

#[typetag::serde]
impl Component for CH4ERFComponent {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::scalar_input("Atmospheric Concentrations|CH4", "ppb"),
            VariableDefinition::scalar_output("Effective Radiative Forcing|CH4", "W/m^2"),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let conc = input_state
            .get_scalar_window("Atmospheric Concentrations|CH4")
            .get();
        let mut output = OutputState::new();
        output.insert(
            "Effective Radiative Forcing|CH4".to_string(),
            StateValue::Scalar(conc * self.forcing_per_ppb),
        );
        Ok(output)
    }
}

fn get_co2_concentrations() -> Timeseries<FloatValue> {
    let values = array![280.0, 400.0].insert_axis(Axis(1));
    Timeseries::new(
        values,
        Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
        ScalarGrid,
        "ppm".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

fn get_ch4_concentrations() -> Timeseries<FloatValue> {
    let values = array![700.0, 1800.0].insert_axis(Axis(1));
    Timeseries::new(
        values,
        Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
        ScalarGrid,
        "ppb".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

#[test]
fn test_aggregate_sum_execution() {
    // Schema with aggregate summing two ERF components
    let schema = VariableSchema::new()
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Atmospheric Concentrations|CH4", "ppb")
        .variable("Effective Radiative Forcing|CO2", "W/m^2")
        .variable("Effective Radiative Forcing|CH4", "W/m^2")
        .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build();

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(CO2ERFComponent {
            forcing_per_ppm: 0.01, // 1 W/m^2 per 100 ppm
        }))
        .with_component(Arc::new(CH4ERFComponent {
            forcing_per_ppb: 0.001, // 1 W/m^2 per 1000 ppb
        }))
        .with_prescribed_variable("Atmospheric Concentrations|CO2", get_co2_concentrations())
        .with_prescribed_variable("Atmospheric Concentrations|CH4", get_ch4_concentrations())
        .build()
        .unwrap();

    // Run the model
    model.run();

    // Check that the aggregate was computed
    let total_erf = model
        .timeseries()
        .get_data("Effective Radiative Forcing")
        .and_then(|d| d.as_scalar())
        .expect("Effective Radiative Forcing should exist");

    // At 2021+ (after first step):
    // CO2: 400 ppm * 0.01 = 4.0 W/m^2
    // CH4: 1800 ppb * 0.001 = 1.8 W/m^2
    // Sum: 5.8 W/m^2
    let value = total_erf.at(1, ScalarRegion::Global).unwrap();
    assert!(
        (value - 5.8).abs() < 1e-10,
        "Effective Radiative Forcing should be 5.8, got {}",
        value
    );
}

#[test]
fn test_aggregate_mean_execution() {
    // Schema with mean aggregate
    let schema = VariableSchema::new()
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Atmospheric Concentrations|CH4", "ppb")
        .variable("Effective Radiative Forcing|CO2", "W/m^2")
        .variable("Effective Radiative Forcing|CH4", "W/m^2")
        .aggregate(
            "Mean Effective Radiative Forcing",
            "W/m^2",
            AggregateOp::Mean,
        )
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build();

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(CO2ERFComponent {
            forcing_per_ppm: 0.01,
        }))
        .with_component(Arc::new(CH4ERFComponent {
            forcing_per_ppb: 0.001,
        }))
        .with_prescribed_variable("Atmospheric Concentrations|CO2", get_co2_concentrations())
        .with_prescribed_variable("Atmospheric Concentrations|CH4", get_ch4_concentrations())
        .build()
        .unwrap();

    model.run();

    let mean_erf = model
        .timeseries()
        .get_data("Mean Effective Radiative Forcing")
        .and_then(|d| d.as_scalar())
        .expect("Mean Effective Radiative Forcing should exist");

    // Mean of 4.0 and 1.8 = 2.9 W/m^2
    let value = mean_erf.at(1, ScalarRegion::Global).unwrap();
    assert!(
        (value - 2.9).abs() < 1e-10,
        "Mean Effective Radiative Forcing should be 2.9, got {}",
        value
    );
}

#[test]
fn test_aggregate_weighted_execution() {
    // Schema with weighted aggregate (80% CO2, 20% CH4)
    let schema = VariableSchema::new()
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Atmospheric Concentrations|CH4", "ppb")
        .variable("Effective Radiative Forcing|CO2", "W/m^2")
        .variable("Effective Radiative Forcing|CH4", "W/m^2")
        .aggregate(
            "Effective Radiative Forcing",
            "W/m^2",
            AggregateOp::WeightedSum(vec![0.8, 0.2]),
        )
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build();

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(CO2ERFComponent {
            forcing_per_ppm: 0.01,
        }))
        .with_component(Arc::new(CH4ERFComponent {
            forcing_per_ppb: 0.001,
        }))
        .with_prescribed_variable("Atmospheric Concentrations|CO2", get_co2_concentrations())
        .with_prescribed_variable("Atmospheric Concentrations|CH4", get_ch4_concentrations())
        .build()
        .unwrap();

    model.run();

    let weighted_erf = model
        .timeseries()
        .get_data("Effective Radiative Forcing")
        .and_then(|d| d.as_scalar())
        .expect("Effective Radiative Forcing should exist");

    // Weighted: 4.0 * 0.8 + 1.8 * 0.2 = 3.2 + 0.36 = 3.56 W/m^2
    let value = weighted_erf.at(1, ScalarRegion::Global).unwrap();
    assert!(
        (value - 3.56).abs() < 1e-10,
        "Effective Radiative Forcing should be 3.56, got {}",
        value
    );
}

#[test]
fn test_aggregate_with_nan_contributor() {
    // Schema where one contributor has no writer (all NaN)
    let schema = VariableSchema::new()
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Effective Radiative Forcing|CO2", "W/m^2")
        .variable("Effective Radiative Forcing|N2O", "W/m^2") // No component writes this
        .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|N2O")
        .build();

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(CO2ERFComponent {
            forcing_per_ppm: 0.01,
        }))
        .with_prescribed_variable("Atmospheric Concentrations|CO2", get_co2_concentrations())
        .build()
        .unwrap();

    model.run();

    let total_erf = model
        .timeseries()
        .get_data("Effective Radiative Forcing")
        .and_then(|d| d.as_scalar())
        .expect("Effective Radiative Forcing should exist");

    // Effective Radiative Forcing|N2O is NaN, so Sum should just be Effective Radiative Forcing|CO2 = 4.0 W/m^2
    let value = total_erf.at(1, ScalarRegion::Global).unwrap();
    assert!(
        (value - 4.0).abs() < 1e-10,
        "Effective Radiative Forcing should be 4.0 (NaN excluded), got {}",
        value
    );
}

#[test]
fn test_chained_aggregates_execution() {
    // Chained aggregates: the forcing sums Greenhouse Gases and Other, Greenhouse Gases sums CO2 and CH4
    let schema = VariableSchema::new()
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Atmospheric Concentrations|CH4", "ppb")
        .variable("Effective Radiative Forcing|CO2", "W/m^2")
        .variable("Effective Radiative Forcing|CH4", "W/m^2")
        .variable("Effective Radiative Forcing|Other", "W/m^2") // Will be NaN
        .aggregate(
            "Effective Radiative Forcing|Greenhouse Gases",
            "W/m^2",
            AggregateOp::Sum,
        )
        .from("Effective Radiative Forcing|CO2")
        .from("Effective Radiative Forcing|CH4")
        .build()
        .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
        .from("Effective Radiative Forcing|Greenhouse Gases")
        .from("Effective Radiative Forcing|Other")
        .build();

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(CO2ERFComponent {
            forcing_per_ppm: 0.01,
        }))
        .with_component(Arc::new(CH4ERFComponent {
            forcing_per_ppb: 0.001,
        }))
        .with_prescribed_variable("Atmospheric Concentrations|CO2", get_co2_concentrations())
        .with_prescribed_variable("Atmospheric Concentrations|CH4", get_ch4_concentrations())
        .build()
        .unwrap();

    model.run();

    // Check Effective Radiative Forcing|Greenhouse Gases = CO2 + CH4 = 4.0 + 1.8 = 5.8
    let ghg_erf = model
        .timeseries()
        .get_data("Effective Radiative Forcing|Greenhouse Gases")
        .and_then(|d| d.as_scalar())
        .expect("Effective Radiative Forcing|Greenhouse Gases should exist");
    let ghg_value = ghg_erf.at(1, ScalarRegion::Global).unwrap();
    assert!(
        (ghg_value - 5.8).abs() < 1e-10,
        "Effective Radiative Forcing|Greenhouse Gases should be 5.8, got {}",
        ghg_value
    );

    // Check Effective Radiative Forcing = Greenhouse Gases + Other (NaN) = 5.8
    let total_erf = model
        .timeseries()
        .get_data("Effective Radiative Forcing")
        .and_then(|d| d.as_scalar())
        .expect("Effective Radiative Forcing should exist");
    let total_value = total_erf.at(1, ScalarRegion::Global).unwrap();
    assert!(
        (total_value - 5.8).abs() < 1e-10,
        "Effective Radiative Forcing should be 5.8, got {}",
        total_value
    );
}

#[test]
fn test_aggregate_appears_in_dot_graph() {
    let schema = VariableSchema::new()
        .variable("Atmospheric Concentrations|CO2", "ppm")
        .variable("Effective Radiative Forcing|CO2", "W/m^2")
        .aggregate("Effective Radiative Forcing", "W/m^2", AggregateOp::Sum)
        .from("Effective Radiative Forcing|CO2")
        .build();

    let model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_schema(schema)
        .with_component(Arc::new(CO2ERFComponent {
            forcing_per_ppm: 0.01,
        }))
        .with_prescribed_variable("Atmospheric Concentrations|CO2", get_co2_concentrations())
        .build()
        .unwrap();

    let dot = format!("{:?}", model.as_dot());

    // The aggregate component should appear in the graph
    assert!(
        dot.contains("AggregateComponent"),
        "Graph should contain AggregateComponent: {}",
        dot
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepEmissions;

#[typetag::serde]
impl Component for StepEmissions {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::scalar_output(
            "Upstream Emissions",
            "GtC/yr",
        )]
    }

    fn solve(&self, t: Time, _next: Time, _inputs: &InputState) -> RSCMResult<OutputState> {
        Ok(OutputState::from([(
            "Upstream Emissions".to_string(),
            StateValue::Scalar(100.0 + (t - 2020.0)),
        )]))
    }
}

#[test]
fn test_flux_aggregate_reads_prescribed_and_upstream_at_step_start() {
    let schema = VariableSchema::new()
        .variable("Prescribed Emissions", "GtC/yr")
        .variable("Upstream Emissions", "GtC/yr")
        .aggregate("Emissions", "GtC/yr", AggregateOp::Sum)
        .from("Prescribed Emissions")
        .from("Upstream Emissions")
        .build();
    let axis = Arc::new(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)));
    let prescribed = Timeseries::new(
        array![1.0, 2.0, 3.0, 4.0, 5.0].insert_axis(Axis(1)),
        axis.clone(),
        ScalarGrid,
        "GtC/yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut model = ModelBuilder::new()
        .with_time_axis((*axis).clone())
        .with_schema(schema)
        .with_component(Arc::new(StepEmissions))
        .with_prescribed_variable("Prescribed Emissions", prescribed)
        .build()
        .unwrap();

    let coupling = model.coupling().unwrap();
    let agg_component = coupling
        .components
        .iter()
        .find(|c| c.name == "AggregateComponent")
        .unwrap();
    let sources: HashMap<_, _> = agg_component
        .inputs
        .iter()
        .map(|v| (v.name.as_str(), (v.source, v.timeseries_type)))
        .collect();
    assert_eq!(
        sources["Prescribed Emissions"],
        (Some(VariableSource::Prescribed), TimeseriesType::Average)
    );
    assert_eq!(
        sources["Upstream Emissions"],
        (Some(VariableSource::Upstream), TimeseriesType::Average)
    );

    model.run();

    // Step N sums the prescribed value and the upstream output at N, and writes the total at N.
    let total = model
        .timeseries()
        .get_data("Emissions")
        .and_then(|d| d.as_scalar())
        .unwrap();
    let values: Vec<_> = (0..4)
        .map(|i| total.at(i, ScalarRegion::Global).unwrap())
        .collect();
    assert_eq!(values, vec![101.0, 103.0, 105.0, 107.0]);
    assert!(total.at(4, ScalarRegion::Global).unwrap().is_nan());
}
