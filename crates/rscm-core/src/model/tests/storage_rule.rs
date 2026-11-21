//! The storage rule: schema entry, then producer, then prescribed data, with reads and writes converting.

use crate::component::{Component, GridType, InputState, OutputState, VariableDefinition};
use crate::errors::{RSCMError, RSCMResult};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::{Model, ModelBuilder};
use crate::schema::VariableSchema;
use crate::spatial::{FourBoxGrid, ScalarGrid, SpatialGrid};
use crate::state::{FourBoxSlice, StateValue};
use crate::timeseries::{FloatValue, GridTimeseries, Time, TimeAxis, Timeseries};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const BOXES: [FloatValue; 4] = [1.0, 2.0, 3.0, 4.0];

/// Reads `input` as a scalar in `unit` and writes what it read to `output`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Echo {
    input: String,
    unit: String,
    output: String,
}

#[typetag::serde]
impl Component for Echo {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::scalar_input(&self.input, &self.unit),
            VariableDefinition::scalar_output(&self.output, &self.unit),
        ]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let value = input_state.get_scalar_window(&self.input).get();
        Ok(OutputState::from([(
            self.output.clone(),
            StateValue::Scalar(value),
        )]))
    }
}

/// Writes `BOXES` to a four-box `T`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxWriter;

#[typetag::serde]
impl Component for FourBoxWriter {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::four_box_output("T", "K")]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        let value = StateValue::FourBox(FourBoxSlice::from_array(BOXES));
        Ok(OutputState::from([("T".to_string(), value)]))
    }
}

/// Writes 1 K to a scalar `T`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarWriter;

#[typetag::serde]
impl Component for ScalarWriter {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::scalar_output("T", "K")]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok(OutputState::from([(
            "T".to_string(),
            StateValue::Scalar(1.0),
        )]))
    }
}

/// Reads `T` on the four-box grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FourBoxReader;

#[typetag::serde]
impl Component for FourBoxReader {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::four_box_input("T", "K")]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok(OutputState::new())
    }
}

fn time_axis() -> Arc<TimeAxis> {
    Arc::new(TimeAxis::from_values(Array::range(2020.0, 2024.0, 1.0)))
}

fn builder() -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder.with_time_axis(time_axis());
    builder
}

fn echo(input: &str, unit: &str, output: &str) -> Arc<Echo> {
    Arc::new(Echo {
        input: input.to_string(),
        unit: unit.to_string(),
        output: output.to_string(),
    })
}

fn scalar_data(value: FloatValue, unit: &str) -> Timeseries<FloatValue> {
    let axis = time_axis();
    Timeseries::new(
        Array::from_elem((axis.len(), 1), value),
        axis,
        ScalarGrid,
        unit.to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

fn four_box_data() -> GridTimeseries<FloatValue, FourBoxGrid> {
    let axis = time_axis();
    let values = Array::from_shape_fn((axis.len(), 4), |(_, region)| BOXES[region]);
    GridTimeseries::new(
        values,
        axis,
        FourBoxGrid::magicc_standard(),
        "K".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

/// Unit, grid and value at the second time of the stored `name`.
fn stored(model: &Model, name: &str) -> (String, GridType, FloatValue) {
    let data = model.timeseries().get_data(name).unwrap();
    let value = match data.as_scalar() {
        Some(ts) => ts.at_scalar(1).unwrap(),
        None => data.as_four_box().unwrap().at_time_index(1).unwrap()[0],
    };
    (data.unit().to_string(), data.grid(), value)
}

fn global_mean() -> FloatValue {
    FourBoxGrid::magicc_standard().aggregate_global(&BOXES)
}

fn assert_grid_mismatch(err: RSCMError, component: &str, access: &str, stored_by: &str) {
    match err {
        RSCMError::GridTypeMismatch {
            component: c,
            access: a,
            stored_by: s,
            ..
        } => assert_eq!(
            (c.as_str(), a.as_str(), s.as_str()),
            (component, access, stored_by)
        ),
        other => panic!("expected GridTypeMismatch, got {other}"),
    }
}

#[test]
fn prescribed_only_variable_read_in_two_units_stays_in_the_supplied_unit() {
    let mut model = builder()
        .with_component_as("InGtC", echo("Emissions|CO2", "GtC / yr", "A"))
        .with_component_as("InGtCO2", echo("Emissions|CO2", "GtCO2 / yr", "B"))
        .with_prescribed_variable("Emissions|CO2", scalar_data(1000.0, "MtCO2 / yr"))
        .build()
        .unwrap();
    model.run();

    let (unit, grid, value) = stored(&model, "Emissions|CO2");
    assert_eq!(
        (unit.as_str(), grid, value),
        ("MtCO2 / yr", GridType::Scalar, 1000.0)
    );
    let (_, _, a) = stored(&model, "A");
    assert!((a - 12.0 / 44.0).abs() < 1e-12, "{a}");
    assert_eq!(stored(&model, "B").2, 1.0);
}

#[test]
fn schema_entry_sets_the_unit_of_prescribed_data() {
    let schema = VariableSchema::new()
        .variable("Emissions|CO2", "GtCO2 / yr")
        .variable("A", "GtC / yr");
    let model = builder()
        .with_schema(schema)
        .with_component_as("InGtC", echo("Emissions|CO2", "GtC / yr", "A"))
        .with_prescribed_variable("Emissions|CO2", scalar_data(1000.0, "MtCO2 / yr"))
        .build()
        .unwrap();

    let (unit, _, value) = stored(&model, "Emissions|CO2");
    assert_eq!((unit.as_str(), value), ("GtCO2 / yr", 1.0));
}

#[test]
fn finer_write_aggregates_to_a_coarser_schema_store() {
    let schema = VariableSchema::new().variable("T", "K");
    let mut model = builder()
        .with_schema(schema)
        .with_component(Arc::new(FourBoxWriter))
        .build()
        .unwrap();
    model.run();

    let (_, grid, value) = stored(&model, "T");
    assert_eq!((grid, value), (GridType::Scalar, global_mean()));
}

#[test]
fn finer_store_aggregates_for_a_coarser_reader_without_a_schema() {
    let mut model = builder()
        .with_component(Arc::new(FourBoxWriter))
        .with_component(echo("T", "K", "Global"))
        .build()
        .unwrap();
    model.run();

    assert_eq!(stored(&model, "T").1, GridType::FourBox);
    assert_eq!(stored(&model, "Global").2, global_mean());
}

#[test]
fn finer_prescribed_data_aggregates_for_a_coarser_reader() {
    let mut model = builder()
        .with_component(echo("T", "K", "Global"))
        .with_prescribed_variable("T", four_box_data())
        .build()
        .unwrap();
    model.run();

    assert_eq!(stored(&model, "T").1, GridType::FourBox);
    assert_eq!(stored(&model, "Global").2, global_mean());
}

#[test]
fn coarser_store_for_a_finer_reader_is_a_build_error() {
    let err = builder()
        .with_component(Arc::new(ScalarWriter))
        .with_component(Arc::new(FourBoxReader))
        .build()
        .unwrap_err();

    assert_grid_mismatch(
        err,
        "FourBoxReader",
        "reads",
        "its producer ScalarWriter declares",
    );
}

#[test]
fn coarser_prescribed_data_for_a_finer_reader_is_a_build_error() {
    let err = builder()
        .with_component(Arc::new(FourBoxReader))
        .with_prescribed_variable("T", scalar_data(1.0, "K"))
        .build()
        .unwrap_err();

    assert_grid_mismatch(err, "FourBoxReader", "reads", "the prescribed data is");
}

#[test]
fn coarser_write_to_a_finer_schema_store_is_a_build_error() {
    let schema = VariableSchema::new().variable_with_grid("T", "K", GridType::FourBox);
    let err = builder()
        .with_schema(schema)
        .with_component(Arc::new(ScalarWriter))
        .build()
        .unwrap_err();

    assert_grid_mismatch(err, "ScalarWriter", "writes", "the schema declares");
}

#[test]
fn read_converts_the_unit_before_aggregating_the_grid() {
    let mut model = builder()
        .with_component(echo("T", "mK", "Global"))
        .with_prescribed_variable("T", four_box_data())
        .build()
        .unwrap();
    model.run();

    let expected = FourBoxGrid::magicc_standard().aggregate_global(&BOXES.map(|v| v * 1000.0));
    assert_eq!(stored(&model, "Global").2, expected);
}
