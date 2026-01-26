//! Basic model tests: step, dot, serialisation.

use crate::example_components::{TestComponent, TestComponentParameters};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::ModelBuilder;
use crate::spatial::{ScalarGrid, ScalarRegion};
use crate::timeseries::{FloatValue, TimeAxis, Timeseries};
use is_close::is_close;
use numpy::array;
use numpy::ndarray::{Array, Axis};
use std::iter::zip;
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
fn step() {
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
    let mut model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    assert_eq!(model.current_time(), 2020.0);
    model.step();
    model.step();
    assert_eq!(model.current_time(), 2022.0);
    model.run();
    assert!(model.finished());

    let concentrations = model
        .timeseries()
        .get_data("Concentrations|CO2")
        .and_then(|data| data.as_scalar())
        .unwrap();

    // The first value for an endogenous timeseries without a y0 value is NaN.
    // This is because the values in the timeseries represents the state at the start
    // of a time step.
    // Since the values from t-1 aren't known we can't solve for y0
    assert!(concentrations.at(0, ScalarRegion::Global).unwrap().is_nan());
    let mut iter = concentrations.values().into_iter();
    iter.next(); // Skip the first value
    assert!(iter.all(|x| !x.is_nan()));
}

#[test]
fn dot() {
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
    let model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    let exp = r#"digraph {
    0 [ label = "NullComponent"]
    1 [ label = "TestComponent { parameters: TestComponentParameters { conversion_factor: 0.5 } }"]
    0 -> 1 [ label = ""]
}
"#;

    let res = format!("{:?}", model.as_dot());
    assert_eq!(res, exp);
}

#[test]
fn serialise_and_deserialise_model() {
    use crate::model::Model;

    let mut model = ModelBuilder::new()
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters {
                conversion_factor: 0.5,
            },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions())
        .build()
        .unwrap();

    model.step();

    let serialised = serde_json::to_string_pretty(&model).unwrap();
    println!("Pretty JSON");
    println!("{}", serialised);
    let serialised = toml::to_string(&model).unwrap();
    println!("TOML");
    println!("{}", serialised);

    let expected = r#"initial_node = 0
time_index = 1

[components]
node_holes = []
edge_property = "directed"
edges = [[0, 1, { name = "", unit = "", requirement_type = "EmptyLink", grid_type = "Scalar" }]]

[[components.nodes]]
type = "NullComponent"

[[components.nodes]]
type = "TestComponent"

[components.nodes.parameters]
conversion_factor = 0.5

[[collection.timeseries]]
name = "Concentrations|CO2"
variable_type = "Endogenous"

[collection.timeseries.data.Scalar]
units = "ppm"
latest = 1
interpolation_strategy = "Linear"

[collection.timeseries.data.Scalar.values]
v = 1
dim = [5, 1]
data = [nan, 5.0, nan, nan, nan]

[collection.timeseries.data.Scalar.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[[collection.timeseries]]
name = "Emissions|CO2"
variable_type = "Exogenous"

[collection.timeseries.data.Scalar]
units = "GtC / yr"
latest = 4
interpolation_strategy = "Previous"

[collection.timeseries.data.Scalar.values]
v = 1
dim = [5, 1]
data = [10.0, 10.0, 10.0, 10.0, 10.0]

[collection.timeseries.data.Scalar.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]
"#;

    assert_eq!(serialised, expected);

    let deserialised = toml::from_str::<Model>(&serialised).unwrap();

    assert!(zip(
        model
            .timeseries()
            .get_data("Emissions|CO2")
            .and_then(|data| data.as_scalar())
            .unwrap()
            .values(),
        deserialised
            .timeseries()
            .get_data("Emissions|CO2")
            .and_then(|data| data.as_scalar())
            .unwrap()
            .values()
    )
    .all(|(x0, x1)| { is_close!(*x0, *x1) || (x0.is_nan() && x0.is_nan()) }));

    assert_eq!(model.current_time_bounds(), (2021.0, 2022.0));
    assert_eq!(deserialised.current_time_bounds(), (2021.0, 2022.0));
}
