//! Every rscm-core component keeps NaN and infinities in its parameters and state through serde.

use rscm_core::component::Component;
use rscm_core::non_finite::testing::assert_components_round_trip;
use serde_json::json;

#[test]
fn every_component_keeps_non_finite_floats() {
    // TestComponent is private to the crate, so both are built from JSON.
    let components: Vec<Box<dyn Component>> = [
        json!({"type": "TestComponent", "parameters": {"conversion_factor": 1.5}}),
        json!({
            "type": "AggregateComponent",
            "aggregate_name": "Total",
            "unit": "W/m^2",
            "grid": "scalar",
            "operation": {"weighted_sum": [0.5, 0.25]},
            "contributors": ["A", "B"],
        }),
    ]
    .into_iter()
    .map(|value| serde_json::from_value(value).unwrap())
    .collect();
    assert_components_round_trip(&components, &["PythonComponent"]);
}
