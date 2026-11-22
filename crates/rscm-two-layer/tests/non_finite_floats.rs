//! The two-layer component keeps NaN and infinities in its parameters and state through serde.

use rscm_core::component::Component;
use rscm_core::non_finite::testing::{assert_components_round_trip, CORE_COMPONENT_TYPES};
use rscm_two_layer::{TwoLayer, TwoLayerParameters};

#[test]
fn every_component_keeps_non_finite_floats() {
    let components: Vec<Box<dyn Component>> = vec![Box::new(
        TwoLayer::from_parameters(TwoLayerParameters {
            lambda0: 1.2,
            a: 0.0,
            deep_ocean_efficacy: 1.3,
            ocean_heat_transfer: 0.7,
            heat_capacity_surface: 8.0,
            heat_capacity_deep: 100.0,
        })
        .unwrap(),
    )];
    assert_components_round_trip(&components, &CORE_COMPONENT_TYPES);
}
