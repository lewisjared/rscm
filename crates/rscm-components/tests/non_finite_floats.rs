//! Every rscm-components component keeps NaN and infinities in its parameters and state through serde.

use rscm_components::*;
use rscm_core::component::Component;
use rscm_core::non_finite::testing::{assert_components_round_trip, CORE_COMPONENT_TYPES};

#[test]
fn every_component_keeps_non_finite_floats() {
    let components: Vec<Box<dyn Component>> = vec![
        Box::new(CO2ERF::from_parameters(CO2ERFParameters {
            erf_2xco2: 3.9,
            co2_pi: 278.0,
        })),
        Box::new(CarbonCycle::from_parameters(CarbonCycleParameters {
            tau: 20.0,
            co2_pi: 278.0,
            alpha_temperature: 0.05,
        })),
        Box::new(FourBoxOceanHeatUptake::from_parameters(
            FourBoxOceanHeatUptakeParameters::default(),
        )),
    ];
    assert_components_round_trip(&components, &CORE_COMPONENT_TYPES);
}
