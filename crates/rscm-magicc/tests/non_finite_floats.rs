//! Every MAGICC component keeps NaN and infinities in its parameters and state through serde.

use rscm_core::component::Component;
use rscm_core::non_finite::testing::{assert_components_round_trip, CORE_COMPONENT_TYPES};
use rscm_magicc::carbon::{CO2Budget, OceanCarbon, Permafrost, TerrestrialCarbon};
use rscm_magicc::chemistry::{CH4Chemistry, CH4Oxidation, HalocarbonChemistry, N2OChemistry};
use rscm_magicc::climate::{ClimateUDEB, SeaLevelRise};
use rscm_magicc::forcing::{ForcingAggregation, GHGForcing, NitrateCloudForcing, OzoneForcing};
use rscm_magicc::parameters::{ForcingAggregationParameters, NitrateCloudForcingParameters};

#[test]
fn every_component_keeps_non_finite_floats() {
    let components: Vec<Box<dyn Component>> = vec![
        Box::new(CO2Budget::default()),
        Box::new(OceanCarbon::default()),
        Box::new(Permafrost::default()),
        Box::new(TerrestrialCarbon::default()),
        Box::new(CH4Chemistry::default()),
        Box::new(CH4Oxidation::default()),
        Box::new(HalocarbonChemistry::default()),
        Box::new(N2OChemistry::default()),
        Box::new(ClimateUDEB::default()),
        Box::new(SeaLevelRise::default()),
        Box::new(
            ForcingAggregation::from_parameters(ForcingAggregationParameters::default()).unwrap(),
        ),
        Box::new(GHGForcing::default()),
        Box::new(OzoneForcing::default()),
        Box::new(
            NitrateCloudForcing::from_parameters(NitrateCloudForcingParameters::default()).unwrap(),
        ),
    ];
    let upstream = [
        &CORE_COMPONENT_TYPES[..],
        &["CO2ERF", "CarbonCycle", "FourBoxOceanHeatUptake"],
    ]
    .concat();
    assert_components_round_trip(&components, &upstream);
}
