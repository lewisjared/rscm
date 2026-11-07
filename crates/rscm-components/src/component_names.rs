//! ComponentIO declares names as literals, so these tests keep them equal to `rscm_core::names`.

use rscm_core::names::*;

fn declares(definitions: Vec<rscm_core::component::VariableDefinition>, expected: &[&str]) {
    let names: Vec<String> = definitions.into_iter().map(|d| d.name).collect();
    for name in expected {
        assert!(
            names.iter().any(|n| n == name),
            "{name} missing from {names:?}"
        );
    }
}

#[test]
fn carbon_cycle_names() {
    declares(
        crate::components::CarbonCycle::generated_definitions(),
        &[
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            CUMULATIVE_CARBON_UPTAKE_LAND,
            CUMULATIVE_EMISSIONS_CO2,
            EMISSIONS_CO2,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn co2_erf_names() {
    declares(
        crate::components::CO2ERF::generated_definitions(),
        &[ATMOSPHERIC_CONCENTRATIONS_CO2, ERF_CO2],
    );
}

#[test]
fn four_box_ocean_heat_uptake_names() {
    declares(
        crate::components::FourBoxOceanHeatUptake::generated_definitions(),
        &[ERF, HEAT_UPTAKE],
    );
}

#[test]
fn ocean_surface_partial_pressure_names() {
    declares(
        crate::components::ocean_carbon_cycle::OceanSurfacePartialPressure::generated_definitions(),
        &[OCEAN_SURFACE_PARTIAL_PRESSURE_CO2, SEA_SURFACE_TEMPERATURE],
    );
}
