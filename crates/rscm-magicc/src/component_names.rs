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
fn co2_budget_names() {
    declares(
        crate::carbon::CO2Budget::generated_definitions(),
        &[
            AIRBORNE_FRACTION_CO2,
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            EMISSIONS_CO2_CH4_OXIDATION,
            EMISSIONS_CO2_FOSSIL,
            NET_ATMOSPHERE_TO_LAND_FLUX_CO2,
            NET_ATMOSPHERE_TO_OCEAN_FLUX_CO2,
            NET_EMISSIONS_CO2,
        ],
    );
}

#[test]
fn ocean_carbon_names() {
    declares(
        crate::carbon::OceanCarbon::generated_definitions(),
        &[
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            CUMULATIVE_CARBON_UPTAKE_OCEAN,
            EMISSIONS_CO2_CH4_OXIDATION,
            EMISSIONS_CO2_FOSSIL,
            NET_ATMOSPHERE_TO_LAND_FLUX_CO2,
            NET_ATMOSPHERE_TO_OCEAN_FLUX_CO2,
            OCEAN_CARBON_ATMOSPHERIC_CONCENTRATIONS_CO2,
            OCEAN_SURFACE_PARTIAL_PRESSURE_CO2,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn permafrost_names() {
    declares(
        crate::carbon::Permafrost::generated_definitions(),
        &[
            EMISSIONS_CH4_PERMAFROST,
            EMISSIONS_CO2_PERMAFROST,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn terrestrial_carbon_names() {
    declares(
        crate::carbon::TerrestrialCarbon::generated_definitions(),
        &[
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            CARBON_FLUX_REGROWTH,
            CARBON_POOL_DETRITUS,
            CARBON_POOL_PLANT,
            CARBON_POOL_SOIL,
            EMISSIONS_CO2_BIOSPHERE,
            EMISSIONS_CO2_GROSS_DEFORESTATION,
            NET_ATMOSPHERE_TO_LAND_FLUX_CO2,
            NET_PRIMARY_PRODUCTION,
            RESPIRATION_LAND,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn ch4_chemistry_names() {
    declares(
        crate::chemistry::CH4Chemistry::generated_definitions(),
        &[
            ATMOSPHERIC_CONCENTRATIONS_CH4,
            EFFECTIVE_LIFETIME_CH4,
            EMISSIONS_CH4,
            EMISSIONS_CO,
            EMISSIONS_NMVOC,
            EMISSIONS_NOX,
            LIFETIME_CH4,
            LIFETIME_CH4_OH,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn ch4_oxidation_names() {
    declares(
        crate::chemistry::CH4Oxidation::generated_definitions(),
        &[
            ATMOSPHERIC_BURDEN_CH4_BIOSPHERE,
            ATMOSPHERIC_BURDEN_CH4_FOSSIL,
            ATMOSPHERIC_BURDEN_CH4_NATURAL,
            ATMOSPHERIC_BURDEN_CH4_PERMAFROST,
            ATMOSPHERIC_CONCENTRATIONS_CH4,
            EMISSIONS_CH4_BIOSPHERE,
            EMISSIONS_CH4_FOSSIL,
            EMISSIONS_CO2_CH4_OXIDATION,
            LIFETIME_CH4_OH,
        ],
    );
}

#[test]
fn n2o_chemistry_names() {
    declares(
        crate::chemistry::N2OChemistry::generated_definitions(),
        &[
            ATMOSPHERIC_CONCENTRATIONS_N2O,
            EMISSIONS_N2O,
            LIFETIME_N2O,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn sea_level_rise_names() {
    declares(
        crate::climate::SeaLevelRise::generated_definitions(),
        &[
            ANNUAL_MEAN_SURFACE_AIR_TEMPERATURE_CHANGE,
            SEA_LEVEL_RISE,
            SEA_LEVEL_RISE_THERMAL_EXPANSION,
        ],
    );
}

#[test]
fn climate_udeb_names() {
    declares(
        crate::climate::ClimateUDEB::generated_definitions(),
        &[
            EQUILIBRIUM_CLIMATE_SENSITIVITY,
            ERF,
            HEAT_UPTAKE,
            OCEAN_HEAT_CONTENT,
            SEA_SURFACE_TEMPERATURE,
            SURFACE_AIR_TEMPERATURE_CHANGE,
        ],
    );
}

#[test]
fn aerosol_direct_names() {
    declares(
        crate::forcing::AerosolDirect::generated_definitions(),
        &[
            EMISSIONS_BC,
            EMISSIONS_NOX,
            EMISSIONS_OC,
            EMISSIONS_SOX,
            ERF_AEROSOLS_DIRECT_EFFECT,
        ],
    );
}

#[test]
fn aerosol_indirect_names() {
    declares(
        crate::forcing::AerosolIndirect::generated_definitions(),
        &[EMISSIONS_OC, EMISSIONS_SOX, ERF_AEROSOLS_INDIRECT_EFFECT],
    );
}

#[test]
fn ghg_forcing_names() {
    declares(
        crate::forcing::GHGForcing::generated_definitions(),
        &[
            ATMOSPHERIC_CONCENTRATIONS_CH4,
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            ATMOSPHERIC_CONCENTRATIONS_N2O,
            ERF_2XCO2,
            ERF_CH4,
            ERF_CH4_OXIDATION_STRATOSPHERIC_H2O,
            ERF_CO2,
            ERF_N2O,
        ],
    );
}
