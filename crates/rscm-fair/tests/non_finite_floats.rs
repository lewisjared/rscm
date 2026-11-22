//! Every FaIR component keeps NaN and infinities in its parameters and state through serde.

use rscm_core::component::Component;
use rscm_core::non_finite::testing::{assert_components_round_trip, CORE_COMPONENT_TYPES};
use rscm_fair::EnergyBalance;
use serde_json::{json, Value};

fn ghg(concentration: &str) -> Value {
    json!({
        "concentration": concentration,
        "concentration_unit": "ppb",
        "forcing": format!("Effective Radiative Forcing|{concentration}"),
        "reference_concentration": 270.0,
        "baseline_concentration": 270.0,
        "radiative_efficiency": 0.0003,
        "temperature_feedback": -0.01,
    })
}

fn gas(driver: &str) -> Value {
    json!({
        "driver": driver,
        "emissions": ["Emissions|CH4"],
        "inverse_emissions": "Inverse Emissions|CH4",
        "emissions_unit": "Mt CH4/yr",
        "amount_unit": "Mt CH4",
        "concentration": "Atmospheric Concentrations|CH4",
        "concentration_unit": "ppb",
        "partition_fraction": [0.6, 0.4],
        "unperturbed_lifetime": [8.25, 100.0],
        "baseline_concentration": 729.2,
        "baseline_emissions": 19.0,
        "concentration_per_emission": 0.35,
        "iirf_0": 8.25,
        "iirf_airborne": 0.00032,
        "iirf_uptake": 0.0,
        "iirf_temperature": -0.3,
        "g0": 0.36,
        "g1": 8.25,
    })
}

fn driver(driver: &str) -> Value {
    json!({
        "variable": "Emissions|SOx",
        "unit": "Mt SO2/yr",
        "driver": driver,
        "baseline": 2.4,
        "shape": 0.01,
    })
}

/// Built from its type, which links rscm-fair so that its components register.
fn energy_balance() -> Box<dyn Component> {
    let parameters = serde_json::from_value(json!({
        "ocean_heat_capacity": [8.0, 14.0, 100.0],
        "ocean_heat_transfer": [1.1, 1.6, 0.9],
        "deep_ocean_efficacy": 1.1,
    }))
    .unwrap();
    Box::new(EnergyBalance::from_parameters(parameters).unwrap())
}

#[test]
fn every_component_keeps_non_finite_floats() {
    let components: Vec<Box<dyn Component>> = [
        json!({"type": "fair.GHGForcing", "parameters": {
            "co2": ghg("CO2"), "ch4": ghg("CH4"), "n2o": ghg("N2O"), "minor": [ghg("CFC11")],
        }}),
        json!({"type": "fair.GasCycle", "parameters": {
            "species": [gas("emissions"), gas("concentration")],
        }}),
        json!({"type": "fair.CH4Lifetime", "parameters": {
            "emissions": [{"variable": "Emissions|NOx", "unit": "Mt NO2/yr",
                           "baseline": 19.4, "sensitivity": -0.0004}],
            "concentrations": [{"variable": "Atmospheric Concentrations|N2O", "unit": "ppb",
                                "baseline": 273.0, "sensitivity": -0.0005}],
            "temperature_sensitivity": -0.04,
        }}),
        json!({"type": "fair.EESC", "parameters": {
            "halogens": [{"concentration": "Atmospheric Concentrations|CFC11", "unit": "ppt",
                          "cl_atoms": 3.0, "br_atoms": 0.0, "fractional_release": 0.47}],
            "cfc11_fractional_release": 0.47,
        }}),
        json!({"type": "fair.LinearForcing", "parameters": {
            "forcing": "Effective Radiative Forcing|Aerosols|Direct Effect",
            "drivers": [{"variable": "Emissions|SOx", "unit": "Mt SO2/yr", "driver": "emissions",
                         "baseline": 2.4, "radiative_efficiency": -0.003},
                        {"variable": "Cumulative Emissions|CO2|AFOLU", "unit": "GtC",
                         "driver": "cumulative_emissions", "radiative_efficiency": -0.0002}],
            "tropospheric_adjustment": 0.1,
        }}),
        json!({"type": "fair.AerosolCloud", "parameters": {
            "forcing": "Effective Radiative Forcing|Aerosols|Indirect Effect",
            "drivers": [driver("emissions"), driver("concentration")],
            "scale": -2.1,
        }}),
        json!({"type": "fair.ForcingAggregation", "parameters": {
            "agents": [{"forcing": "Effective Radiative Forcing|CO2", "category": "greenhouse_gas",
                        "efficacy": 1.0, "temperature_feedback": 0.0}],
        }}),
    ]
    .into_iter()
    .map(|value| serde_json::from_value(value).unwrap())
    .chain([energy_balance()])
    .collect();
    let upstream = [
        &CORE_COMPONENT_TYPES[..],
        &["CO2ERF", "CarbonCycle", "FourBoxOceanHeatUptake"],
    ]
    .concat();
    assert_components_round_trip(&components, &upstream);
}
