//! Variable names shared between components, crates and the Python package.
//!
//! A name is `Quantity|Species|Sector`, then any further breakdown, as in `Emissions|CO2|Fossil`.
//! A child segment is always a part of its parent.
//! Other qualifiers lead the quantity segment, as in `Prescribed Atmospheric Concentrations|CO2`.
//! Names follow gcages and MAGICC7 where they overlap.
//!
//! Python reads the same constants from `rscm.names`.

macro_rules! names {
    ($($(#[doc = $doc:literal])* $id:ident = $value:literal;)*) => {
        $($(#[doc = $doc])* pub const $id: &str = $value;)*

        /// Every constant in this module, as (identifier, name) pairs.
        pub const ALL: &[(&str, &str)] = &[$((stringify!($id), $id)),*];
    };
}

names! {
    /// Quantity segment of emissions.
    EMISSIONS = "Emissions";
    /// Quantity segment of atmospheric concentrations.
    ATMOSPHERIC_CONCENTRATIONS = "Atmospheric Concentrations";
    /// Total effective radiative forcing, and the parent of each agent's forcing.
    ERF = "Effective Radiative Forcing";

    /// Global mean surface air temperature anomaly.
    SURFACE_AIR_TEMPERATURE_CHANGE = "Surface Air Temperature Change";
    /// Annual mean surface air temperature anomaly, where substeps resolve it.
    ANNUAL_MEAN_SURFACE_AIR_TEMPERATURE_CHANGE = "Annual Mean Surface Air Temperature Change";
    /// Sea surface temperature anomaly.
    SEA_SURFACE_TEMPERATURE = "Sea Surface Temperature";
    /// Parent of the ocean layer temperatures, with the surface as layer 1.
    OCEAN_TEMPERATURE = "Ocean Temperature";
    /// Heat flux into the ocean.
    HEAT_UPTAKE = "Heat Uptake";
    /// Global ocean heat content anomaly.
    OCEAN_HEAT_CONTENT = "Ocean Heat Content";
    /// Top of atmosphere energy imbalance.
    NET_ENERGY_IMBALANCE = "Net Energy Imbalance";
    /// Equilibrium climate sensitivity.
    EQUILIBRIUM_CLIMATE_SENSITIVITY = "Equilibrium Climate Sensitivity";

    /// Total forcing after each agent's efficacy.
    EFFICACY_ADJUSTED_ERF = "Efficacy Adjusted Effective Radiative Forcing";
    /// Forcing of doubled CO2.
    ERF_2XCO2 = "2xCO2 Effective Radiative Forcing";
    /// Forcing of the stochastic energy balance term.
    STOCHASTIC_FORCING = "Stochastic Forcing";
    /// CO2 forcing.
    ERF_CO2 = "Effective Radiative Forcing|CO2";
    /// CH4 forcing.
    ERF_CH4 = "Effective Radiative Forcing|CH4";
    /// N2O forcing.
    ERF_N2O = "Effective Radiative Forcing|N2O";
    /// Forcing of the greenhouse gases.
    ERF_GREENHOUSE_GASES = "Effective Radiative Forcing|Greenhouse Gases";
    /// Forcing of everything but natural agents.
    ERF_ANTHROPOGENIC = "Effective Radiative Forcing|Anthropogenic";
    /// F-gas forcing.
    ERF_F_GASES = "Effective Radiative Forcing|F-Gases";
    /// Montreal Protocol gas forcing.
    ERF_MONTREAL_PROTOCOL_HALOGEN_GASES = "Effective Radiative Forcing|Montreal Protocol Halogen Gases";
    /// Aerosol forcing.
    ERF_AEROSOLS = "Effective Radiative Forcing|Aerosols";
    /// Aerosol-radiation interaction forcing.
    ERF_AEROSOLS_DIRECT_EFFECT = "Effective Radiative Forcing|Aerosols|Direct Effect";
    /// Aerosol-cloud interaction forcing.
    ERF_AEROSOLS_INDIRECT_EFFECT = "Effective Radiative Forcing|Aerosols|Indirect Effect";
    /// Ozone forcing.
    ERF_OZONE = "Effective Radiative Forcing|Ozone";
    /// Stratospheric ozone forcing.
    ERF_OZONE_STRATOSPHERIC = "Effective Radiative Forcing|Ozone|Stratospheric";
    /// Tropospheric ozone forcing.
    ERF_OZONE_TROPOSPHERIC = "Effective Radiative Forcing|Ozone|Tropospheric";
    /// Forcing of the ozone that N2O produces.
    ERF_OZONE_N2O_FEEDBACK = "Effective Radiative Forcing|Ozone|N2O Feedback";
    /// Forcing of the ozone response to warming.
    ERF_OZONE_TEMPERATURE_FEEDBACK = "Effective Radiative Forcing|Ozone|Temperature Feedback";
    /// Forcing of stratospheric water vapour from CH4 oxidation.
    ERF_CH4_OXIDATION_STRATOSPHERIC_H2O = "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O";
    /// Forcing of black carbon on snow.
    ERF_BLACK_CARBON_ON_SNOW = "Effective Radiative Forcing|Black Carbon on Snow";
    /// Land surface albedo forcing.
    ERF_LAND_USE_CHANGE = "Effective Radiative Forcing|Land-use Change";
    /// Contrail forcing.
    ERF_AVIATION_CONTRAIL = "Effective Radiative Forcing|Aviation|Contrail";
    /// Solar forcing.
    ERF_SOLAR = "Effective Radiative Forcing|Solar";
    /// Volcanic forcing.
    ERF_VOLCANIC = "Effective Radiative Forcing|Volcanic";

    /// Atmospheric CO2 concentration.
    ATMOSPHERIC_CONCENTRATIONS_CO2 = "Atmospheric Concentrations|CO2";
    /// Atmospheric CH4 concentration.
    ATMOSPHERIC_CONCENTRATIONS_CH4 = "Atmospheric Concentrations|CH4";
    /// Atmospheric N2O concentration.
    ATMOSPHERIC_CONCENTRATIONS_N2O = "Atmospheric Concentrations|N2O";
    /// Atmospheric CO2 at the end of the ocean carbon cycle's last substep.
    OCEAN_CARBON_ATMOSPHERIC_CONCENTRATIONS_CO2 = "Ocean Carbon Atmospheric Concentrations|CO2";
    /// Equivalent effective stratospheric chlorine.
    EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE = "Equivalent Effective Stratospheric Chlorine";

    /// Total CO2 emissions, the parent of the fossil and biosphere sectors.
    EMISSIONS_CO2 = "Emissions|CO2";
    /// Fossil and industrial CO2 emissions.
    EMISSIONS_CO2_FOSSIL = "Emissions|CO2|Fossil";
    /// Land-use change CO2 emissions.
    EMISSIONS_CO2_BIOSPHERE = "Emissions|CO2|Biosphere";
    /// CO2 from the oxidation of fossil CH4.
    EMISSIONS_CO2_CH4_OXIDATION = "Emissions|CO2|CH4 Oxidation";
    /// CO2 from thawing permafrost.
    EMISSIONS_CO2_PERMAFROST = "Emissions|CO2|Permafrost";
    /// Total anthropogenic CH4 emissions.
    EMISSIONS_CH4 = "Emissions|CH4";
    /// Fossil and industrial CH4 emissions.
    EMISSIONS_CH4_FOSSIL = "Emissions|CH4|Fossil";
    /// Land-use CH4 emissions.
    EMISSIONS_CH4_BIOSPHERE = "Emissions|CH4|Biosphere";
    /// CH4 from thawing permafrost.
    EMISSIONS_CH4_PERMAFROST = "Emissions|CH4|Permafrost";
    /// N2O emissions.
    EMISSIONS_N2O = "Emissions|N2O";
    /// NOx emissions.
    EMISSIONS_NOX = "Emissions|NOx";
    /// Sulfur emissions.
    EMISSIONS_SOX = "Emissions|SOx";
    /// CO emissions.
    EMISSIONS_CO = "Emissions|CO";
    /// Non-methane volatile organic compound emissions.
    EMISSIONS_NMVOC = "Emissions|NMVOC";
    /// Black carbon emissions.
    EMISSIONS_BC = "Emissions|BC";
    /// Organic carbon emissions.
    EMISSIONS_OC = "Emissions|OC";
    /// Ammonia emissions.
    EMISSIONS_NH3 = "Emissions|NH3";
    /// CO2 emitted to the atmosphere net of the land and ocean sinks.
    NET_EMISSIONS_CO2 = "Net Emissions|CO2";
    /// CH4 emissions inferred from prescribed concentrations.
    INVERSE_EMISSIONS_CH4 = "Inverse Emissions|CH4";
    /// Cumulative CO2 emissions.
    CUMULATIVE_EMISSIONS_CO2 = "Cumulative Emissions|CO2";

    /// Net CO2 flux from the atmosphere into the land, positive for uptake.
    NET_ATMOSPHERE_TO_LAND_FLUX_CO2 = "Net Atmosphere to Land Flux|CO2";
    /// Net CO2 flux from the atmosphere into the ocean, positive for uptake.
    NET_ATMOSPHERE_TO_OCEAN_FLUX_CO2 = "Net Atmosphere to Ocean Flux|CO2";
    /// Carbon taken up by the land since the start.
    CUMULATIVE_CARBON_UPTAKE_LAND = "Cumulative Carbon Uptake|Land";
    /// Carbon taken up by the ocean since the start.
    CUMULATIVE_CARBON_UPTAKE_OCEAN = "Cumulative Carbon Uptake|Ocean";
    /// Partial pressure of CO2 at the ocean surface.
    OCEAN_SURFACE_PARTIAL_PRESSURE_CO2 = "Ocean Surface Partial Pressure|CO2";
    /// Fraction of CO2 sources remaining in the atmosphere.
    AIRBORNE_FRACTION_CO2 = "Airborne Fraction|CO2";
    /// Plant carbon pool.
    CARBON_POOL_PLANT = "Carbon Pool|Plant";
    /// Detritus carbon pool.
    CARBON_POOL_DETRITUS = "Carbon Pool|Detritus";
    /// Soil carbon pool.
    CARBON_POOL_SOIL = "Carbon Pool|Soil";
    /// Land net primary production.
    NET_PRIMARY_PRODUCTION = "Net Primary Production";
    /// Land respiration.
    RESPIRATION_LAND = "Respiration|Land";
    /// Gross CO2 emissions from deforestation.
    EMISSIONS_CO2_GROSS_DEFORESTATION = "Emissions|CO2|Gross Deforestation";
    /// Carbon taken up by regrowing land.
    CARBON_FLUX_REGROWTH = "Carbon Flux|Regrowth";
    /// Nitrogen deposited on land.
    NITROGEN_FLUX_DEPOSITION = "Nitrogen Flux|Deposition";
    /// Nitrogen fixed on land.
    NITROGEN_FLUX_FIXATION = "Nitrogen Flux|Fixation";

    /// Atmospheric CH4 lifetime.
    LIFETIME_CH4 = "Lifetime|CH4";
    /// Atmospheric CH4 lifetime against OH.
    LIFETIME_CH4_OH = "Lifetime|CH4|OH";
    /// CH4 lifetime that closes the budget of prescribed concentrations.
    EFFECTIVE_LIFETIME_CH4 = "Effective Lifetime|CH4";
    /// Atmospheric N2O lifetime.
    LIFETIME_N2O = "Lifetime|N2O";
    /// CH4 lifetime relative to its preindustrial value.
    LIFETIME_SCALING_CH4 = "Lifetime Scaling|CH4";
    /// CH4 in the atmosphere from fossil sources.
    ATMOSPHERIC_BURDEN_CH4_FOSSIL = "Atmospheric Burden|CH4|Fossil";
    /// CH4 in the atmosphere from land-use sources.
    ATMOSPHERIC_BURDEN_CH4_BIOSPHERE = "Atmospheric Burden|CH4|Biosphere";
    /// CH4 in the atmosphere from natural sources.
    ATMOSPHERIC_BURDEN_CH4_NATURAL = "Atmospheric Burden|CH4|Natural";
    /// CH4 in the atmosphere from permafrost.
    ATMOSPHERIC_BURDEN_CH4_PERMAFROST = "Atmospheric Burden|CH4|Permafrost";

    /// Total sea level rise.
    SEA_LEVEL_RISE = "Sea Level Rise";
    /// Sea level rise from ocean thermal expansion.
    SEA_LEVEL_RISE_THERMAL_EXPANSION = "Sea Level Rise|Thermal Expansion";
}

/// Joins name segments with `|`, as in `child(EMISSIONS, "CO2")` for `Emissions|CO2`.
pub fn child(parent: &str, part: &str) -> String {
    format!("{parent}|{part}")
}

/// Name of the prescribed input that drives `name`, as in `Prescribed Atmospheric Concentrations|CO2`.
pub fn prescribed(name: &str) -> String {
    format!("Prescribed {name}")
}

/// Name of the temperature of ocean `layer`, counting the surface as layer 1.
pub fn ocean_layer_temperature(layer: usize) -> String {
    format!("{OCEAN_TEMPERATURE}|Layer {layer}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn names_are_unique() {
        let values: HashSet<_> = ALL.iter().map(|(_, name)| *name).collect();
        assert_eq!(values.len(), ALL.len());
    }

    #[test]
    fn helpers_join_segments() {
        assert_eq!(child(EMISSIONS, "CO2"), EMISSIONS_CO2);
        assert_eq!(
            prescribed(ATMOSPHERIC_CONCENTRATIONS_CO2),
            "Prescribed Atmospheric Concentrations|CO2"
        );
        assert_eq!(ocean_layer_temperature(2), "Ocean Temperature|Layer 2");
    }
}
