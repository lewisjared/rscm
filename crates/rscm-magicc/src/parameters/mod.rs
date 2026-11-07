//! MAGICC component parameters
//!
//! This module contains parameter structures for all MAGICC components.
//! Each parameter struct provides sensible defaults matching MAGICC7's
//! default configuration.

mod aerosol;
mod ch4_chemistry;
mod ch4_oxidation;
mod climate_udeb;
mod co2_budget;
mod forcing_aggregation;
mod ghg_forcing;
mod halocarbon;
mod n2o_chemistry;
mod nitrogen;
mod ocean_carbon;
mod ozone_forcing;
mod permafrost;
mod sea_level;
mod terrestrial_carbon;

pub use aerosol::{AerosolDirectParameters, AerosolIndirectParameters};
pub use ch4_chemistry::CH4ChemistryParameters;
pub use ch4_oxidation::CH4OxidationParameters;
pub use climate_udeb::{
    ClimateUdebParameters, EfficacyMethod, OceanTemperatureProfile, SubannualForcing, CP_SEAWATER,
    DIFFUSIVITY_CM2S_TO_M2YR, RHO_SEAWATER, SECONDS_PER_YEAR,
};
pub use co2_budget::CO2BudgetParameters;
pub use forcing_aggregation::{
    EfficacyClimate, ForcingAgent, ForcingAggregationParameters, ForcingTerm,
};
pub use ghg_forcing::{ForcingMethod, GhgForcingParameters};
pub use halocarbon::{FgasRegions, HalocarbonChemistryParameters, HalocarbonSpecies};
pub use n2o_chemistry::N2OChemistryParameters;
pub use nitrogen::{NitrogenParameters, PrognosticFixation};
pub use ocean_carbon::{IrfForm, OceanCarbonMethod, OceanCarbonParameters};
pub use ozone_forcing::{OzoneForcingParameters, TroposphericOzoneRegions};
pub use permafrost::{PermafrostParameters, PermafrostSoilParameters};
pub use sea_level::{
    AntarcticDischargeParameters, AntarcticRegionParameters, DischargeCaseParameters,
    GlacierParameters, GreenlandDischargeParameters, SeaLevelRiseParameters,
    SurfaceMassBalanceParameters, ThermalExpansionParameters,
};
pub use terrestrial_carbon::{FertilisationMethod, TerrestrialCarbonParameters};
