//! MAGICC component parameters
//!
//! This module contains parameter structures for all MAGICC components.
//! Each parameter struct provides sensible defaults matching MAGICC7's
//! default configuration.

mod aerosol;
mod ch4_chemistry;
mod climate_udeb;
mod co2_budget;
mod halocarbon;
mod n2o_chemistry;
mod ocean_carbon;
mod ozone_forcing;
mod terrestrial_carbon;

pub use aerosol::{AerosolDirectParameters, AerosolIndirectParameters};
pub use ch4_chemistry::CH4ChemistryParameters;
pub use climate_udeb::ClimateUDEBParameters;
pub use co2_budget::CO2BudgetParameters;
pub use halocarbon::{HalocarbonParameters, HalocarbonSpecies};
pub use n2o_chemistry::N2OChemistryParameters;
pub use ocean_carbon::OceanCarbonParameters;
pub use ozone_forcing::OzoneForcingParameters;
pub use terrestrial_carbon::TerrestrialCarbonParameters;
