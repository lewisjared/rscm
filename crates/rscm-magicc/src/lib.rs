//! MAGICC components for RSCM
//!
//! This crate provides components implementing the MAGICC7 reduced-complexity climate model
//! following the modular architecture documented in the MAGICC7 module specifications.
//!
//! # Module Organisation
//!
//! Components are organised by domain:
//! - `forcing`: Radiative forcing calculations (ozone, aerosol)
//! - `chemistry`: Atmospheric chemistry (CH4, N2O, halocarbons)
//! - `climate`: Climate response (UDEB energy balance model)
//! - `carbon`: Carbon cycle (terrestrial, ocean, CO2 budget)
//!
//! # Parameters
//!
//! Each component has an associated parameters struct in the `parameters` module
//! with sensible defaults matching MAGICC7 default configuration.

mod bounds;
pub mod carbon;
pub mod chemistry;
pub mod climate;
pub mod forcing;
pub mod parameters;
pub mod python;
mod reference;

/// Sectors of the emissions that components read by sector.
pub(crate) const EMISSION_SECTORS: [&str; 2] = ["Fossil", "Biosphere"];
/// NOx emissions by sector.
pub(crate) const NOX_SECTORS: [&str; 2] = ["Emissions|NOx|Fossil", "Emissions|NOx|Biosphere"];
/// NH3 emissions by sector.
pub(crate) const NH3_SECTORS: [&str; 2] = ["Emissions|NH3|Fossil", "Emissions|NH3|Biosphere"];

#[cfg(test)]
mod tests {
    use rscm_core::names::{child, EMISSIONS_NH3, EMISSIONS_NOX};

    #[test]
    fn sector_names_match() {
        for (gas, names) in [
            (EMISSIONS_NOX, super::NOX_SECTORS),
            (EMISSIONS_NH3, super::NH3_SECTORS),
        ] {
            assert_eq!(names, super::EMISSION_SECTORS.map(|s| child(gas, s)));
        }
    }
}

#[cfg(test)]
mod component_names;
