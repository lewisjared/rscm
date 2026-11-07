//! Carbon domain components for MAGICC
//!
//! This module contains carbon cycle components:
//!
//! - `TerrestrialCarbon`: 3-pool terrestrial carbon cycle with CO2 fertilisation,
//!   temperature feedbacks, MAGICC7-compatible regrowth/attribution and optional
//!   nitrogen limitation
//! - `OceanCarbon`: IRF-based ocean carbon cycle with air-sea exchange and
//!   temperature feedback on solubility
//! - `Permafrost`: Zonal band permafrost thaw releasing CO2 and CH4
//! - `CO2Budget`: Mass balance integrator that closes the carbon cycle

mod budget;
mod nitrogen;
mod ocean;
mod permafrost;
mod terrestrial;

pub use budget::CO2Budget;
pub use nitrogen::{NitrogenBudget, NitrogenDrivers, NitrogenState};
pub use ocean::{OceanCarbon, OceanCarbonState};
pub use permafrost::{Permafrost, PermafrostState, SoilBand};
pub use terrestrial::{TerrestrialCarbon, TerrestrialCarbonState};
