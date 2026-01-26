//! Carbon domain components for MAGICC
//!
//! This module contains carbon cycle components:
//!
//! - `TerrestrialCarbon`: 4-pool terrestrial carbon cycle with CO2 fertilization
//!   and temperature feedbacks
//! - `OceanCarbon`: IRF-based ocean carbon cycle with air-sea exchange and
//!   temperature feedback on solubility
//! - `CO2Budget`: Mass balance integrator that closes the carbon cycle

mod budget;
mod ocean;
mod terrestrial;

pub use budget::CO2Budget;
pub use ocean::OceanCarbon;
pub use terrestrial::TerrestrialCarbon;
