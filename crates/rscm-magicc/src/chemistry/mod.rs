//! Chemistry domain components for MAGICC
//!
//! This module contains atmospheric chemistry components:
//!
//! - `CH4Chemistry`: Methane chemistry with OH feedback (Prather method)
//! - `CH4Oxidation`: CO2 released by oxidation of fossil methane
//! - `N2OChemistry`: Nitrous oxide chemistry with stratospheric delay
//! - `HalocarbonChemistry`: F-gases and Montreal Protocol gases with exponential decay

mod ch4;
mod ch4_oxidation;
mod halocarbon;
mod n2o;

pub use ch4::CH4Chemistry;
pub use ch4_oxidation::CH4Oxidation;
pub use halocarbon::HalocarbonChemistry;
pub use n2o::N2OChemistry;
