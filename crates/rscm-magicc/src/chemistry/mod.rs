//! Chemistry domain components for MAGICC
//!
//! This module contains atmospheric chemistry components:
//!
//! - `CH4Chemistry`: Methane chemistry with OH feedback (Prather method)
//! - `N2OChemistry`: Nitrous oxide chemistry with stratospheric delay
//! - `HalocarbonChemistry`: F-gases and Montreal Protocol gases with exponential decay

mod ch4;
mod halocarbon;
mod n2o;

pub use ch4::CH4Chemistry;
pub use halocarbon::HalocarbonChemistry;
pub use n2o::N2OChemistry;
