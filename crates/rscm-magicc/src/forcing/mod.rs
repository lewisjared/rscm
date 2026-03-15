//! Forcing domain components for MAGICC
//!
//! This module contains radiative forcing calculation components:
//!
//! - `GhgForcing`: Well-mixed GHG forcing (CO2, CH4, N2O) with IPCCTAR/Etminan methods
//! - `OzoneForcing`: Stratospheric and tropospheric ozone forcing
//! - `AerosolDirect`: Direct aerosol radiative effects (SO2, BC, OC)
//! - `AerosolIndirect`: Indirect aerosol effects (cloud-albedo)

mod aerosol_direct;
mod aerosol_indirect;
mod ghg;
mod ozone;

pub use aerosol_direct::AerosolDirect;
pub use aerosol_indirect::AerosolIndirect;
pub use ghg::GhgForcing;
pub use ozone::OzoneForcing;
