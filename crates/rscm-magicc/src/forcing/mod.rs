//! Forcing domain components for MAGICC
//!
//! This module contains radiative forcing calculation components:
//!
//! - `OzoneForcing`: Stratospheric and tropospheric ozone forcing
//! - `AerosolDirect`: Direct aerosol radiative effects (SO2, BC, OC)
//! - `AerosolIndirect`: Indirect aerosol effects (cloud-albedo)
//!
//! TODO: GHG forcing components (IPCCTAR, OLBL) to be added

mod aerosol_direct;
mod aerosol_indirect;
mod ozone;

pub use aerosol_direct::AerosolDirect;
pub use aerosol_indirect::AerosolIndirect;
pub use ozone::OzoneForcing;
