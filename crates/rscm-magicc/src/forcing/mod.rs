//! Forcing domain components for MAGICC
//!
//! This module contains radiative forcing calculation components:
//!
//! - `GHGForcing`: Well-mixed GHG forcing (CO2, CH4, N2O) with IPCCTAR/Etminan methods
//! - `OzoneForcing`: Stratospheric and tropospheric ozone forcing
//! - `AerosolDirect`: Direct aerosol radiative effects (SO2, BC, OC)
//! - `AerosolIndirect`: Indirect aerosol effects (cloud-albedo)
//! - `NitrateCloudForcing`: Nitrate direct and cloud albedo forcing as in MAGICC7
//! - `ForcingAggregation`: Regional, efficacy-adjusted total forcing

mod aerosol_direct;
mod aerosol_indirect;
mod aggregation;
mod ghg;
mod nitrate_cloud;
mod ozone;

pub use aerosol_direct::AerosolDirect;
pub use aerosol_indirect::AerosolIndirect;
pub use aggregation::ForcingAggregation;
pub use ghg::GHGForcing;
pub use nitrate_cloud::{
    optical_thickness_name, Drivers, EmissionsLookup, NitrateCloud, NitrateCloudForcing,
    OpticalThicknessLookup,
};
pub use ozone::OzoneForcing;
