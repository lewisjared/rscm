//! Forcing domain components for MAGICC
//!
//! This module contains radiative forcing calculation components:
//!
//! - `GHGForcing`: Well-mixed GHG forcing (CO2, CH4, N2O) with IPCCTAR/Etminan methods
//! - `OzoneForcing`: Stratospheric and tropospheric ozone forcing
//! - `NitrateCloudForcing`: Nitrate direct and cloud albedo forcing as in MAGICC7
//! - `ForcingAggregation`: Regional, efficacy-adjusted total forcing

mod aggregation;
mod ghg;
mod nitrate_cloud;
mod ozone;

pub use aggregation::ForcingAggregation;
pub use ghg::GHGForcing;
pub use nitrate_cloud::{
    optical_thickness_name, Drivers, EmissionsLookup, NitrateCloud, NitrateCloudForcing,
    OpticalThicknessLookup,
};
pub use ozone::OzoneForcing;
