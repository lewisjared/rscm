//! Climate domain components for MAGICC
//!
//! This module contains climate model components implementing MAGICC-style
//! parameterisations and physics.
//!
//! - [`ClimateUdeb`]: four-box upwelling-diffusion energy balance model with
//!   50-layer ocean
//! - [`SeaLevelRise`]: Sea level rise from land ice
//! - [`lamcalc`]: Iterative feedback parameter solver

pub mod lamcalc;
mod sea_level;
pub(crate) mod state;
mod udeb;

pub use sea_level::{
    thermal_expansion_increment, SeaLevelRise, SeaLevelRiseState, ANNUAL_MEAN_TEMPERATURE,
    THERMAL_EXPANSION,
};
pub use state::ClimateUdebState;
pub use udeb::{ClimateUdeb, CLIMATE_SENSITIVITY, CO2_DOUBLING_ERF, REGIONAL_FORCING};
