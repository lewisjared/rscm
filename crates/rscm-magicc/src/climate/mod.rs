//! Climate domain components for MAGICC
//!
//! This module contains climate model components implementing MAGICC-style
//! parameterisations and physics.
//!
//! - [`ClimateUDEB`]: four-box upwelling-diffusion energy balance model with
//!   50-layer ocean
//! - [`SeaLevelRise`]: Sea level rise from land ice
//! - [`lamcalc`]: Iterative feedback parameter solver

pub mod lamcalc;
mod sea_level;
pub(crate) mod state;
mod udeb;

pub use sea_level::{thermal_expansion_increment, SeaLevelRise, SeaLevelRiseState};
pub use state::ClimateUDEBState;
pub use udeb::ClimateUDEB;
