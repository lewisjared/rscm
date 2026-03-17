//! Climate domain components for MAGICC
//!
//! This module contains climate model components implementing MAGICC-style
//! parameterisations and physics.
//!
//! - [`ClimateUDEB`]: 4-box upwelling-diffusion energy balance model with
//!   50-layer ocean
//! - [`lamcalc`]: Iterative feedback parameter solver

pub mod lamcalc;
pub(crate) mod state;
mod udeb;

pub use state::ClimateUDEBState;
pub use udeb::ClimateUDEB;
