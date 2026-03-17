//! Climate domain components for MAGICC
//!
//! This module contains climate model components implementing MAGICC-style
//! parameterisations and physics.
//!
//! - [`ClimateUDEB`]: 4-box upwelling-diffusion energy balance model with
//!   50-layer ocean
//! - [`state`]: Internal state structures (ocean temperatures, upwelling rates)
//! - [`ocean_column`]: Ocean column solver (tridiagonal diffusion-advection)
//! - [`lamcalc`]: Iterative feedback parameter solver

pub mod lamcalc;
mod ocean_column;
pub mod state;
mod udeb;

pub use state::ClimateUDEBState;
pub use udeb::ClimateUDEB;
