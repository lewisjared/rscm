//! Climate domain components for MAGICC
//!
//! This module contains climate model components implementing MAGICC-style
//! parameterisations and physics.
//!
//! - `ClimateUDEB`: 4-box upwelling-diffusion energy balance model with
//!   50-layer ocean

mod udeb;

pub use udeb::ClimateUDEB;
