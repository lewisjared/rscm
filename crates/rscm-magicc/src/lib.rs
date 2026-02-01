//! MAGICC components for RSCM
//!
//! This crate provides components implementing the MAGICC7 reduced-complexity climate model
//! following the modular architecture documented in the MAGICC7 module specifications.
//!
//! # Module Organisation
//!
//! Components are organised by domain:
//! - `forcing`: Radiative forcing calculations (ozone, aerosol)
//! - `chemistry`: Atmospheric chemistry (CH4, N2O, halocarbons)
//! - `climate`: Climate response (UDEB energy balance model)
//! - `carbon`: Carbon cycle (terrestrial, ocean, CO2 budget)
//!
//! # Parameters
//!
//! Each component has an associated parameters struct in the `parameters` module
//! with sensible defaults matching MAGICC7 default configuration.

pub mod carbon;
pub mod chemistry;
pub mod climate;
pub mod forcing;
pub mod parameters;
pub mod python;
