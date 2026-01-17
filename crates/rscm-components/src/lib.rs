//! Shared climate model components for RSCM
//!
//! This crate provides a collection of reusable climate model components that implement
//! the [`Component`](rscm_core::component::Component) trait from `rscm-core`. These components
//! can be combined using the [`Model`](rscm_core::model::Model) orchestrator to build
//! reduced-complexity climate models.
//!
//! # Available Components
//!
//! ## Carbon Cycle
//!
//! - [`CarbonCycle`] - A one-box carbon cycle model that tracks atmospheric CO2 concentrations
//!   and land uptake based on emissions and temperature
//!
//! ## Radiative Forcing
//!
//! - [`CO2ERF`] - Calculates effective radiative forcing from CO2 concentrations using the
//!   standard logarithmic relationship
//!
//! ## Ocean Heat Uptake
//!
//! - [`FourBoxOceanHeatUptake`] - Distributes scalar ERF to regional ocean heat uptake using
//!   a four-box grid structure (Northern/Southern Ocean/Land)
//!
//! ## Ocean Carbon Cycle
//!
//! - [`ocean_carbon_cycle::OceanSurfacePartialPressure`] - Calculates partial pressure of CO2
//!   at the ocean surface based on temperature and dissolved inorganic carbon
//!
//! # Usage
//!
//! Components are designed to be composed into a [`Model`](rscm_core::model::Model) using
//! the [`ModelBuilder`](rscm_core::model::ModelBuilder). Each component declares its input
//! and output requirements, and the model orchestrator automatically resolves dependencies
//! and executes components in the correct order.
//!
//! ```rust,ignore
//! use rscm_core::model::ModelBuilder;
//! use rscm_components::{CarbonCycle, CarbonCycleParameters, CO2ERF, CO2ERFParameters};
//!
//! let carbon_cycle = CarbonCycle::from_parameters(CarbonCycleParameters {
//!     tau: 30.0,
//!     conc_pi: 278.0,
//!     alpha_temperature: 0.0,
//! });
//!
//! let co2_erf = CO2ERF::from_parameters(CO2ERFParameters {
//!     erf_2xco2: 3.7,
//!     conc_pi: 278.0,
//! });
//!
//! let model = ModelBuilder::new()
//!     .with_component(Box::new(carbon_cycle))
//!     .with_component(Box::new(co2_erf))
//!     .build();
//! ```
//!
//! # Constants
//!
//! The [`constants`] module provides physical constants used across components, such as
//! the conversion factor between GtC and ppm of atmospheric CO2.

mod components;
pub mod constants;
pub mod python;

pub use components::*;
