//! Climate model component implementations
//!
//! This module contains the core climate model components that can be composed
//! to build reduced-complexity climate models. Each component implements the
//! [`Component`](rscm_core::component::Component) trait and can be orchestrated
//! by the [`Model`](rscm_core::model::Model) to form complete climate simulations.
//!
//! # Component Categories
//!
//! ## Carbon Cycle Components
//!
//! Components that model the exchange of carbon between atmosphere, ocean, and land:
//!
//! - [`CarbonCycle`] - One-box atmospheric carbon model with temperature-dependent uptake
//! - [`ocean_carbon_cycle`] - Ocean carbon cycle submodule with surface partial pressure calculations
//!
//! ## Radiative Forcing Components
//!
//! Components that calculate radiative forcing from greenhouse gas concentrations:
//!
//! - [`CO2ERF`] - CO2 effective radiative forcing using the logarithmic relationship
//!
//! ## Ocean Heat Uptake Components
//!
//! Components that model heat uptake and distribution in the ocean:
//!
//! - [`FourBoxOceanHeatUptake`] - Regional heat uptake using a four-box (hemispheric ocean/land) grid
//!
//! # Using Components
//!
//! Components are instantiated with their parameters and added to a model:
//!
//! ```rust,ignore
//! use rscm_components::{CO2ERF, CO2ERFParameters};
//!
//! let component = CO2ERF::from_parameters(CO2ERFParameters {
//!     erf_2xco2: 3.7,    // W/m^2
//!     conc_pi: 278.0,    // ppm
//! });
//!
//! // Add to model via ModelBuilder
//! let model = ModelBuilder::new()
//!     .with_component(Box::new(component))
//!     .build();
//! ```
//!
//! Each component declares its input requirements and output variables through the
//! `definitions()` method. The model orchestrator uses this information to resolve
//! dependencies and execute components in topological order.

mod carbon_cycle;
mod co2_erf;
pub mod four_box_ocean_heat_uptake;
pub mod ocean_carbon_cycle;

pub use carbon_cycle::{CarbonCycle, CarbonCycleParameters, SolverOptions};
pub use co2_erf::{CO2ERFParameters, CO2ERF};
pub use four_box_ocean_heat_uptake::{FourBoxOceanHeatUptake, FourBoxOceanHeatUptakeParameters};
