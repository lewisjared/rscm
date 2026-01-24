//! Core traits and abstractions for building reduced-complexity climate models.
//!
//! This crate provides the fundamental building blocks for constructing modular
//! climate models in Rust. It defines the core abstractions that allow components
//! to be composed into complete climate models with automatic dependency resolution.
//!
//! # Key Types
//!
//! - [`Component`](component::Component): The fundamental trait for model components.
//!   Each component encapsulates some physics (e.g., carbon cycle, radiative forcing)
//!   and declares its inputs and outputs.
//!
//! - [`Model`](model::Model): Orchestrates multiple components, managing state flow
//!   between them and solving them in dependency order.
//!
//! - [`Timeseries`](timeseries::Timeseries): Time-indexed data with configurable
//!   interpolation strategies. Supports both scalar and spatially-resolved data
//!   via [`GridTimeseries`](timeseries::GridTimeseries).
//!
//! - [`TimeseriesCollection`](timeseries_collection::TimeseriesCollection): A named
//!   collection of timeseries that holds all model state during a simulation.
//!
//! # Architecture Overview
//!
//! Models are built by composing components:
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │ Emissions       │────▶│  Carbon Cycle    │────▶│  ERF Component  │
//! │ (exogenous)     │     │  (concentration) │     │  (forcing)      │
//! └─────────────────┘     └──────────────────┘     └─────────────────┘
//! ```
//!
//! The [`ModelBuilder`](model::ModelBuilder) constructs a dependency graph and
//! validates that all inputs are satisfied (either by other components or
//! exogenous data).
//!
//! # Basic Usage
//!
//! ```ignore
//! use std::sync::Arc;
//! use rscm_core::model::ModelBuilder;
//! use rscm_core::timeseries::{TimeAxis, Timeseries};
//!
//! // Build a model with components
//! let mut model = ModelBuilder::new()
//!     .with_time_axis(TimeAxis::from_values(array![2020.0, 2021.0, 2022.0]))
//!     .with_component(Arc::new(my_carbon_cycle))
//!     .with_component(Arc::new(my_erf_component))
//!     .with_exogenous_variable("Emissions|CO2", emissions_timeseries)
//!     .build()?;
//!
//! // Run the model
//! model.run();
//!
//! // Access results
//! let concentrations = model.timeseries().get_data("Atmospheric Concentration|CO2");
//! ```
//!
//! # Creating Components
//!
//! Components are best created using the [`ComponentIO`] derive macro from `rscm-macros`:
//!
//! ```ignore
//! use rscm_core::ComponentIO;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Debug, Serialize, Deserialize, ComponentIO)]
//! #[inputs(
//!     concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
//! )]
//! #[outputs(
//!     erf { name = "Effective Radiative Forcing|CO2", unit = "W / m^2" },
//! )]
//! pub struct CO2ERF {
//!     pub erf_2xco2: f64,
//!     pub conc_pi: f64,
//! }
//!
//! #[typetag::serde]
//! impl Component for CO2ERF {
//!     fn definitions(&self) -> Vec<RequirementDefinition> {
//!         Self::generated_definitions()
//!     }
//!
//!     fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
//!         let inputs = CO2ERFInputs::from_input_state(input_state);
//!         let concentration = inputs.concentration.at_start();
//!         let erf = self.erf_2xco2 / 2.0_f64.ln()
//!             * (1.0 + (concentration - self.conc_pi) / self.conc_pi).ln();
//!         Ok(CO2ERFOutputs { erf }.into())
//!     }
//! }
//! ```
//!
//! # Spatial Grids
//!
//! The framework supports spatially-resolved variables through different grid types:
//!
//! - **Scalar**: Global mean values (default)
//! - **FourBox**: Four regions (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
//! - **Hemispheric**: Two hemispheres (Northern, Southern)
//!
//! Grid transformations (e.g., FourBox to Scalar aggregation) are handled automatically
//! by the [`grid_transform`] module when components with different spatial resolutions
//! are connected.

// Allow macro-generated code to reference this crate by name
extern crate self as rscm_core;

pub mod component;
mod example_components;
pub mod grid_transform;
pub mod interpolate;
pub mod ivp;
pub mod model;
pub mod python;
pub mod schema;
pub mod spatial;
pub mod state;
pub mod timeseries;
pub mod timeseries_collection;

pub mod errors;

// Re-export derive macro for convenience
pub use rscm_macros::ComponentIO;
