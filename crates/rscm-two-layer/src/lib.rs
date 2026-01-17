//! Two-layer energy balance model for climate simulations.
//!
//! This crate provides a two-layer energy balance model implementation following
//! Held et al. (2010). The model represents the climate system as two coupled
//! thermal reservoirs:
//!
//! - **Surface layer**: A fast-responding mixed layer ocean combined with the atmosphere
//! - **Deep layer**: A slow-responding deep ocean reservoir
//!
//! This simple yet physically-motivated structure captures the essential features of
//! the global mean temperature response to radiative forcing, including the fast
//! transient response and slow equilibration timescales observed in comprehensive
//! climate models.
//!
//! # Scientific Background
//!
//! The two-layer model is derived from energy balance considerations. The surface
//! layer exchanges heat with space (via radiative feedback) and with the deep ocean
//! (via diffusive mixing). Key features include:
//!
//! - **State-dependent feedback**: The climate feedback parameter can vary with
//!   temperature, capturing nonlinear effects observed in GCMs
//! - **Ocean heat uptake efficacy**: Accounts for the finding that ocean heat uptake
//!   is more effective at reducing surface warming than equilibrium feedbacks suggest
//! - **Separation of timescales**: Distinct fast (~5-10 year) and slow (~100+ year)
//!   response modes
//!
//! # Key Types
//!
//! - [`TwoLayer`]: The main component implementing the [`Component`](rscm_core::component::Component)
//!   trait for integration into RSCM models
//! - [`TwoLayerParameters`]: Configuration parameters for the model (feedback strength,
//!   heat capacities, efficacy, etc.)
//!
//! # Usage
//!
//! The [`TwoLayer`] component can be used standalone or as part of a larger RSCM model.
//!
//! ## Creating a Component
//!
//! ```rust
//! use rscm_two_layer::{TwoLayer, TwoLayerParameters};
//!
//! // Create a two-layer model with typical parameter values
//! let component = TwoLayer::from_parameters(TwoLayerParameters {
//!     lambda0: 1.0,               // W/(m² K) - climate feedback parameter
//!     a: 0.0,                     // W/(m² K²) - nonlinear feedback (0 = linear)
//!     efficacy: 1.2,              // dimensionless - ocean heat uptake efficacy
//!     eta: 0.7,                   // W/(m² K) - heat exchange coefficient
//!     heat_capacity_surface: 8.0, // W yr/(m² K) - surface layer (~80m mixed layer)
//!     heat_capacity_deep: 100.0,  // W yr/(m² K) - deep ocean
//! });
//! ```
//!
//! ## Integration with RSCM Models
//!
//! The component requires "Effective Radiative Forcing" as input and produces
//! "Surface Temperature" as output:
//!
//! ```rust,ignore
//! use rscm_core::model::ModelBuilder;
//! use rscm_two_layer::{TwoLayer, TwoLayerParameters};
//!
//! // Build a model with the two-layer component
//! let model = ModelBuilder::new()
//!     .with_component(Box::new(TwoLayer::from_parameters(TwoLayerParameters {
//!         lambda0: 1.0,
//!         a: 0.0,
//!         efficacy: 1.2,
//!         eta: 0.7,
//!         heat_capacity_surface: 8.0,
//!         heat_capacity_deep: 100.0,
//!     })))
//!     .build();
//! ```
//!
//! # References
//!
//! Held, I. M., Winton, M., Takahashi, K., Delworth, T., Zeng, F., & Vallis, G. K. (2010).
//! Probing the fast and slow components of global warming by returning abruptly to
//! preindustrial forcing. *Journal of Climate*, 23(9), 2418-2427.
//! <https://doi.org/10.1175/2009JCLI3466.1>

pub mod component;
pub mod python;

pub use component::{TwoLayer, TwoLayerParameters};
