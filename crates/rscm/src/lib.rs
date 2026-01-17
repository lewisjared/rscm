//! PyO3 Python bindings for the RSCM (Rust Simple Climate Model) framework.
//!
//! This crate provides the Python interface to RSCM, exposing the Rust core library
//! as a native Python extension module. It is the primary entry point for Python users
//! and is not intended to be used directly from Rust code.
//!
//! # Python Module Structure
//!
//! The extension module is named `_lib` and is typically accessed through the `rscm`
//! Python package. The module hierarchy exposed to Python is:
//!
//! - `rscm._lib` - Root extension module
//!   - `rscm._lib.core` - Core abstractions (Model, Component, Timeseries, etc.)
//!     - `rscm._lib.core.spatial` - Spatial grid types (FourBox, Hemispheric)
//!     - `rscm._lib.core.state` - State value types (StateValue, FourBoxSlice, etc.)
//!   - `rscm._lib.components` - Generic climate model components (CarbonCycle, CO2ERF, etc.)
//!   - `rscm._lib.two_layer` - Two-layer energy balance model
//!   - `rscm._lib.magicc` - MAGICC component implementations (scaffold)
//!
//! # Usage from Python
//!
//! ```python
//! from rscm._lib.core import Model, ModelBuilder, TimeAxis, Timeseries
//! from rscm._lib.two_layer import TwoLayerBuilder
//!
//! # Create a component from parameters
//! builder = TwoLayerBuilder.from_parameters({
//!     "lambda0": 1.0,
//!     "a": 0.0,
//!     "efficacy": 1.0,
//!     "eta": 0.7,
//!     "heat_capacity_surface": 8.0,
//!     "heat_capacity_deep": 100.0,
//! })
//! component = builder.build()
//!
//! # Build and run a model
//! model = (
//!     ModelBuilder()
//!     .with_time_axis(time_axis)
//!     .with_rust_component(component)
//!     .with_exogenous_variable("Effective Radiative Forcing", erf_timeseries)
//!     .build()
//! )
//! model.run()
//! ```
//!
//! # Crate Structure
//!
//! This crate re-exports Python bindings from the other workspace crates:
//!
//! - [`rscm_core`] - Core traits and abstractions
//! - [`rscm_components`] - Generic climate components
//! - [`rscm_two_layer`] - Two-layer model
//! - [`rscm_magicc`] - MAGICC components (future)
//!
//! The [`python`] module contains the PyO3 module registration that wires everything together.

pub mod python;
