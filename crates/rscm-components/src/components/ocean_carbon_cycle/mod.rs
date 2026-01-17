//! Ocean carbon cycle components
//!
//! This module contains components that model the ocean's role in the global carbon cycle,
//! including CO2 exchange between the atmosphere and ocean surface.
//!
//! # Overview
//!
//! The ocean is a major sink for anthropogenic CO2, absorbing approximately 25% of annual
//! emissions. The rate of CO2 uptake depends on:
//!
//! - **Temperature**: Warmer water holds less dissolved CO2
//! - **Ocean chemistry**: The carbonate buffering system determines CO2 solubility
//! - **Circulation**: Mixing transports carbon from surface to deep ocean
//!
//! # Available Components
//!
//! - [`OceanSurfacePartialPressure`] - Calculates the partial pressure of CO2 at the ocean
//!   surface based on sea surface temperature and dissolved inorganic carbon. Uses the
//!   parameterisation from [Joos et al. (2001)](https://doi.org/10.1029/2000GB001375).
//!
//! # Physics
//!
//! The partial pressure of CO2 at the ocean surface ($pCO_{2,s}$) determines the direction
//! and magnitude of air-sea CO2 flux:
//!
//! $$F_{CO_2} = k \cdot (pCO_{2,atm} - pCO_{2,s})$$
//!
//! Where:
//! - $F_{CO_2}$ is the air-sea CO2 flux
//! - $k$ is the gas transfer velocity
//! - $pCO_{2,atm}$ is atmospheric CO2 partial pressure
//! - $pCO_{2,s}$ is ocean surface CO2 partial pressure
//!
//! # References
//!
//! - Joos, F., et al. (2001). Global warming feedbacks on terrestrial carbon uptake under
//!   the Intergovernmental Panel on Climate Change (IPCC) emission scenarios.
//!   *Global Biogeochemical Cycles*, 15(4), 891-907.
//!   [DOI: 10.1029/2000GB001375](https://doi.org/10.1029/2000GB001375)

mod ocean_surface_partial_pressure;

pub use ocean_surface_partial_pressure::{
    OceanSurfacePartialPressure, OceanSurfacePartialPressureParameters,
};
