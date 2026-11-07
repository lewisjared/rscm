//! FaIR v2 components for RSCM
//!
//! Each component reimplements one step of the main loop of FaIR v2.2.5
//! (Leach et al. 2021, Smith et al. 2024).
//! Coupled in the order below, they reproduce a FaIR run:
//!
//! 1. [`CH4Lifetime`]: CH4 lifetime from precursors and temperature (Thornhill et al. 2021)
//! 2. [`GasCycle`]: greenhouse gas emissions to concentrations, or back
//! 3. [`Eesc`]: equivalent effective stratospheric chlorine
//! 4. [`GhgForcing`]: greenhouse gas forcing (Meinshausen et al. 2020)
//! 5. [`LinearForcing`]: forcing linear in its drivers (aerosol-radiation, ozone, contrails,
//!    light-absorbing particles, stratospheric water vapour and land use)
//! 6. [`AerosolCloud`]: aerosol-cloud interactions
//! 7. [`ForcingAggregation`]: total and efficacy-adjusted forcing
//! 8. [`EnergyBalance`]: the n-layer energy balance model (Cummins et al. 2020)
//!
//! Components take the names of the variables they read and write as parameters,
//! so `rscm.fair` decides the naming.

pub mod aerosol_cloud;
pub mod aggregation;
pub mod ch4_lifetime;
pub mod constants;
pub mod driver;
pub mod eesc;
pub mod energy_balance;
pub mod gas_cycle;
pub mod ghg_forcing;
pub mod linear_forcing;
pub mod python;

pub use aerosol_cloud::{AerosolCloud, AerosolCloudParameters};
pub use aggregation::{ForcingAggregation, ForcingAggregationParameters};
pub use ch4_lifetime::{CH4Lifetime, CH4LifetimeParameters};
pub use driver::Driver;
pub use eesc::{Eesc, EescParameters};
pub use energy_balance::{EnergyBalance, EnergyBalanceParameters};
pub use gas_cycle::{GasCycle, GasCycleParameters};
pub use ghg_forcing::{GhgForcing, GhgForcingParameters};
pub use linear_forcing::{LinearForcing, LinearForcingParameters};
