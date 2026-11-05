//! FaIR v2 components for RSCM
//!
//! Each component reimplements one step of the main loop of FaIR v2.2.5
//! (Leach et al. 2021, Smith et al. 2024).
//! Coupled in the order below, they reproduce a FaIR run:
//!
//! 1. [`FairCH4Lifetime`]: CH4 lifetime from precursors and temperature (Thornhill et al. 2021)
//! 2. [`FairGasCycle`]: greenhouse gas emissions to concentrations, or back
//! 3. [`FairEesc`]: equivalent effective stratospheric chlorine
//! 4. [`FairGhgForcing`]: greenhouse gas forcing (Meinshausen et al. 2020)
//! 5. [`FairLinearForcing`]: forcing linear in its drivers (aerosol-radiation, ozone, contrails,
//!    light-absorbing particles, stratospheric water vapour and land use)
//! 6. [`FairAerosolCloud`]: aerosol-cloud interactions
//! 7. [`FairForcingAggregation`]: total and efficacy-adjusted forcing
//! 8. [`FairEnergyBalance`]: the n-layer energy balance model (Cummins et al. 2020)
//!
//! Components take the names of the variables they read and write as parameters,
//! so `rscm.fair` decides the naming.

pub mod aerosol_cloud;
pub mod aggregation;
pub mod ch4_lifetime;
pub mod constants;
pub mod eesc;
pub mod energy_balance;
pub mod gas_cycle;
pub mod ghg_forcing;
pub mod linear_forcing;
pub mod python;

pub use aerosol_cloud::{FairAerosolCloud, FairAerosolCloudParameters};
pub use aggregation::{FairForcingAggregation, FairForcingAggregationParameters};
pub use ch4_lifetime::{FairCH4Lifetime, FairCH4LifetimeParameters};
pub use eesc::{FairEesc, FairEescParameters};
pub use energy_balance::{FairEnergyBalance, FairEnergyBalanceParameters};
pub use gas_cycle::{FairGasCycle, FairGasCycleParameters};
pub use ghg_forcing::{FairGhgForcing, FairGhgForcingParameters};
pub use linear_forcing::{FairLinearForcing, FairLinearForcingParameters};
