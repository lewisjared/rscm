mod carbon_cycle;
mod co2_erf;
pub mod four_box_ocean_heat_uptake;
pub mod ocean_carbon_cycle;

pub use carbon_cycle::{CarbonCycleComponent, CarbonCycleParameters, SolverOptions};
pub use co2_erf::{CO2ERFParameters, CO2ERF};
pub use four_box_ocean_heat_uptake::{
    FourBoxOceanHeatUptakeComponent, FourBoxOceanHeatUptakeParameters,
};
