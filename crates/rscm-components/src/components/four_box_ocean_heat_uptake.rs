//! Four-box ocean heat uptake component
//!
//! This example component demonstrates using grid timeseries in RSCM.
//! It takes scalar effective radiative forcing (ERF) as input and produces
//! regional ocean heat uptake values for a four-box grid structure.
//!
//! # Physics
//!
//! Ocean heat uptake varies by region due to:
//! - Ocean vs. land: Oceans have higher heat capacity and absorb more energy
//! - Latitude: Polar regions have different uptake characteristics
//! - Mixing: Deep ocean mixing affects heat penetration
//!
//! This simplified model uses fixed regional fractions to distribute global forcing.

use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    TimeseriesWindow,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::Time;
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Parameters for the four-box ocean heat uptake component
///
/// The ratio parameters represent regional_uptake / global_ERF.
/// With equal area weights (0.25 each), these ratios should average to 1.0
/// for the area-weighted mean to equal the global ERF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourBoxOceanHeatUptakeParameters {
    /// Ratio of Northern Ocean uptake to global ERF (regional / global)
    pub northern_ocean_ratio: f64,
    /// Ratio of Northern Land uptake to global ERF (regional / global)
    pub northern_land_ratio: f64,
    /// Ratio of Southern Ocean uptake to global ERF (regional / global)
    pub southern_ocean_ratio: f64,
    /// Ratio of Southern Land uptake to global ERF (regional / global)
    pub southern_land_ratio: f64,
}

impl Default for FourBoxOceanHeatUptakeParameters {
    fn default() -> Self {
        Self {
            // Ratios represent regional/global
            // Oceans have higher ratios (absorb more than average)
            // Land has lower ratios (absorb less than average)
            // With equal weights, these average to 1.0
            northern_ocean_ratio: 1.2, // 20% above average
            northern_land_ratio: 0.6,  // 40% below average
            southern_ocean_ratio: 1.6, // 60% above average (Southern Ocean is most efficient)
            southern_land_ratio: 0.6,  // 40% below average
        }
    }
}

/// Four-box ocean heat uptake component
///
/// Takes scalar ERF input and produces regional ocean heat uptake.
/// Demonstrates disaggregation from scalar to grid resolution.
///
/// # Example
///
/// ```rust
/// use rscm_components::{
///     FourBoxOceanHeatUptakeComponent,
///     FourBoxOceanHeatUptakeParameters,
/// };
///
/// let params = FourBoxOceanHeatUptakeParameters::default();
/// let component = FourBoxOceanHeatUptakeComponent::from_parameters(params);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[inputs(
    erf { name = "Effective Radiative Forcing|Aggregated", unit = "W/m^2" },
)]
#[outputs(
    heat_uptake { name = "Ocean Heat Uptake|FourBox", unit = "W/m^2", grid = "FourBox" },
)]
pub struct FourBoxOceanHeatUptakeComponent {
    pub parameters: FourBoxOceanHeatUptakeParameters,
}

impl FourBoxOceanHeatUptakeComponent {
    pub fn from_parameters(parameters: FourBoxOceanHeatUptakeParameters) -> Self {
        // Validate that ratios average to approximately 1.0 with equal weights
        // This ensures the area-weighted mean equals the global value
        let avg = (parameters.northern_ocean_ratio
            + parameters.northern_land_ratio
            + parameters.southern_ocean_ratio
            + parameters.southern_land_ratio)
            / 4.0;

        assert!(
            (avg - 1.0).abs() < 0.01,
            "Regional ratios must average to 1.0 with equal weights (got {})",
            avg
        );

        Self { parameters }
    }
}

#[typetag::serde]
impl Component for FourBoxOceanHeatUptakeComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        // Get scalar ERF input using typed inputs
        let inputs = FourBoxOceanHeatUptakeComponentInputs::from_input_state(input_state);
        let erf = inputs.erf.current();

        // Disaggregate to four regions using ratios
        // Regional uptake = global ERF * (regional/global ratio)
        // In a real model, this would be based on physical parameterizations
        let outputs = FourBoxOceanHeatUptakeComponentOutputs {
            heat_uptake: FourBoxSlice::from_array([
                erf * self.parameters.northern_ocean_ratio,
                erf * self.parameters.northern_land_ratio,
                erf * self.parameters.southern_ocean_ratio,
                erf * self.parameters.southern_land_ratio,
            ]),
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;
    use rscm_core::spatial::SpatialGrid;
    use rscm_core::state::StateValue;
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};

    #[test]
    fn test_default_parameters_average_to_one() {
        let params = FourBoxOceanHeatUptakeParameters::default();
        let avg = (params.northern_ocean_ratio
            + params.northern_land_ratio
            + params.southern_ocean_ratio
            + params.southern_land_ratio)
            / 4.0;

        assert!((avg - 1.0).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "Regional ratios must average to 1.0")]
    fn test_invalid_parameters_panic() {
        let params = FourBoxOceanHeatUptakeParameters {
            northern_ocean_ratio: 2.0,
            northern_land_ratio: 2.0,
            southern_ocean_ratio: 2.0,
            southern_land_ratio: 2.0,
        };
        FourBoxOceanHeatUptakeComponent::from_parameters(params);
    }

    #[test]
    fn test_solve_basic() {
        let component = FourBoxOceanHeatUptakeComponent::from_parameters(
            FourBoxOceanHeatUptakeParameters::default(),
        );

        // Create ERF input - use consistent values since get_scalar_window().current()
        // returns the latest value in the timeseries
        let erf_timeseries = TimeseriesItem {
            data: TimeseriesData::Scalar(Timeseries::from_values(
                array![2.5, 2.5],
                array![2020.0, 2021.0],
            )),
            name: "Effective Radiative Forcing|Aggregated".to_string(),
            variable_type: VariableType::Exogenous,
        };

        let input_state = InputState::build(vec![&erf_timeseries], 2020.0);

        // Verify input value using the new API
        let erf_value = input_state
            .get_scalar_window("Effective Radiative Forcing|Aggregated")
            .current();
        println!("ERF input value: {}", erf_value);

        let output = component.solve(2020.0, 2021.0, &input_state).unwrap();

        // Output should be FourBox heat uptake
        assert!(output.contains_key("Ocean Heat Uptake|FourBox"));
        let uptake_state = output.get("Ocean Heat Uptake|FourBox").unwrap();

        // Extract the FourBox grid from the StateValue
        let uptake_slice = match uptake_state {
            StateValue::FourBox(slice) => slice,
            _ => panic!("Expected FourBox output"),
        };
        println!("Heat uptake output: {:?}", uptake_slice);

        // Aggregate back to global to verify
        let grid = rscm_core::spatial::FourBoxGrid::magicc_standard();
        let global_uptake = grid.aggregate_global(&uptake_slice.0);

        // Since fractions sum to 1.0, global uptake should equal input ERF
        assert!((global_uptake - erf_value).abs() < 0.01);
    }

    #[test]
    fn test_solve_with_custom_ratios() {
        // Ratios that average to 1.0: (1.5 + 0.5 + 1.5 + 0.5) / 4 = 1.0
        let params = FourBoxOceanHeatUptakeParameters {
            northern_ocean_ratio: 1.5,
            northern_land_ratio: 0.5,
            southern_ocean_ratio: 1.5,
            southern_land_ratio: 0.5,
        };
        let component = FourBoxOceanHeatUptakeComponent::from_parameters(params);

        let erf_timeseries = TimeseriesItem {
            data: TimeseriesData::Scalar(Timeseries::from_values(
                array![10.0, 10.0],
                array![2020.0, 2021.0],
            )),
            name: "Effective Radiative Forcing|Aggregated".to_string(),
            variable_type: VariableType::Exogenous,
        };

        let input_state = InputState::build(vec![&erf_timeseries], 2020.0);
        let erf_value = input_state
            .get_scalar_window("Effective Radiative Forcing|Aggregated")
            .current();

        let output = component.solve(2020.0, 2021.0, &input_state).unwrap();

        let uptake_state = output.get("Ocean Heat Uptake|FourBox").unwrap();

        // Extract the FourBox grid from the StateValue
        let uptake_slice = match uptake_state {
            StateValue::FourBox(slice) => slice,
            _ => panic!("Expected FourBox output"),
        };

        // Aggregate back to global to verify
        let grid = rscm_core::spatial::FourBoxGrid::magicc_standard();
        let global_uptake = grid.aggregate_global(&uptake_slice.0);

        // With equal weights in grid and ratios averaging to 1.0,
        // aggregated uptake should equal input ERF
        assert!((global_uptake - erf_value).abs() < 0.01);
    }

    #[test]
    fn test_component_definitions() {
        let component = FourBoxOceanHeatUptakeComponent::from_parameters(
            FourBoxOceanHeatUptakeParameters::default(),
        );

        let inputs = component.input_names();
        assert_eq!(inputs.len(), 1);
        assert!(inputs.contains(&"Effective Radiative Forcing|Aggregated".to_string()));

        let outputs = component.output_names();
        assert_eq!(outputs.len(), 1);
        assert!(outputs.contains(&"Ocean Heat Uptake|FourBox".to_string()));
    }
}
