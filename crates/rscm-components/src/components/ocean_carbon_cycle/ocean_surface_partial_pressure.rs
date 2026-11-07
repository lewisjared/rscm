//! Ocean surface partial pressure component
//!
//! The partial pressure of CO2 at the ocean surface rises with dissolved inorganic carbon
//! and with sea surface temperature.
//! The component follows equations A24 and A25 of
//! [Joos et al. (2001)](https://doi.org/10.1029/2000GB001375).

use numpy::array;
use numpy::ndarray::Array1;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, ScalarWindow, VariableDefinition, VariableRole,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};
use std::iter::zip;

/// Parameters for the ocean surface partial pressure component
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OceanSurfacePartialPressureParameters {
    /// Pre-industrial ocean surface partial pressure of CO2
    /// unit: ppm
    pub ospp_pi: f64,
    /// Sensitivity of the ocean surface partial pressure to sea surface temperature change
    /// unit: 1/K
    pub sensitivity_ospp_to_temperature: f64,

    /// Pre-industrial sea surface temperature
    /// unit: degC
    pub sst_pi: f64,

    /// Offsets of the five polynomial coefficients for the partial pressure change
    /// unit: ppm
    pub delta_ospp_offsets: [f64; 5],
    /// Slopes of the five polynomial coefficients with pre-industrial sea surface temperature
    /// unit: ppm/K
    pub delta_ospp_coefficients: [f64; 5],
}

/// Ocean surface partial pressure of CO2
///
/// The partial pressure is
///
/// $$
/// \text{CO}_{2_s} = \left[
///     p\text{CO}_{2_{s_0}}
///     + \delta p \text{CO}_{2_s}
/// \right] \exp (\alpha \Delta T)
/// $$
///
/// $$
/// \delta p \text{CO}_{2_s} =
///     (\vec{\beta} + T_0 \vec{\gamma})
///     \cdot \vec{\Sigma}
/// $$
///
/// $$
/// \vec{\Sigma} = \begin{pmatrix}
///     \Delta \Sigma \text{CO}_2 \\
///     (\Delta \Sigma \text{CO}_2)^2 \times 10^{-3} \\
///     -(\Delta \Sigma \text{CO}_2)^3 \times 10^{-5} \\
///     (\Delta \Sigma \text{CO}_2)^4 \times 10^{-7} \\
///     -(\Delta \Sigma \text{CO}_2)^5 \times 10^{-10}
/// \end{pmatrix}
/// $$
///
/// where:
///
/// - $p\text{CO}_{2_{s_0}}$ is the pre-industrial partial pressure
/// - $\alpha$ is the sensitivity to sea surface temperature change
/// - $\Delta T$ is the sea surface temperature change since pre-industrial
/// - $T_0$ is the pre-industrial sea surface temperature
/// - $\Delta \Sigma \text{CO}_2$ is the dissolved inorganic carbon change since pre-industrial
/// - $\vec{\beta}$ and $\vec{\gamma}$ are the polynomial offsets and slopes
///
/// Joos et al. (2001) give, for example,
///
/// $$
/// \vec{\beta} = \begin{pmatrix}
///     1.5568 & 7.4706 & 1.2748 & 2.4491 & 1.5468
/// \end{pmatrix}
/// $$
///
/// $$
/// \vec{\gamma} = \begin{pmatrix}
///     -0.013993 & -0.20207 & -0.12015 & -0.12639 & -0.15326
/// \end{pmatrix}
/// $$
///
/// The paper may have a typo in the first coefficient.
/// It may mean $(1.5568 - 1.3993 T_0) \times 10^{-2}$
/// rather than $1.5568 - 1.3993 T_0 \times 10^{-2}$,
/// which the brackets in the lines that follow suggest.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["ocean", "carbon-cycle", "magicc", "experimental"], category = "Carbon Cycle")]
#[inputs(
    sea_surface_temperature { name = "Sea Surface Temperature", unit = "K", description = "Sea surface temperature change since pre-industrial" },
    dissolved_inorganic_carbon { name = "Dissolved Inorganic Carbon", unit = "micromol/kg", description = "Dissolved inorganic carbon change since pre-industrial" },
)]
#[outputs(
    ospp_co2 { name = "Ocean Surface Partial Pressure|CO2", unit = "ppm", description = "Partial pressure of CO2 at the ocean surface" },
)]
pub struct OceanSurfacePartialPressure {
    parameters: OceanSurfacePartialPressureParameters,
}

impl OceanSurfacePartialPressure {
    pub fn from_parameters(parameters: OceanSurfacePartialPressureParameters) -> Self {
        Self { parameters }
    }

    fn calculate_ospp(&self, delta_dissolved_inorganic_carbon: FloatValue) -> FloatValue {
        // TODO: investigate units
        // let delta_dioc_scaled = ((delta_dissolved_inorganic_carbon
        //     / UNIT_REGISTRY.Quantity(1, DISSOLVED_INORGANIC_CARBON_UNITS))
        // .to("dimensionless")
        // .magnitude);
        //
        let delta_dioc_scaled = delta_dissolved_inorganic_carbon;
        let delta_dissolved_inorganic_carbon_bits = array![
            delta_dioc_scaled,
            delta_dioc_scaled.powi(2) * 10e-3,
            -delta_dioc_scaled.powi(3) * 10e-5,
            delta_dioc_scaled.powi(4) * 10e-7,
            -delta_dioc_scaled.powi(4) * 10e-10,
        ];

        Array1::from_iter(
            zip(
                self.parameters.delta_ospp_offsets,
                self.parameters.delta_ospp_coefficients,
            )
            .map(|(offset, coeff)| offset + coeff * self.parameters.sst_pi),
        )
        .dot(&delta_dissolved_inorganic_carbon_bits)
    }
}

#[typetag::serde]
impl Component for OceanSurfacePartialPressure {
    fn definitions(&self) -> Vec<VariableDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = OceanSurfacePartialPressureInputs::from_input_state(input_state);
        let delta_sea_surface_temperature = inputs.sea_surface_temperature.get();
        let delta_dissolved_inorganic_carbon = inputs.dissolved_inorganic_carbon.get();

        let delta_ocean_surface_partial_pressure =
            self.calculate_ospp(delta_dissolved_inorganic_carbon);

        // this exponential is basically just 1 given the scale of the constant
        let ocean_surface_partial_pressure = (self.parameters.ospp_pi
            + delta_ocean_surface_partial_pressure)
            * (self.parameters.sensitivity_ospp_to_temperature * delta_sea_surface_temperature)
                .exp();

        Ok(OceanSurfacePartialPressureOutputs {
            ospp_co2: ocean_surface_partial_pressure,
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rscm_core::model::extract_state;
    use rscm_core::state::StateValue;
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::TimeseriesCollection;
    use rstest::rstest;

    fn build_timeseries_collection(
        sea_surface_temperature: f64,
        dissolved_inorganic_carbon: f64,
    ) -> TimeseriesCollection {
        let mut collection = TimeseriesCollection::new();

        collection.add_timeseries(
            "Sea Surface Temperature".to_string(),
            Timeseries::from_values(
                array![f64::NAN, sea_surface_temperature],
                array![2020.0, 2021.0],
            ),
        );

        collection.add_timeseries(
            "Dissolved Inorganic Carbon".to_string(),
            Timeseries::from_values(
                array![f64::NAN, dissolved_inorganic_carbon],
                array![2020.0, 2021.0],
            ),
        );
        collection
    }

    #[rstest]
    #[case(
        OceanSurfacePartialPressureParameters {
            ospp_pi: 278.0,
            sensitivity_ospp_to_temperature: 0.043,
            delta_ospp_offsets: [1.5568, 7.4706, 1.2748, 2.4491, 1.5468],
            delta_ospp_coefficients: [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326],
            sst_pi: 17.9,
        },
        339.089
    )]
    #[case(
        OceanSurfacePartialPressureParameters {
            ospp_pi: 315.0,
            sensitivity_ospp_to_temperature: 0.0423,
            delta_ospp_offsets: [1.5, 7.5, 1.3, 2.5, 1.6],
            delta_ospp_coefficients: [-0.02, -0.2, -0.1, -0.14, -0.2],
            sst_pi: 17.9,
        },
        381.003
    )]
    fn solve(
        #[case] parameters: OceanSurfacePartialPressureParameters,
        #[case] expected_ospp: f64,
    ) {
        let component = OceanSurfacePartialPressure::from_parameters(parameters);

        let collection = build_timeseries_collection(4.0, 5.0);
        let input_state = extract_state(&collection, component.inputs(), 2020.0);
        let output_state = component.solve(2020.0, 2021.0, &input_state).unwrap();

        let ospp_value = match output_state
            .get("Ocean Surface Partial Pressure|CO2")
            .unwrap()
        {
            StateValue::Scalar(value) => *value,
            _ => panic!("Expected Scalar output"),
        };

        assert_relative_eq!(ospp_value, expected_ospp, max_relative = 10e-5)
    }
}
