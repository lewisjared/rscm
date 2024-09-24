/// Ocean Surface Partial Pressure(OSPP) calculations
use numpy::array;
use numpy::ndarray::Array1;
use rscm_core::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType, State,
};
use rscm_core::errors::RSCMResult;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::iter::zip;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanSurfacePartialPressureParameters {
    /// Pre-industrial ocean surface partial pressure [ppm]
    ospp_preindustrial: f64,
    /// Sensitivity of the ocean surface's partial pressure to changes in sea
    /// surface temperature relative to pre-industrial [1 / delta_degC]
    sensitivity_ospp_to_temperature: f64,

    /// Pre-industrial sea surface temperature [degC]
    sea_surface_temperature_preindustrial: f64,

    /// Vector of length 5 of offsets to be used when calculating the change in
    /// ocean surface partial pressure [ppm]
    delta_ospp_offsets: [f64; 5],
    /// Vector of length 5 of coefficients (applied to pre-industrial sea surface temperatures)
    /// to be used when calculating the change in
    /// ocean surface partial pressure [ppm / delta_degC]
    delta_ospp_coefficients: [f64; 5],
}

/// See docs in level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanSurfacePartialPressure {
    parameters: OceanSurfacePartialPressureParameters,
}

impl OceanSurfacePartialPressure {
    fn from_parameters(parameters: OceanSurfacePartialPressureParameters) -> Self {
        Self { parameters }
    }

    fn calculate_ospp(&self, delta_dissolved_inorganic_carbon: &FloatValue) -> FloatValue {
        // let delta_dioc_scaled = ((delta_dissolved_inorganic_carbon
        //     / UNIT_REGISTRY.Quantity(1, DISSOLVED_INORGANIC_CARBON_UNITS))
        // .to("dimensionless")
        // .magnitude);
        //
        let delta_dioc_scaled = *delta_dissolved_inorganic_carbon;
        let delta_dissolved_inorganic_carbon_bits = array![
            delta_dioc_scaled,
            delta_dioc_scaled.powi(2) * 10e-3,
            -delta_dioc_scaled.powi(3) * 10e-5,
            delta_dioc_scaled.powi(4) * 10e-7,
            -delta_dioc_scaled.powi(4) * 10e-10,
        ];

        let delta_ocean_surface_partial_pressure = Array1::from_iter(
            zip(
                self.parameters.delta_ospp_offsets,
                self.parameters.delta_ospp_coefficients,
            )
            .map(|(offset, coeff)| {
                offset + coeff * self.parameters.sea_surface_temperature_preindustrial
            }),
        )
        .dot(&delta_dissolved_inorganic_carbon_bits);

        delta_ocean_surface_partial_pressure
    }
}

#[typetag::serde]
impl Component for OceanSurfacePartialPressure {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new("Sea Surface Temperature", "K", RequirementType::Input),
            RequirementDefinition::new(
                "Dissolved Inorganic Carbon",
                "micromol / kg",
                RequirementType::Input,
            ),
            RequirementDefinition::new(
                "Ocean Surface Partial Pressure|CO2",
                "ppm",
                RequirementType::Output,
            ),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let delta_sea_surface_temperature = input_state.get("Sea Surface Temperature");
        let delta_dissolved_inorganic_carbon = input_state.get("Dissolved Inorganic Carbon");

        let delta_ocean_surface_partial_pressure =
            self.calculate_ospp(delta_dissolved_inorganic_carbon);

        // this exponential is basically just 1 given the scale of the constant
        let ocean_surface_partial_pressure = (self.parameters.ospp_preindustrial
            + delta_ocean_surface_partial_pressure)
            * (self.parameters.sensitivity_ospp_to_temperature * delta_sea_surface_temperature)
                .exp();

        Ok(OutputState::from_vectors(
            vec![ocean_surface_partial_pressure],
            self.output_names(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    #[rstest]
    #[case(
        OceanSurfacePartialPressureParameters {
            ospp_preindustrial: 278.0,
            sensitivity_ospp_to_temperature: 0.043,
            delta_ospp_offsets: [1.5568, 7.4706, 1.2748, 2.4491, 1.5468],
            delta_ospp_coefficients: [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326],
            sea_surface_temperature_preindustrial: 17.9,
        },
        339.089
    )]
    #[case(
        OceanSurfacePartialPressureParameters {
            ospp_preindustrial: 315.0,
            sensitivity_ospp_to_temperature: 0.0423,
            delta_ospp_offsets: [1.5, 7.5, 1.3, 2.5, 1.6],
            delta_ospp_coefficients: [-0.02, -0.2, -0.1, -0.14, -0.2],
            sea_surface_temperature_preindustrial: 17.9,
        },
        381.003
    )]
    fn solve(
        #[case] parameters: OceanSurfacePartialPressureParameters,
        #[case] expected_ospp: f64,
    ) {
        let component = OceanSurfacePartialPressure::from_parameters(parameters);

        let input_state = InputState::from_vectors(
            vec![4.0, 5.0],
            vec![
                "Sea Surface Temperature".to_string(),
                "Dissolved Inorganic Carbon".to_string(),
            ],
        );
        let output_state = component.solve(2020.0, 2021.0, &input_state).unwrap();

        assert_relative_eq!(
            *output_state.get("Ocean Surface Partial Pressure|CO2"),
            expected_ospp,
            max_relative = 10e-5
        )
    }
}
