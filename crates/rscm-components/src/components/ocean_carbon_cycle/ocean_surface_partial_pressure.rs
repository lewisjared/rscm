/// Ocean Surface Partial Pressure(OSPP) calculations
use numpy::array;
use numpy::ndarray::Array1;
use rscm_core::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter::zip;

/// Parameters for the Ocean Surface Partial Pressure component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanSurfacePartialPressureParameters {
    /// Pre-industrial ocean surface partial pressure
    /// Units: `ppm`
    pub ospp_preindustrial: f64,
    /// Sensitivity of the ocean surface's partial pressure to changes in sea
    /// surface temperature relative to pre-industrial
    /// Units: `1 / delta_degC`
    pub sensitivity_ospp_to_temperature: f64,

    /// Pre-industrial sea surface temperature
    /// Units: `degC`
    pub sea_surface_temperature_preindustrial: f64,

    /// Vector of length 5 of offsets to be used when calculating the change in
    /// ocean surface partial pressure
    /// Units: `ppm`
    pub delta_ospp_offsets: [f64; 5],
    /// Vector of length 5 of coefficients (applied to pre-industrial sea surface temperatures)
    /// to be used when calculating the change in
    /// ocean surface partial pressure
    ///
    /// Units: `ppm / delta_degC`
    pub delta_ospp_coefficients: [f64; 5],
}

/// Calculate partial pressure of |CO2| at the ocean's surface
///
/// # Requirements
/// delta_sea_surface_temperature
///    Change in sea surface temperature relative to pre-industrial
///    Units: `delta_degC`
///
/// delta_dissolved_inorganic_carbon
///    Change in dissolved inorganic carbon relative to pre-industrial
///    Units: `micromol / kg`
///
/// Returns
/// -------
/// Partial pressure of |CO2| at the ocean's surface \[ppm\]
///
/// Notes
/// -----
/// Eq. A24 and A25 of [`joos_et_al_2001_feedbacks`]
///
/// ```math
///
///     \text{CO}_{2_s} = [
///         p\text{CO}_{2_{s_0}}
///         + \delta p \text{CO}_{2_s}
///     ] \exp (\alpha \Delta T) \\
///
///     \delta p \text{CO}_{2_s} =
///         (\vec{\beta} + T_0 \vec{\gamma})
///         \cdot \vec{\Sigma} \\
///
///     \vec{\Sigma} = \begin{pmatrix}
///         \Delta \Sigma \text{CO}_2
///         & (\Delta \Sigma \text{CO}_2)^2 \times 10^{-3}
///         & -(\Delta \Sigma \text{CO}_2)^3 \times 10^{-5}
///         & (\Delta \Sigma \text{CO}_2)^4 \times 10^{-7}
///         & -(\Delta \Sigma \text{CO}_2)^5 \times 10^{-10}
///     \end{pmatrix}
/// ```
///
/// and e.g.
///
/// ```math
///     \vec{\beta} = \begin{pmatrix}
///         1.5568
///         & 7.4706
///         & 1.2748
///         & 2.4491
///         & 1.5468
///     \end{pmatrix} \\
///
///     \vec{\gamma} = \begin{pmatrix}
///         -0.013993
///         & -0.20207
///         & -0.12015
///         & -0.12639
///         & -0.15326
///     \end{pmatrix}
/// ```
///
/// Is there a typo in Joos et al., 2001? Should it be
/// (1.5568 - 1.3993 T_0) * 10 ** -2 rather than (1.5568 - 1.3993 T_0 * 10 ** -2) ?
/// The former would make more sense given the brackets in the lines below in the paper. (#208)
///
/// [`joos_et_al_2001_feedbacks`]: https://doi.org/10.1029/2000GB001375
#[derive(Debug, Clone, Serialize, Deserialize)]
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
            .map(|(offset, coeff)| {
                offset + coeff * self.parameters.sea_surface_temperature_preindustrial
            }),
        )
        .dot(&delta_dissolved_inorganic_carbon_bits)
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
        let delta_sea_surface_temperature = input_state.get_latest("Sea Surface Temperature");
        let delta_dissolved_inorganic_carbon = input_state.get_latest("Dissolved Inorganic Carbon");

        let delta_ocean_surface_partial_pressure =
            self.calculate_ospp(delta_dissolved_inorganic_carbon);

        // this exponential is basically just 1 given the scale of the constant
        let ocean_surface_partial_pressure = (self.parameters.ospp_preindustrial
            + delta_ocean_surface_partial_pressure)
            * (self.parameters.sensitivity_ospp_to_temperature * delta_sea_surface_temperature)
                .exp();

        Ok(HashMap::from([(
            "Ocean Surface Partial Pressure|CO2".to_string(),
            StateValue::Scalar(ocean_surface_partial_pressure),
        )]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rscm_core::model::extract_state;
    use rscm_core::state::StateValue;
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::{TimeseriesCollection, VariableType};
    use rstest::rstest;
    use std::f64;

    fn build_timeseries_collection(
        sea_surface_temperature: f64,
        dissolved_inorganic_carbon: f64,
    ) -> TimeseriesCollection {
        let mut collection = TimeseriesCollection::new();

        collection.add_timeseries(
            "Sea Surface Temperature".to_string(),
            Timeseries::from_values(
                array![sea_surface_temperature, f64::NAN],
                array![2020.0, 2021.0],
            ),
            VariableType::Exogenous,
        );

        collection.add_timeseries(
            "Dissolved Inorganic Carbon".to_string(),
            Timeseries::from_values(
                array![dissolved_inorganic_carbon, f64::NAN],
                array![2020.0, 2021.0],
            ),
            VariableType::Exogenous,
        );
        collection
    }

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

        let collection = build_timeseries_collection(4.0, 5.0);
        let input_state = extract_state(&collection, component.input_names(), 2020.0);
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
