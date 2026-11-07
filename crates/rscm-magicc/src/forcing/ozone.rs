//! Ozone Forcing Component
//!
//! Ozone ERF terms of MAGICC7.
//!
//! # Inputs
//!
//! - `Equivalent Effective Stratospheric Chlorine` (ppt)
//! - `Atmospheric Concentrations|CH4` (ppb) and `Atmospheric Concentrations|N2O` (ppb)
//! - `Emissions|{NOx,CO,NMVOC}|{Fossil,Biosphere}` (Mt N/yr, Mt CO/yr and Mt NMVOC/yr)
//! - `Surface Air Temperature Change` (K)
//!
//! # Outputs (W/m²) at the end of each year $Y$
//!
//! - `Effective Radiative Forcing|Ozone|Stratospheric`:
//!   $s \max(0, (E_Y - E_{Y_0}) / 100)^{\beta}$ from the threshold year $Y_0$, zero before.
//! - `Effective Radiative Forcing|Ozone|Tropospheric`:
//!   $r o_{CH4} \ln(M_Y / M_0) + r [o_{NOx} \Delta NOx_{Y-1} + o_{CO} \Delta CO_{Y-1} + o_{VOC} \Delta VOC_{Y-1}]$,
//!   four-box with `tropospheric_ozone_regions` as MAGICC7 spreads it within each hemisphere.
//! - `Effective Radiative Forcing|Ozone|N2O Feedback`: $k (N_Y - N_0)$
//! - `Effective Radiative Forcing|Ozone|Temperature Feedback`: $\gamma T_{Y-1}$
//!
//! Changes ($\Delta$) and the subscript 0 are relative to the first year of the run.
//! Each change is the sum of the fossil and biosphere changes, in MAGICC7's order.
//! Aviation NOx is not separated from other NOx.

use crate::parameters::OzoneForcingParameters;
use crate::reference::{first_value, require_annual, value_at_year};
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::RscmResult;
use rscm_core::names::{
    child, ATMOSPHERIC_CONCENTRATIONS_CH4, ATMOSPHERIC_CONCENTRATIONS_N2O, EMISSIONS,
    EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE, ERF_OZONE_N2O_FEEDBACK, ERF_OZONE_STRATOSPHERIC,
    ERF_OZONE_TEMPERATURE_FEEDBACK, ERF_OZONE_TROPOSPHERIC, SURFACE_AIR_TEMPERATURE_CHANGE,
};
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

/// Precursors with their emission units, each given as fossil and biosphere sectors.
const PRECURSORS: [(&str, &str); 3] = [
    ("NOx", "Mt N/yr"),
    ("CO", "Mt CO/yr"),
    ("NMVOC", "Mt NMVOC/yr"),
];
const SECTORS: [&str; 2] = ["Fossil", "Biosphere"];

fn precursor_name(gas: &str, sector: &str) -> String {
    child(&child(EMISSIONS, gas), sector)
}

/// Ozone forcing component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OzoneForcing {
    parameters: OzoneForcingParameters,
}

impl OzoneForcing {
    /// Create a component with MAGICC7 default parameters.
    pub fn new() -> Self {
        Self::from_parameters(OzoneForcingParameters::default())
    }

    /// Create a component from parameters.
    pub fn from_parameters(parameters: OzoneForcingParameters) -> Self {
        Self { parameters }
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &OzoneForcingParameters {
        &self.parameters
    }

    /// Stratospheric ozone ERF from EESC and its threshold-year value.
    pub fn strat_forcing(&self, eesc: FloatValue, eesc_threshold: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let excess = ((eesc - eesc_threshold) / 100.0).max(0.0);
        p.stratospheric_o3_scale * excess.powf(p.stratospheric_cl_exponent)
    }

    /// Tropospheric ozone ERF from the CH4 ratio and precursor emission changes.
    pub fn trop_forcing(
        &self,
        ch4_ratio: FloatValue,
        d_nox: FloatValue,
        d_co: FloatValue,
        d_nmvoc: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters;
        let precursors = (p.tropospheric_ozone_nox_sensitivity * d_nox
            + p.tropospheric_ozone_co_sensitivity * d_co)
            + p.tropospheric_ozone_nmvoc_sensitivity * d_nmvoc;
        p.tropospheric_ozone_radiative_efficiency
            * p.tropospheric_ozone_ch4_sensitivity
            * ch4_ratio.ln()
            + precursors * p.tropospheric_ozone_radiative_efficiency
    }
}

impl Default for OzoneForcing {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde(name = "magicc.OzoneForcing")]
impl Component for OzoneForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = vec![
            VariableDefinition::scalar_input(EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE, "ppt"),
            VariableDefinition::scalar_input(ATMOSPHERIC_CONCENTRATIONS_CH4, "ppb"),
            VariableDefinition::scalar_input(ATMOSPHERIC_CONCENTRATIONS_N2O, "ppb"),
        ];
        for (gas, unit) in PRECURSORS {
            for sector in SECTORS {
                definitions.push(VariableDefinition::scalar_input(
                    &precursor_name(gas, sector),
                    unit,
                ));
            }
        }
        definitions.extend([
            VariableDefinition::scalar_input(SURFACE_AIR_TEMPERATURE_CHANGE, "K"),
            VariableDefinition::scalar_output(ERF_OZONE_STRATOSPHERIC, "W/m^2"),
            if self.parameters.tropospheric_ozone_regions.is_some() {
                VariableDefinition::four_box_output(ERF_OZONE_TROPOSPHERIC, "W/m^2")
            } else {
                VariableDefinition::scalar_output(ERF_OZONE_TROPOSPHERIC, "W/m^2")
            },
            VariableDefinition::scalar_output(ERF_OZONE_N2O_FEEDBACK, "W/m^2"),
            VariableDefinition::scalar_output(ERF_OZONE_TEMPERATURE_FEEDBACK, "W/m^2"),
        ]);
        definitions
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        require_annual("OzoneForcing", t_current, t_next)?;
        let p = &self.parameters;
        let window = |name| input_state.get_scalar_window(name);

        let eesc = window(EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE);
        let strat = if t_next < p.stratospheric_reference_year {
            0.0
        } else {
            let threshold = value_at_year(&eesc, t_current, p.stratospheric_reference_year);
            self.strat_forcing(eesc.get(), threshold)
        };

        let ch4 = window(ATMOSPHERIC_CONCENTRATIONS_CH4);
        let change = |gas: &str| {
            let [fossil, biosphere] =
                SECTORS.map(|sector| input_state.get_scalar_window(&precursor_name(gas, sector)));
            (fossil.get() - first_value(&fossil)) + (biosphere.get() - first_value(&biosphere))
        };
        let trop = self.trop_forcing(
            ch4.get() / first_value(&ch4),
            change("NOx"),
            change("CO"),
            change("NMVOC"),
        );

        let n2o = window(ATMOSPHERIC_CONCENTRATIONS_N2O);
        let n2o_ozone = p.n2o_ozone_radiative_efficiency * (n2o.get() - first_value(&n2o));
        let temperature = p.temperature_scale * window(SURFACE_AIR_TEMPERATURE_CHANGE).get();

        let trop = match &p.tropospheric_ozone_regions {
            Some(regions) => StateValue::FourBox(FourBoxSlice::from_array(
                regions.pattern().map(|q| q * trop),
            )),
            None => StateValue::Scalar(trop),
        };
        let mut output = OutputState::new();
        output.insert(ERF_OZONE_TROPOSPHERIC.to_string(), trop);
        for (name, value) in [
            (ERF_OZONE_STRATOSPHERIC, strat),
            (ERF_OZONE_N2O_FEEDBACK, n2o_ozone),
            (ERF_OZONE_TEMPERATURE_FEEDBACK, temperature),
        ] {
            output.insert(name.to_string(), StateValue::Scalar(value));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strat_forcing_is_zero_below_threshold_eesc() {
        let c = OzoneForcing::new();
        assert_eq!(c.strat_forcing(1500.0, 1600.0), 0.0);
        assert_eq!(c.strat_forcing(1600.0, 1600.0), 0.0);
    }

    #[test]
    fn strat_forcing_follows_power_law() {
        let c = OzoneForcing::new();
        let expected = -0.0043 * 2.0_f64.powf(1.7);
        assert!((c.strat_forcing(1800.0, 1600.0) - expected).abs() < 1e-15);
    }

    #[test]
    fn trop_forcing_is_zero_at_reference() {
        assert_eq!(OzoneForcing::new().trop_forcing(1.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn trop_forcing_terms() {
        let c = OzoneForcing::new();
        let ch4 = 0.032 * 5.7 * 2.0_f64.ln();
        assert!((c.trop_forcing(2.0, 0.0, 0.0, 0.0) - ch4).abs() < 1e-15);
        let precursors = 0.032 * (0.168 * 10.0 + 0.00396 * 100.0 + 0.01008 * 50.0);
        assert!((c.trop_forcing(1.0, 10.0, 100.0, 50.0) - precursors).abs() < 1e-15);
    }
}
