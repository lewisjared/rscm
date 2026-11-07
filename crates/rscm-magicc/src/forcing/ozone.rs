//! Ozone Forcing Component
//!
//! Ozone ERF terms of MAGICC7.
//!
//! # Inputs
//!
//! - `EESC` (ppt)
//! - `Atmospheric Concentration|CH4` (ppb) and `Atmospheric Concentration|N2O` (ppb)
//! - `Emissions|{NOx,CO,NMVOC}|{Fossil,AFOLU}` (Mt N/yr, Mt CO/yr and Mt NMVOC/yr)
//! - `Surface Temperature` (K)
//!
//! # Outputs (W/m²) at the end of each year $Y$
//!
//! - `Effective Radiative Forcing|Stratospheric Ozone`:
//!   $s \max(0, (E_Y - E_{Y_0}) / 100)^{\beta}$ from the threshold year $Y_0$, zero before.
//! - `Effective Radiative Forcing|Tropospheric Ozone`:
//!   $r o_{CH4} \ln(M_Y / M_0) + r [o_{NOx} \Delta NOx_{Y-1} + o_{CO} \Delta CO_{Y-1} + o_{VOC} \Delta VOC_{Y-1}]$,
//!   four-box with `trop_regions` as MAGICC7 spreads it within each hemisphere.
//! - `Effective Radiative Forcing|Ozone due to N2O`: $k (N_Y - N_0)$
//! - `Effective Radiative Forcing|Ozone due to Temperature`: $\gamma T_{Y-1}$
//!
//! Changes ($\Delta$) and the subscript 0 are relative to the first year of the run.
//! Each change is the sum of the fossil and AFOLU changes, in MAGICC7's order.
//! Aviation NOx is not separated from other NOx.

use crate::parameters::OzoneForcingParameters;
use crate::reference::{first_value, require_annual, value_at_year};
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::RscmResult;
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

/// Stratospheric ozone ERF output.
pub const STRAT_OZONE_FORCING: &str = "Effective Radiative Forcing|Stratospheric Ozone";
/// Tropospheric ozone ERF output.
pub const TROP_OZONE_FORCING: &str = "Effective Radiative Forcing|Tropospheric Ozone";
/// ERF of ozone produced by N2O.
pub const N2O_OZONE_FORCING: &str = "Effective Radiative Forcing|Ozone due to N2O";
/// ERF of the ozone response to warming.
pub const TEMPERATURE_OZONE_FORCING: &str = "Effective Radiative Forcing|Ozone due to Temperature";

const EESC: &str = "EESC";
const CH4: &str = "Atmospheric Concentration|CH4";
const N2O: &str = "Atmospheric Concentration|N2O";
/// Precursors with their emission units, each given as fossil and AFOLU sectors.
const PRECURSORS: [(&str, &str); 3] = [
    ("NOx", "Mt N/yr"),
    ("CO", "Mt CO/yr"),
    ("NMVOC", "Mt NMVOC/yr"),
];
const SECTORS: [&str; 2] = ["Fossil", "AFOLU"];
const TEMPERATURE: &str = "Surface Temperature";

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
        p.strat_o3_scale * excess.powf(p.strat_cl_exponent)
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
        let precursors = (p.trop_oz_nox * d_nox + p.trop_oz_co * d_co) + p.trop_oz_voc * d_nmvoc;
        p.trop_radeff * p.trop_oz_ch4 * ch4_ratio.ln() + precursors * p.trop_radeff
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
            VariableDefinition::scalar_input(EESC, "ppt"),
            VariableDefinition::scalar_input(CH4, "ppb"),
            VariableDefinition::scalar_input(N2O, "ppb"),
        ];
        for (gas, unit) in PRECURSORS {
            for sector in SECTORS {
                let name = format!("Emissions|{gas}|{sector}");
                definitions.push(VariableDefinition::scalar_input(&name, unit));
            }
        }
        definitions.extend([
            VariableDefinition::scalar_input(TEMPERATURE, "K"),
            VariableDefinition::scalar_output(STRAT_OZONE_FORCING, "W/m^2"),
            if self.parameters.trop_regions.is_some() {
                VariableDefinition::four_box_output(TROP_OZONE_FORCING, "W/m^2")
            } else {
                VariableDefinition::scalar_output(TROP_OZONE_FORCING, "W/m^2")
            },
            VariableDefinition::scalar_output(N2O_OZONE_FORCING, "W/m^2"),
            VariableDefinition::scalar_output(TEMPERATURE_OZONE_FORCING, "W/m^2"),
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

        let eesc = window(EESC);
        let strat = if t_next < p.strat_threshold_year {
            0.0
        } else {
            let threshold = value_at_year(&eesc, t_current, p.strat_threshold_year);
            self.strat_forcing(eesc.get(), threshold)
        };

        let ch4 = window(CH4);
        let change = |gas: &str| {
            let [fossil, afolu] = SECTORS
                .map(|sector| input_state.get_scalar_window(&format!("Emissions|{gas}|{sector}")));
            (fossil.get() - first_value(&fossil)) + (afolu.get() - first_value(&afolu))
        };
        let trop = self.trop_forcing(
            ch4.get() / first_value(&ch4),
            change("NOx"),
            change("CO"),
            change("NMVOC"),
        );

        let n2o = window(N2O);
        let n2o_ozone = p.n2o_radeff * (n2o.get() - first_value(&n2o));
        let temperature = p.temperature_scale * window(TEMPERATURE).get();

        let trop = match &p.trop_regions {
            Some(regions) => StateValue::FourBox(FourBoxSlice::from_array(
                regions.pattern().map(|q| q * trop),
            )),
            None => StateValue::Scalar(trop),
        };
        let mut output = OutputState::new();
        output.insert(TROP_OZONE_FORCING.to_string(), trop);
        for (name, value) in [
            (STRAT_OZONE_FORCING, strat),
            (N2O_OZONE_FORCING, n2o_ozone),
            (TEMPERATURE_OZONE_FORCING, temperature),
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
