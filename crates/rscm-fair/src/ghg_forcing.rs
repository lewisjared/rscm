//! Greenhouse gas effective radiative forcing (FaIR step 5)
//!
//! Calculates the effective radiative forcing of each greenhouse gas from its concentration.
//! CO2, CH4 and N2O use the Meinshausen et al. (2020) method, with the overlaps between them,
//! and the minor gases have forcing linear in concentration.
//!
//! Each forcing is relative to the forcing at the baseline concentrations,
//! scaled by `forcing_scale * (1 + tropospheric_adjustment)`,
//! plus a feedback on the surface air temperature change at the start of the step.
//! A minor gas's radiative efficiency is per ppb and its concentration is in ppt.
//!
//! # Inputs
//!
//! - The surface air temperature change named by `surface_temperature` (K).
//!   Its default is `Surface Air Temperature Change`.
//! - The `concentration` of `co2`, `ch4`, `n2o` and each `minor` gas, in its `concentration_unit`.
//!   rscm.fair names it `Atmospheric Concentrations|<species>`,
//!   such as `Atmospheric Concentrations|CO2` in ppm.
//!
//! # Outputs
//!
//! - The `forcing` of each gas (W/m^2).
//!   rscm.fair names it `Effective Radiative Forcing|<species>` for CO2, CH4 and N2O,
//!   `Effective Radiative Forcing|F-Gases|<species>` for the F-gases
//!   and `Effective Radiative Forcing|Montreal Protocol Halogen Gases|<species>` for the halogens.

use crate::constants::{forcing_unit, get, one, surface_temperature, temperature_unit};
use rscm_components::meinshausen2020::Meinshausen2020;
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// One greenhouse gas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GHGSpecies {
    /// Name of the concentration input.
    pub concentration: String,
    /// Unit of the concentration.
    pub concentration_unit: String,
    /// Name of the effective radiative forcing output.
    pub forcing: String,
    /// Reference concentration of the forcing formula, in `concentration_unit`
    /// (FaIR `forcing_reference_concentration`).
    pub reference_concentration: FloatValue,
    /// Concentration with zero effective radiative forcing (FaIR `baseline_concentration`).
    pub baseline_concentration: FloatValue,
    /// Multiplier on the effective radiative forcing (FaIR `forcing_scale`).
    #[serde(default = "one")]
    pub forcing_scale: FloatValue,
    /// Tropospheric adjustment to the stratospherically adjusted radiative forcing
    /// (FaIR `tropospheric_adjustment`).
    #[serde(default)]
    pub tropospheric_adjustment: FloatValue,
    /// Radiative efficiency, used only for the minor gases
    /// (FaIR `greenhouse_gas_radiative_efficiency`).
    /// unit: W/m^2/ppb
    #[serde(default)]
    pub radiative_efficiency: FloatValue,
    /// Effective radiative forcing per kelvin of surface air temperature change
    /// (FaIR `forcing_temperature_feedback`).
    /// unit: W/m^2/K
    #[serde(default)]
    pub temperature_feedback: FloatValue,
}

impl GHGSpecies {
    fn scale(&self) -> FloatValue {
        self.forcing_scale * (1.0 + self.tropospheric_adjustment)
    }
}

/// Parameters of [`GHGForcing`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GHGForcingParameters {
    /// CO2, with the Meinshausen et al. (2020) CO2 formula.
    pub co2: GHGSpecies,
    /// CH4, with the Meinshausen et al. (2020) CH4 formula.
    pub ch4: GHGSpecies,
    /// N2O, with the Meinshausen et al. (2020) N2O formula.
    pub n2o: GHGSpecies,
    /// Gases with effective radiative forcing linear in concentration.
    #[serde(default)]
    pub minor: Vec<GHGSpecies>,
    /// Name of the surface air temperature change input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

/// FaIR's Meinshausen et al. (2020) greenhouse gas effective radiative forcing.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["forcing", "ghg", "fair"], category = "Radiative Forcing")]
pub struct GHGForcing {
    parameters: GHGForcingParameters,
    #[serde(skip)]
    offset: OnceLock<Vec<FloatValue>>,
}

impl GHGForcing {
    /// Create the component.
    pub fn from_parameters(parameters: GHGForcingParameters) -> Self {
        Self {
            parameters,
            offset: OnceLock::new(),
        }
    }

    fn species(&self) -> impl Iterator<Item = &GHGSpecies> {
        let p = &self.parameters;
        [&p.co2, &p.ch4, &p.n2o].into_iter().chain(&p.minor)
    }

    /// Unoffset forcing of each gas, CO2, CH4, N2O then the minor gases.
    pub fn meinshausen2020(&self, concentrations: &[FloatValue]) -> Vec<FloatValue> {
        let p = &self.parameters;
        let m = Meinshausen2020::default();
        let (co2, ch4, n2o) = (concentrations[0], concentrations[1], concentrations[2]);

        let mut forcing = Vec::with_capacity(concentrations.len());
        forcing.push(m.co2(co2, n2o, p.co2.reference_concentration, p.co2.scale()));
        forcing.push(m.ch4(ch4, n2o, p.ch4.reference_concentration, p.ch4.scale()));
        forcing.push(m.n2o(co2, ch4, n2o, p.n2o.reference_concentration, p.n2o.scale()));
        for (gas, c) in p.minor.iter().zip(&concentrations[3..]) {
            forcing.push(
                (c - gas.reference_concentration) * gas.radiative_efficiency * 0.001 * gas.scale(),
            );
        }
        forcing
    }

    /// Forcing of each gas relative to the baseline, before the temperature feedback.
    pub fn forcing(&self, concentrations: &[FloatValue]) -> Vec<FloatValue> {
        let offset = self.offset.get_or_init(|| {
            let baseline: Vec<_> = self.species().map(|s| s.baseline_concentration).collect();
            self.meinshausen2020(&baseline)
        });
        self.meinshausen2020(concentrations)
            .into_iter()
            .zip(offset)
            .map(|(f, o)| f - o)
            .collect()
    }
}

#[typetag::serde(name = "fair.GHGForcing")]
impl Component for GHGForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = vec![VariableDefinition::scalar_input(
            &self.parameters.surface_temperature,
            &temperature_unit(),
        )];
        for s in self.species() {
            definitions.push(VariableDefinition::scalar_input(
                &s.concentration,
                &s.concentration_unit,
            ));
            definitions.push(VariableDefinition::scalar_output(
                &s.forcing,
                &forcing_unit(),
            ));
        }
        definitions
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let concentrations: Vec<_> = self
            .species()
            .map(|s| get(input_state, &s.concentration))
            .collect();
        let temperature = get(input_state, &self.parameters.surface_temperature);
        let mut output = OutputState::with_capacity(concentrations.len());
        for (s, f) in self.species().zip(self.forcing(&concentrations)) {
            output.insert(
                s.forcing.clone(),
                StateValue::Scalar(f + temperature * s.temperature_feedback),
            );
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn gas(name: &str, reference: FloatValue) -> GHGSpecies {
        GHGSpecies {
            concentration: name.into(),
            concentration_unit: "ppm".into(),
            forcing: format!("Effective Radiative Forcing|{name}"),
            reference_concentration: reference,
            baseline_concentration: reference,
            forcing_scale: 1.0,
            tropospheric_adjustment: 0.0,
            radiative_efficiency: 0.0,
            temperature_feedback: 0.0,
        }
    }

    fn component() -> GHGForcing {
        GHGForcing::from_parameters(GHGForcingParameters {
            co2: gas("CO2", 277.15),
            ch4: gas("CH4", 731.41),
            n2o: gas("N2O", 273.87),
            minor: vec![],
            surface_temperature: surface_temperature(),
        })
    }

    #[test]
    fn baseline_has_no_forcing() {
        for f in component().forcing(&[277.15, 731.41, 273.87]) {
            assert_relative_eq!(f, 0.0);
        }
    }

    #[test]
    fn doubled_co2_is_near_four_watts() {
        let f = component().forcing(&[2.0 * 277.15, 731.41, 273.87]);
        assert!((3.5..4.5).contains(&f[0]), "{}", f[0]);
    }
}
