//! Greenhouse gas forcing (FaIR step 5)
//!
//! Meinshausen et al. (2020) forcing for CO2, CH4 and N2O, with the overlaps between them,
//! and forcing linear in concentration for the minor gases.
//! Each forcing is relative to the forcing at the baseline concentrations,
//! scaled by `forcing_scale * (1 + tropospheric_adjustment)`,
//! plus a feedback on the surface temperature at the start of the step.

use crate::constants::{end, forcing_unit, one, start, surface_temperature, temperature_unit};
use rscm_components::meinshausen2020::Meinshausen2020;
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// One greenhouse gas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GhgSpecies {
    /// Concentration input.
    pub concentration: String,
    /// Concentration unit.
    pub concentration_unit: String,
    /// Forcing output.
    pub forcing: String,
    /// Reference concentration of the forcing formula.
    pub reference_concentration: FloatValue,
    /// Concentration with zero forcing.
    pub baseline_concentration: FloatValue,
    /// Forcing scaling.
    #[serde(default = "one")]
    pub forcing_scale: FloatValue,
    /// Tropospheric adjustment to the stratospherically adjusted forcing.
    #[serde(default)]
    pub tropospheric_adjustment: FloatValue,
    /// Radiative efficiency (W m-2 ppb-1). Only used for the minor gases.
    #[serde(default)]
    pub radiative_efficiency: FloatValue,
    /// Forcing per kelvin of surface temperature.
    #[serde(default)]
    pub temperature_feedback: FloatValue,
}

impl GhgSpecies {
    fn scale(&self) -> FloatValue {
        self.forcing_scale * (1.0 + self.tropospheric_adjustment)
    }
}

/// Parameters of [`FairGhgForcing`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FairGhgForcingParameters {
    /// CO2.
    pub co2: GhgSpecies,
    /// CH4.
    pub ch4: GhgSpecies,
    /// N2O.
    pub n2o: GhgSpecies,
    /// Gases with forcing linear in concentration.
    #[serde(default)]
    pub minor: Vec<GhgSpecies>,
    /// Surface temperature input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

/// FaIR's Meinshausen et al. (2020) greenhouse gas forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairGhgForcing {
    parameters: FairGhgForcingParameters,
    #[serde(skip)]
    offset: OnceLock<Vec<FloatValue>>,
}

impl FairGhgForcing {
    /// Create the component.
    pub fn from_parameters(parameters: FairGhgForcingParameters) -> Self {
        Self {
            parameters,
            offset: OnceLock::new(),
        }
    }

    fn species(&self) -> impl Iterator<Item = &GhgSpecies> {
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

#[typetag::serde]
impl Component for FairGhgForcing {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let mut definitions = vec![RequirementDefinition::scalar_input(
            &self.parameters.surface_temperature,
            &temperature_unit(),
        )];
        for s in self.species() {
            definitions.push(RequirementDefinition::scalar_input(
                &s.concentration,
                &s.concentration_unit,
            ));
            definitions.push(RequirementDefinition::scalar_output(
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
            .map(|s| end(input_state, &s.concentration))
            .collect();
        let temperature = start(input_state, &self.parameters.surface_temperature);
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

    fn gas(name: &str, reference: FloatValue) -> GhgSpecies {
        GhgSpecies {
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

    fn component() -> FairGhgForcing {
        FairGhgForcing::from_parameters(FairGhgForcingParameters {
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
