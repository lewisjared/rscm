//! Equivalent effective stratospheric chlorine (FaIR step 8)
//!
//! Calculates equivalent effective stratospheric chlorine from the concentrations of
//! ozone-depleting halogens, weighting bromine by its greater ozone depletion potency.
//!
//! $$EESC = \sum_h \Big(n_{Cl,h} C_h \frac{f_h}{f_{11}}
//!   + \beta\, n_{Br,h} C_h \frac{f_h}{f_{11}}\Big) f_{11}$$
//!
//! over halogens $h$ with concentration $C_h$, chlorine and bromine atoms $n$,
//! fractional release $f$ (CFC-11's is $f_{11}$) and bromine potency $\beta$.
//! Halogens with a missing concentration are skipped.
//!
//! # Inputs
//!
//! - Each halogen's `concentration`, in its `unit`.
//!   rscm.fair passes each `cfc-11` and `other halogen` species,
//!   such as `Atmospheric Concentrations|CFC11` in ppt.
//!
//! # Outputs
//!
//! - The equivalent effective stratospheric chlorine named by `output`, in `unit` (ppt by default).
//!   Its default name is `Equivalent Effective Stratospheric Chlorine`.

use crate::constants::get;
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::names::EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};

/// One ozone-depleting halogen.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Halogen {
    /// Name of the concentration input.
    pub concentration: String,
    /// Unit of the concentration, which is also the unit of the output.
    pub unit: String,
    /// Chlorine atoms per molecule (FaIR `cl_atoms`).
    pub cl_atoms: FloatValue,
    /// Bromine atoms per molecule (FaIR `br_atoms`).
    pub br_atoms: FloatValue,
    /// Fraction released in the stratosphere (FaIR `fractional_release`).
    pub fractional_release: FloatValue,
}

/// Parameters of [`EESC`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EESCParameters {
    /// Ozone-depleting halogens.
    pub halogens: Vec<Halogen>,
    /// Fraction of CFC-11 released in the stratosphere (FaIR `fractional_release` of CFC-11).
    pub cfc11_fractional_release: FloatValue,
    /// Ozone depletion potency of bromine relative to chlorine (FaIR `br_cl_ods_potential`).
    #[serde(default = "default_br_cl_ratio")]
    pub br_cl_ratio: FloatValue,
    /// Name of the equivalent effective stratospheric chlorine output.
    #[serde(default = "default_output")]
    pub output: String,
    /// Unit of the equivalent effective stratospheric chlorine output.
    #[serde(default = "default_unit")]
    pub unit: String,
}

fn default_br_cl_ratio() -> FloatValue {
    45.0
}

fn default_output() -> String {
    EQUIVALENT_EFFECTIVE_STRATOSPHERIC_CHLORINE.to_string()
}

fn default_unit() -> String {
    "ppt".to_string()
}

/// FaIR's equivalent effective stratospheric chlorine.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["chemistry", "halogens", "fair"], category = "Atmospheric Chemistry")]
pub struct EESC {
    parameters: EESCParameters,
}

impl EESC {
    /// Create the component.
    pub fn from_parameters(parameters: EESCParameters) -> Self {
        Self { parameters }
    }

    /// EESC for concentrations in the order of the halogens.
    pub fn eesc(&self, concentrations: &[FloatValue]) -> FloatValue {
        let p = &self.parameters;
        let fr11 = p.cfc11_fractional_release;
        p.halogens
            .iter()
            .zip(concentrations)
            .map(|(h, c)| {
                (h.cl_atoms * c * h.fractional_release / fr11
                    + p.br_cl_ratio * h.br_atoms * c * h.fractional_release / fr11)
                    * fr11
            })
            .filter(|x| !x.is_nan())
            .sum()
    }
}

#[typetag::serde(name = "fair.EESC")]
impl Component for EESC {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let p = &self.parameters;
        let mut definitions: Vec<_> = p
            .halogens
            .iter()
            .map(|h| VariableDefinition::scalar_input(&h.concentration, &h.unit))
            .collect();
        definitions.push(VariableDefinition::scalar_output(&p.output, &p.unit));
        definitions
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let concentrations = self
            .parameters
            .halogens
            .iter()
            .map(|h| get(input_state, &h.concentration))
            .collect::<Vec<_>>();
        let mut output = OutputState::with_capacity(1);
        output.insert(
            self.parameters.output.clone(),
            StateValue::Scalar(self.eesc(&concentrations)),
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cfc11_counts_its_chlorine() {
        let c = EESC::from_parameters(EESCParameters {
            halogens: vec![Halogen {
                concentration: "CFC11".into(),
                unit: "ppt".into(),
                cl_atoms: 3.0,
                br_atoms: 0.0,
                fractional_release: 0.47,
            }],
            cfc11_fractional_release: 0.47,
            br_cl_ratio: 45.0,
            output: default_output(),
            unit: default_unit(),
        });
        assert_relative_eq!(c.eesc(&[200.0]), 3.0 * 200.0 * 0.47, max_relative = 1e-14);
    }
}
