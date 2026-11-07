//! Forcing subtotals shared by the forcing aggregation components.
//!
//! Each forcing agent belongs to one [`ForcingCategory`].
//! [`ForcingSubtotals`] adds agent ERF into MAGICC7's reported subtotals:
//!
//! - `Effective Radiative Forcing|Greenhouse Gases`: CO2, CH4, N2O, F-gases and Montreal Protocol gases.
//! - `Effective Radiative Forcing|Aerosols`: direct and indirect aerosol effects.
//! - `Effective Radiative Forcing|Anthropogenic`: every agent but solar and volcanic forcing.

use crate::component::VariableDefinition;
use crate::names::{ERF_AEROSOLS, ERF_ANTHROPOGENIC, ERF_GREENHOUSE_GASES};
use crate::state::{OutputState, StateValue};
use crate::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Which of the reported forcing subtotals an agent counts towards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcingCategory {
    /// A well-mixed greenhouse gas, counted in the greenhouse gas and anthropogenic subtotals.
    GreenhouseGas,
    /// An aerosol effect, counted in the aerosol and anthropogenic subtotals.
    Aerosol,
    /// Any other anthropogenic agent, such as ozone, land use or black carbon on snow.
    Anthropogenic,
    /// Solar or volcanic forcing, counted in no subtotal.
    Natural,
}

/// Greenhouse gas, aerosol and anthropogenic ERF.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ForcingSubtotals {
    /// Greenhouse gas ERF (W/m²).
    pub greenhouse_gases: FloatValue,
    /// Aerosol ERF (W/m²).
    pub aerosols: FloatValue,
    /// Anthropogenic ERF (W/m²).
    pub anthropogenic: FloatValue,
}

impl ForcingCategory {
    /// Subtotals of one agent's ERF, zero in those it does not count towards.
    pub fn subtotals(self, erf: FloatValue) -> ForcingSubtotals {
        let (greenhouse_gases, aerosols, anthropogenic) = match self {
            Self::GreenhouseGas => (erf, 0.0, erf),
            Self::Aerosol => (0.0, erf, erf),
            Self::Anthropogenic => (0.0, 0.0, erf),
            Self::Natural => (0.0, 0.0, 0.0),
        };
        ForcingSubtotals {
            greenhouse_gases,
            aerosols,
            anthropogenic,
        }
    }
}

impl std::ops::Add for ForcingSubtotals {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            greenhouse_gases: self.greenhouse_gases + other.greenhouse_gases,
            aerosols: self.aerosols + other.aerosols,
            anthropogenic: self.anthropogenic + other.anthropogenic,
        }
    }
}

impl std::iter::Sum for ForcingSubtotals {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |total, s| total + s)
    }
}

impl ForcingSubtotals {
    /// Scalar output definitions of the subtotals.
    pub fn definitions(unit: &str) -> Vec<VariableDefinition> {
        [ERF_GREENHOUSE_GASES, ERF_AEROSOLS, ERF_ANTHROPOGENIC]
            .into_iter()
            .map(|name| VariableDefinition::scalar_output(name, unit))
            .collect()
    }

    /// Write the subtotals into a component's output.
    pub fn insert_into(&self, output: &mut OutputState) {
        for (name, value) in [
            (ERF_GREENHOUSE_GASES, self.greenhouse_gases),
            (ERF_AEROSOLS, self.aerosols),
            (ERF_ANTHROPOGENIC, self.anthropogenic),
        ] {
            output.insert(name.to_string(), StateValue::Scalar(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_count_towards_their_subtotals() {
        let subtotals: ForcingSubtotals = [
            (ForcingCategory::GreenhouseGas, 2.0),
            (ForcingCategory::Aerosol, -0.5),
            (ForcingCategory::Anthropogenic, 0.25),
            (ForcingCategory::Natural, 0.1),
        ]
        .into_iter()
        .map(|(category, erf)| category.subtotals(erf))
        .sum();
        assert_eq!(
            subtotals,
            ForcingSubtotals {
                greenhouse_gases: 2.0,
                aerosols: -0.5,
                anthropogenic: 1.75,
            }
        );
    }

    #[test]
    fn categories_read_as_snake_case() {
        let category: ForcingCategory = serde_json::from_str("\"greenhouse_gas\"").unwrap();
        assert_eq!(category, ForcingCategory::GreenhouseGas);
    }
}
