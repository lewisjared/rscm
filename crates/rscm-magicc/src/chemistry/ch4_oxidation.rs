//! CH4 Oxidation Component
//!
//! Tracks atmospheric methane by source and the CO2 released as its fossil share
//! is oxidised, which MAGICC7 adds to fossil CO2 emissions.
//!
//! # Inputs
//!
//! - `Atmospheric Concentrations|CH4` (ppb), at the start and end of the step
//! - `Lifetime|CH4|OH` (yr): OH lifetime over the step
//! - `Emissions|CH4|Fossil` (Mt CH4/yr) - Fossil and industrial CH4 emissions
//! - `Emissions|CH4|Biosphere` (Mt CH4/yr) - Agriculture and land use CH4 emissions
//! - `Emissions|CH4|Permafrost` (Mt CH4/yr), only with the `include_permafrost` parameter
//!
//! # States (tracked between steps)
//!
//! - `Atmospheric Burden|CH4|{Fossil,Biosphere,Natural,Permafrost}` (Mt CH4): CH4 in the atmosphere by source
//!
//! # Outputs
//!
//! - `Emissions|CO2|CH4 Oxidation` (GtC/yr) - CO2 released by oxidation
//!
//! As in MAGICC7, the sink is the mid-step concentration over the total lifetime,
//! shared between the pools in proportion to their size.
//! Anthropogenic pools receive the emissions that close the CH4 budget,
//! split between fossil and biosphere in proportion to their emissions.
//! The fossil share counts the fossil pool and `permafrost_fraction` of the permafrost pool.

use crate::parameters::CH4OxidationParameters;
use crate::reference::{permafrost_emissions, require_annual};
use rscm_core::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::RscmResult;
use rscm_core::names::EMISSIONS_CH4_PERMAFROST;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["chemistry", "ch4", "carbon-cycle", "magicc"], category = "Atmospheric Chemistry")]
#[inputs(
    concentration { name = "Atmospheric Concentrations|CH4", unit = "ppb" },
    oh_lifetime { name = "Lifetime|CH4|OH", unit = "yr", timeseries_type = "average" },
    fossil_emissions { name = "Emissions|CH4|Fossil", unit = "Mt CH4/yr" },
    biosphere_emissions { name = "Emissions|CH4|Biosphere", unit = "Mt CH4/yr" },
)]
#[states(
    fossil_burden { name = "Atmospheric Burden|CH4|Fossil", unit = "Mt CH4" },
    biosphere_burden { name = "Atmospheric Burden|CH4|Biosphere", unit = "Mt CH4" },
    natural_burden { name = "Atmospheric Burden|CH4|Natural", unit = "Mt CH4" },
    permafrost_burden { name = "Atmospheric Burden|CH4|Permafrost", unit = "Mt CH4" },
)]
#[outputs(
    co2_emissions { name = "Emissions|CO2|CH4 Oxidation", unit = "GtC/yr" },
)]
pub struct CH4Oxidation {
    parameters: CH4OxidationParameters,
}

/// CH4 burden (Mt CH4) by source
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CH4Pools {
    pub fossil: FloatValue,
    pub biosphere: FloatValue,
    pub natural: FloatValue,
    pub permafrost: FloatValue,
}

impl CH4Pools {
    fn total(&self) -> FloatValue {
        self.fossil + self.biosphere + self.natural + self.permafrost
    }
}

/// CH4 emissions (Mt CH4/yr) over a step, by source
#[derive(Debug, Clone, Copy)]
pub struct CH4Sources {
    pub fossil: FloatValue,
    pub biosphere: FloatValue,
    pub permafrost: FloatValue,
}

impl CH4Oxidation {
    pub fn new() -> Self {
        Self::from_parameters(CH4OxidationParameters::default())
    }

    pub fn from_parameters(parameters: CH4OxidationParameters) -> Self {
        Self { parameters }
    }

    /// Pools at the end of an annual step from `start` to `end` ppb, and the CO2 released (GtC/yr).
    pub fn solve_oxidation(
        &self,
        pools: CH4Pools,
        sources: CH4Sources,
        concentration: (FloatValue, FloatValue),
        oh_lifetime: FloatValue,
    ) -> (CH4Pools, FloatValue) {
        let p = &self.parameters;
        let (start, end) = concentration;
        let sink_ppb = (1.0 / oh_lifetime + 1.0 / p.non_oh_lifetime) * ((start + end) * 0.5);
        let budget = (end - start + sink_ppb) * p.tg_per_ppb * p.mixing_box_size
            - p.natural_emissions
            - sources.permafrost;
        let anthropogenic = sources.fossil + sources.biosphere;
        let (fossil_share, biosphere_share) = if anthropogenic != 0.0 {
            (sources.fossil, sources.biosphere)
        } else if pools.fossil + pools.biosphere != 0.0 {
            (pools.fossil, pools.biosphere)
        } else {
            // MAGICC7 takes the split of the first year with emissions, which a step cannot see.
            (0.0, 1.0)
        };
        let split = fossil_share + biosphere_share;

        let total = pools.total();
        let sink = sink_ppb * p.tg_per_ppb * p.mixing_box_size;
        let decay = |pool: FloatValue| pool * sink / total;
        let next = CH4Pools {
            fossil: fossil_share * budget / split + pools.fossil - decay(pools.fossil),
            biosphere: budget * biosphere_share / split + pools.biosphere - decay(pools.biosphere),
            natural: p.natural_emissions + pools.natural - decay(pools.natural),
            permafrost: sources.permafrost + pools.permafrost - decay(pools.permafrost),
        };
        let fossil_fraction = (p.permafrost_fraction * pools.permafrost + pools.fossil) / total;
        let co2 = p.tg_per_ppb * fossil_fraction * p.mixing_box_size * sink_ppb / 1000.0
            * p.carbon_per_ch4;
        (next, co2)
    }
}

impl Default for CH4Oxidation {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde(name = "magicc.CH4Oxidation")]
impl Component for CH4Oxidation {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = Self::generated_definitions();
        if self.parameters.include_permafrost {
            definitions.push(VariableDefinition::scalar_input(
                EMISSIONS_CH4_PERMAFROST,
                "Mt CH4/yr",
            ));
        }
        definitions
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        require_annual("CH4Oxidation", t_current, t_next)?;
        let inputs = CH4OxidationInputs::from_input_state(input_state);
        let pools = CH4Pools {
            fossil: inputs.fossil_burden.get(),
            biosphere: inputs.biosphere_burden.get(),
            natural: inputs.natural_burden.get(),
            permafrost: inputs.permafrost_burden.get(),
        };
        let sources = CH4Sources {
            fossil: inputs.fossil_emissions.get(),
            biosphere: inputs.biosphere_emissions.get(),
            permafrost: permafrost_emissions(
                self.parameters.include_permafrost,
                input_state,
                EMISSIONS_CH4_PERMAFROST,
            ),
        };
        let (next, co2_emissions) = self.solve_oxidation(
            pools,
            sources,
            inputs.concentration.interval()?,
            inputs.oh_lifetime.get(),
        );
        Ok(CH4OxidationOutputs {
            fossil_burden: next.fossil,
            biosphere_burden: next.biosphere,
            natural_burden: next.natural,
            permafrost_burden: next.permafrost,
            co2_emissions,
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONV: FloatValue = 2.824 * 0.973;

    fn pools(fossil: FloatValue, biosphere: FloatValue, natural: FloatValue) -> CH4Pools {
        CH4Pools {
            fossil,
            biosphere,
            natural,
            permafrost: 0.0,
        }
    }

    fn sources(fossil: FloatValue, biosphere: FloatValue) -> CH4Sources {
        CH4Sources {
            fossil,
            biosphere,
            permafrost: 0.0,
        }
    }

    #[test]
    fn test_pools_track_concentration() {
        let start = pools(100.0, 300.0, 700.0 * CONV - 400.0);
        let (next, _) =
            CH4Oxidation::new().solve_oxidation(start, sources(80.0, 160.0), (700.0, 720.0), 9.0);
        assert!((next.total() - 720.0 * CONV).abs() < 1e-9);
        // Every pool loses the same fraction, and the anthropogenic budget is split 1:2
        let lost = (182.1908 + start.natural - next.natural) / start.natural;
        let added = |next: FloatValue, start: FloatValue| next - start * (1.0 - lost);
        assert!(
            (added(next.biosphere, start.biosphere) / added(next.fossil, start.fossil) - 2.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_only_fossil_share_is_oxidised() {
        let component = CH4Oxidation::new();
        let total_lifetime = 1.0 / (1.0 / 9.0 + 1.0 / component.parameters.non_oh_lifetime);
        let sink = 700.0 * CONV / total_lifetime;
        let (_, all_fossil) = component.solve_oxidation(
            pools(700.0 * CONV, 0.0, 0.0),
            sources(1.0, 0.0),
            (700.0, 700.0),
            9.0,
        );
        assert!((all_fossil - sink * 0.75 / 1000.0).abs() < 1e-12);
        let (_, quarter) = component.solve_oxidation(
            pools(175.0 * CONV, 0.0, 525.0 * CONV),
            sources(1.0, 0.0),
            (700.0, 700.0),
            9.0,
        );
        assert!((quarter - all_fossil / 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_no_anthropogenic_emissions_split_by_pools() {
        let (next, _) = CH4Oxidation::new().solve_oxidation(
            pools(100.0, 300.0, 1000.0),
            sources(0.0, 0.0),
            (1400.0 / CONV, 1500.0 / CONV),
            9.0,
        );
        let lost = (182.1908 + 1000.0 - next.natural) / 1000.0;
        let added = |next: FloatValue, start: FloatValue| next - start * (1.0 - lost);
        assert!((added(next.biosphere, 300.0) / added(next.fossil, 100.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_definitions() {
        let names: Vec<_> = CH4Oxidation::new()
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        for name in [
            "Emissions|CO2|CH4 Oxidation",
            "Atmospheric Burden|CH4|Natural",
            "Atmospheric Concentrations|CH4",
            "Lifetime|CH4|OH",
            "Emissions|CH4|Biosphere",
        ] {
            assert!(names.contains(&name.to_string()), "{name}");
        }
        assert!(!names.contains(&"Emissions|CH4|Permafrost".to_string()));
    }

    #[test]
    fn test_solve_reads_concentration_change() {
        use rscm_core::timeseries_collection::TimeseriesCollection;
        let mut collection = TimeseriesCollection::new();
        let mut item = |name: &str, start: FloatValue, end: FloatValue| {
            collection.add_timeseries(
                name.to_string(),
                rscm_core::timeseries::Timeseries::from_values(
                    ndarray::array![start, end],
                    ndarray::array![2000.0, 2001.0],
                ),
            )
        };
        item("Atmospheric Concentrations|CH4", 700.0, 720.0);
        item("Lifetime|CH4|OH", 9.0, 9.0);
        item("Emissions|CH4|Fossil", 80.0, 80.0);
        item("Emissions|CH4|Biosphere", 160.0, 160.0);
        item("Atmospheric Burden|CH4|Fossil", 100.0, 100.0);
        item("Atmospheric Burden|CH4|Biosphere", 300.0, 300.0);
        item("Atmospheric Burden|CH4|Natural", 700.0 * CONV - 400.0, 0.0);
        item("Atmospheric Burden|CH4|Permafrost", 0.0, 0.0);
        let component = CH4Oxidation::new();
        let input_state = rscm_core::model::extract_state(&collection, component.inputs(), 2000.0);
        let output = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let (next, co2) = component.solve_oxidation(
            pools(100.0, 300.0, 700.0 * CONV - 400.0),
            sources(80.0, 160.0),
            (700.0, 720.0),
            9.0,
        );
        let get = |name: &str| output[name].as_scalar().unwrap();
        assert_eq!(get("Emissions|CO2|CH4 Oxidation"), co2);
        assert_eq!(get("Atmospheric Burden|CH4|Fossil"), next.fossil);
        assert_eq!(get("Atmospheric Burden|CH4|Natural"), next.natural);
    }
}
