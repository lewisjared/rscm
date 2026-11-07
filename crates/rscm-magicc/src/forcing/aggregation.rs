//! Forcing Aggregation Component
//!
//! Combines the ERF of each forcing agent into the four-box efficacy-adjusted
//! forcing that drives `ClimateUdeb`, following MAGICC7's DELTAQ.
//!
//! Each agent is either global, spread over the boxes with its regional
//! pattern, or already regional. Its box forcing is then scaled by
//!
//! $$\text{EFFRF} = \text{ERF} \times \frac{\epsilon_{\text{prescribed}}}{\epsilon_{\text{internal}}}$$
//!
//! when `efficacy_method` is `ar6` (the MAGICC7 default), by $1 / \epsilon_{\text{internal}}$
//! when it is `unit`, and left unchanged when it is `none`.
//!
//! With `climate` set, the internal efficacies follow MAGICC7 in coming from
//! LAMCALC at the previous year's `Equilibrium Climate Sensitivity` and
//! `2xCO2 Effective Radiative Forcing`, so they drift as the climate warms.
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing` (W/m²): area-weighted sum of agent ERF.
//! - `Effective Radiative Forcing|Greenhouse Gases`, `|Aerosols` and `|Anthropogenic` (W/m²):
//!   the same sum over the agents of each [`ForcingCategory`](rscm_core::forcing::ForcingCategory).
//! - `Efficacy Adjusted Effective Radiative Forcing` (W/m², FourBox): the
//!   climate driver.

use crate::climate::lamcalc::{
    calc_internal_efficacy, compute_qfrac, hemispheric_qfrac, lamcalc, LamcalcParameters,
};
use crate::parameters::{EfficacyMethod, ForcingAgent, ForcingAggregationParameters, ForcingTerm};
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::{RscmError, RscmResult};
use rscm_core::forcing::ForcingSubtotals;
use rscm_core::names::{EFFICACY_ADJUSTED_ERF, EQUILIBRIUM_CLIMATE_SENSITIVITY, ERF, ERF_2XCO2};
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

/// Aggregates agent ERF into four-box efficacy-adjusted forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcingAggregation {
    parameters: ForcingAggregationParameters,
}

impl ForcingAggregation {
    /// Create the component, validating the agent list.
    pub fn from_parameters(parameters: ForcingAggregationParameters) -> RscmResult<Self> {
        for agent in &parameters.agents {
            if !(agent.efficacy.is_finite() && agent.internal_efficacy.is_finite())
                || agent.internal_efficacy <= 0.0
            {
                return Err(RscmError::Error(format!(
                    "invalid efficacy for {}: prescribed {}, internal {}",
                    agent.variable, agent.efficacy, agent.internal_efficacy
                )));
            }
        }
        validate_summation(&parameters)?;
        let component = Self { parameters };
        if let Some(climate) = &component.parameters.climate {
            component
                .internal_efficacies(climate.ecs, climate.erf_2xco2)
                .ok_or_else(|| {
                    RscmError::Error(format!(
                        "LAMCALC failed to converge for ECS={}, RLO={}",
                        climate.ecs, climate.land_ocean_warming_ratio
                    ))
                })?;
        }
        Ok(component)
    }

    /// Internal efficacy of each agent at equilibrium climate sensitivity `ecs`
    /// and 2xCO2 effective radiative forcing `erf_2xco2`.
    ///
    /// Without `climate` these are the agents' own `internal_efficacy`.
    /// Returns None if LAMCALC does not converge.
    pub fn internal_efficacies(
        &self,
        ecs: FloatValue,
        erf_2xco2: FloatValue,
    ) -> Option<Vec<FloatValue>> {
        let agents = &self.parameters.agents;
        let Some(climate) = &self.parameters.climate else {
            return Some(agents.iter().map(|a| a.internal_efficacy).collect());
        };
        let grid = self.parameters.grid();
        let result = lamcalc(&LamcalcParameters {
            erf_2xco2,
            land_ocean_heat_exchange: climate.land_ocean_heat_exchange,
            north_south_heat_exchange: climate.north_south_heat_exchange,
            ecs,
            land_ocean_warming_ratio: climate.land_ocean_warming_ratio,
            amplify_ocean_to_land: climate.amplify_ocean_to_land,
            grid: grid.clone(),
            co2_regional_pattern: climate.co2_regional_pattern,
        })?;
        Some(
            agents
                .iter()
                .map(|a| {
                    if a.use_co2_internal_efficacy {
                        return result.co2_internal_efficacy;
                    }
                    calc_internal_efficacy(
                        erf_2xco2,
                        &result.matrix_inverse,
                        grid.weights(),
                        &a.regional_pattern,
                        ecs,
                    )
                })
                .collect(),
        )
    }

    fn initial_internal_efficacies(&self) -> Vec<FloatValue> {
        let (ecs, q) = self
            .parameters
            .climate
            .as_ref()
            .map_or((FloatValue::NAN, FloatValue::NAN), |c| (c.ecs, c.erf_2xco2));
        self.internal_efficacies(ecs, q).unwrap_or_else(|| {
            self.parameters
                .agents
                .iter()
                .map(|a| a.internal_efficacy)
                .collect()
        })
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &ForcingAggregationParameters {
        &self.parameters
    }

    /// Box ERF and efficacy-adjusted box forcing from each agent's value,
    /// with the internal efficacies before the climate model has run.
    ///
    /// `values` holds four entries for a regional agent and one otherwise.
    pub fn aggregate(&self, values: &[Vec<FloatValue>]) -> (FloatValue, [FloatValue; 4]) {
        let forcing: Vec<_> = self
            .parameters
            .agents
            .iter()
            .zip(values)
            .map(|(agent, value)| self.agent_forcing(agent, std::slice::from_ref(value)))
            .collect();
        self.aggregate_with(&forcing, &self.initial_internal_efficacies())
            .0
    }

    /// Global and box ERF of each of an agent's variables.
    fn agent_forcing(&self, agent: &ForcingAgent, values: &[Vec<FloatValue>]) -> Vec<Forcing> {
        let area = *self.parameters.grid().weights();
        let qfrac = if agent.hemispheric {
            hemispheric_qfrac(&agent.regional_pattern, &area)
        } else {
            compute_qfrac(&agent.regional_pattern, &area)
        };
        values
            .iter()
            .map(|value| {
                if agent.regional {
                    let boxes: [FloatValue; 4] = std::array::from_fn(|i| value[i]);
                    let global = (0..4).fold(0.0, |sum, i| sum + boxes[i] * area[i]);
                    (global, boxes)
                } else {
                    (value[0], qfrac.map(|q| value[0] * q))
                }
            })
            .collect()
    }

    /// Total forcing and global subtotals, both added in the order of `summation`.
    fn aggregate_with(
        &self,
        forcing: &[Vec<Forcing>],
        internal: &[FloatValue],
    ) -> (Forcing, ForcingSubtotals) {
        let per_agent: Vec<(Forcing, ForcingSubtotals)> = self
            .parameters
            .agents
            .iter()
            .zip(forcing)
            .zip(internal)
            .map(|((agent, parts), internal)| {
                let scale = match self.parameters.efficacy_method {
                    EfficacyMethod::None => 1.0,
                    EfficacyMethod::Unit => 1.0 / internal,
                    EfficacyMethod::Ar6 => agent.efficacy / internal,
                };
                // Scaling leaves the global value alone, so it is the agent's unadjusted ERF.
                let scaled = |(global, boxes): Forcing| (global, boxes.map(|b| b * scale));
                let adjusted = if agent.efficacy_per_variable {
                    sum(parts.iter().copied().map(scaled))
                } else {
                    scaled(sum(parts.iter().copied()))
                };
                (adjusted, agent.category.subtotals(adjusted.0))
            })
            .collect();
        let add = |(f1, s1): (Forcing, ForcingSubtotals), (f2, s2): (Forcing, ForcingSubtotals)| {
            (sum([f1, f2].into_iter()), s1 + s2)
        };
        let p = &self.parameters;
        let total = if p.summation.is_empty() {
            per_agent.into_iter().reduce(add)
        } else {
            let index = |name: &str| p.agents.iter().position(|a| a.variable == name).unwrap();
            fold_terms(&p.summation, &|name| per_agent[index(name)], &add)
        };
        total.unwrap_or_default()
    }
}

/// Global ERF and efficacy-adjusted (or plain) box forcing.
type Forcing = (FloatValue, [FloatValue; 4]);

fn sum(terms: impl Iterator<Item = Forcing>) -> Forcing {
    terms
        .reduce(|(global, boxes), (g, b)| (global + g, std::array::from_fn(|i| boxes[i] + b[i])))
        .unwrap_or((0.0, [0.0; 4]))
}

fn fold_terms<T: Copy>(
    terms: &[ForcingTerm],
    agent: &dyn Fn(&str) -> T,
    add: &dyn Fn(T, T) -> T,
) -> Option<T> {
    terms
        .iter()
        .filter_map(|term| match term {
            ForcingTerm::Agent(name) => Some(agent(name)),
            ForcingTerm::Sum(nested) => fold_terms(nested, agent, add),
        })
        .reduce(add)
}

/// Check that `summation`, when given, names every agent exactly once.
fn validate_summation(parameters: &ForcingAggregationParameters) -> RscmResult<()> {
    fn names<'a>(terms: &'a [ForcingTerm], out: &mut Vec<&'a str>) {
        for term in terms {
            match term {
                ForcingTerm::Agent(name) => out.push(name),
                ForcingTerm::Sum(nested) => names(nested, out),
            }
        }
    }
    if parameters.summation.is_empty() {
        return Ok(());
    }
    let mut listed = Vec::new();
    names(&parameters.summation, &mut listed);
    let mut agents: Vec<&str> = parameters
        .agents
        .iter()
        .map(|a| a.variable.as_str())
        .collect();
    listed.sort_unstable();
    agents.sort_unstable();
    if listed != agents {
        return Err(RscmError::Error(format!(
            "summation must name each agent once: {listed:?} against agents {agents:?}"
        )));
    }
    Ok(())
}

/// Variables read for an agent.
fn agent_inputs(agent: &ForcingAgent) -> Vec<&str> {
    if agent.sum_of.is_empty() {
        vec![agent.variable.as_str()]
    } else {
        agent.sum_of.iter().map(String::as_str).collect()
    }
}

#[typetag::serde(name = "magicc.ForcingAggregation")]
impl Component for ForcingAggregation {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions: Vec<_> = self
            .parameters
            .agents
            .iter()
            .flat_map(|agent| {
                agent_inputs(agent).into_iter().map(|name| {
                    if agent.regional {
                        VariableDefinition::four_box_input(name, "W/m^2")
                    } else {
                        VariableDefinition::scalar_input(name, "W/m^2")
                    }
                })
            })
            .collect();
        if self.parameters.climate.is_some() {
            definitions.push(VariableDefinition::scalar_input(
                EQUILIBRIUM_CLIMATE_SENSITIVITY,
                "K",
            ));
            definitions.push(VariableDefinition::scalar_input(ERF_2XCO2, "W/m^2"));
        }
        definitions.push(VariableDefinition::scalar_output(ERF, "W/m^2"));
        definitions.extend(ForcingSubtotals::definitions("W/m^2"));
        definitions.push(VariableDefinition::four_box_output(
            EFFICACY_ADJUSTED_ERF,
            "W/m^2",
        ));
        definitions
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let forcing: Vec<Vec<Forcing>> = self
            .parameters
            .agents
            .iter()
            .map(|agent| {
                let values = agent_inputs(agent)
                    .into_iter()
                    .map(|name| {
                        if agent.regional {
                            input_state.get_four_box_window(name).get_all()
                        } else {
                            vec![input_state.get_scalar_window(name).get()]
                        }
                    })
                    .collect::<Vec<Vec<FloatValue>>>();
                self.agent_forcing(agent, &values)
            })
            .collect();
        let internal = match &self.parameters.climate {
            None => self.initial_internal_efficacies(),
            Some(climate) => {
                let ecs = input_state
                    .get_scalar_window(EQUILIBRIUM_CLIMATE_SENSITIVITY)
                    .get();
                let (q, _) = input_state.get_scalar_window(ERF_2XCO2).interval()?;
                let (ecs, q) = if ecs.is_finite() && q.is_finite() {
                    (ecs, q)
                } else {
                    (climate.ecs, climate.erf_2xco2)
                };
                self.internal_efficacies(ecs, q).ok_or_else(|| {
                    RscmError::Error(format!("LAMCALC failed to converge for ECS={ecs}"))
                })?
            }
        };
        let ((total, adjusted), subtotals) = self.aggregate_with(&forcing, &internal);
        let mut output = OutputState::new();
        output.insert(ERF.to_string(), StateValue::Scalar(total));
        subtotals.insert_into(&mut output);
        output.insert(
            EFFICACY_ADJUSTED_ERF.to_string(),
            StateValue::FourBox(FourBoxSlice::from_array(adjusted)),
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::{EfficacyClimate, ForcingAgent};
    use rscm_core::forcing::ForcingCategory;
    use rscm_core::state::{ReadTransformInfo, TransformContext, VariableSource};
    use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem};
    use std::collections::HashMap;

    fn agent(regional: bool, efficacy: FloatValue, internal: FloatValue) -> ForcingAgent {
        ForcingAgent {
            variable: "x".into(),
            category: ForcingCategory::Anthropogenic,
            regional,
            sum_of: Vec::new(),
            efficacy_per_variable: false,
            regional_pattern: [1.0, 2.0, 1.0, 2.0],
            hemispheric: false,
            use_co2_internal_efficacy: false,
            efficacy,
            internal_efficacy: internal,
        }
    }

    fn component(agents: Vec<ForcingAgent>, efficacy_method: EfficacyMethod) -> ForcingAggregation {
        ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents,
            efficacy_method,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn global_agent_keeps_global_mean() {
        let c = component(vec![agent(false, 1.0, 1.0)], EfficacyMethod::Ar6);
        let (total, boxes) = c.aggregate(&[vec![2.0]]);
        let area = *c.parameters().grid().weights();
        let mean: FloatValue = (0..4).map(|i| boxes[i] * area[i]).sum();
        assert!((total - 2.0).abs() < 1e-12);
        assert!((mean - 2.0).abs() < 1e-12);
        assert!((boxes[1] / boxes[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn hemispheric_agent_keeps_hemispheric_means() {
        let mut tropoz = agent(false, 1.0, 1.0);
        tropoz.regional_pattern = [0.46565, 0.51646, 0.17687, 0.23793];
        tropoz.hemispheric = true;
        let c = component(vec![tropoz], EfficacyMethod::None);
        let (total, boxes) = c.aggregate(&[vec![1.0]]);
        let expected = [0.95618, 1.06051, 0.9324, 1.25429];
        for i in 0..4 {
            assert!((boxes[i] - expected[i]).abs() < 1e-5, "{boxes:?}");
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn efficacy_methods_scale_box_forcing() {
        let values = [vec![1.0, 2.0, 3.0, 4.0]];
        let adjusted = |method| component(vec![agent(true, 1.5, 1.2)], method).aggregate(&values);
        let (total, none) = adjusted(EfficacyMethod::None);
        assert_eq!(none, [1.0, 2.0, 3.0, 4.0]);
        assert!((adjusted(EfficacyMethod::Unit).1[3] - 4.0 / 1.2).abs() < 1e-12);
        assert!((adjusted(EfficacyMethod::Ar6).1[3] - 5.0).abs() < 1e-12);
        assert_eq!(adjusted(EfficacyMethod::Ar6).0, total);
    }

    fn with_climate(agents: Vec<ForcingAgent>) -> ForcingAggregation {
        ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents,
            climate: Some(EfficacyClimate::default()),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn climate_internal_efficacies_match_udeb() {
        let udeb = crate::climate::ClimateUdeb::new();
        let co2 = EfficacyClimate::default().co2_regional_pattern;
        let mut co2_agent = agent(false, 1.0, 1.0);
        co2_agent.regional_pattern = co2;
        co2_agent.use_co2_internal_efficacy = true;
        let pattern = agent(false, 1.0, 1.0).regional_pattern;
        let c = with_climate(vec![agent(false, 1.0, 1.0), co2_agent]);
        let climate = c.parameters().climate.clone().unwrap();

        let internal = c
            .internal_efficacies(climate.ecs, climate.erf_2xco2)
            .unwrap();
        assert!((internal[0] - udeb.internal_efficacy(&pattern)).abs() < 1e-12);
        assert_eq!(internal[1], udeb.co2_internal_efficacy());
    }

    #[test]
    fn internal_efficacies_drift_with_sensitivity() {
        let c = with_climate(vec![agent(false, 1.0, 1.0)]);
        let low = c.internal_efficacies(2.0, 3.71).unwrap()[0];
        let high = c.internal_efficacies(4.5, 3.71).unwrap()[0];
        assert!((low - high).abs() > 1e-4, "{low} {high}");

        // Without climate the configured internal efficacy holds at any sensitivity
        let fixed = component(vec![agent(false, 1.0, 1.3)], EfficacyMethod::Ar6);
        assert_eq!(fixed.internal_efficacies(2.0, 3.71).unwrap(), vec![1.3]);
        assert_eq!(fixed.internal_efficacies(4.5, 3.71).unwrap(), vec![1.3]);
    }

    fn scalar_item(name: &str, values: [FloatValue; 2]) -> TimeseriesItem {
        TimeseriesItem {
            data: TimeseriesData::Scalar(rscm_core::timeseries::Timeseries::from_values(
                ndarray::Array1::from_vec(values.to_vec()),
                ndarray::array![2000.0, 2001.0],
            )),
            name: name.to_string(),
        }
    }

    fn solve_adjusted(c: &ForcingAggregation, ecs: [FloatValue; 2]) -> [FloatValue; 4] {
        let items = [
            scalar_item("x", [1.0, 1.0]),
            scalar_item(EQUILIBRIUM_CLIMATE_SENSITIVITY, ecs),
            scalar_item(ERF_2XCO2, [3.9, 4.2]),
        ];
        // MAGICC7 lags the sensitivity into the aggregation
        let lagged = TransformContext {
            read_transforms: HashMap::from([(
                EQUILIBRIUM_CLIMATE_SENSITIVITY.to_string(),
                ReadTransformInfo::default().with_source(VariableSource::Lagged),
            )]),
        };
        let input_state = InputState::build_with_transforms(items.iter().collect(), 2000.0, lagged);
        let output = c.solve(2000.0, 2001.0, &input_state).unwrap();
        match &output[EFFICACY_ADJUSTED_ERF] {
            StateValue::FourBox(slice) => slice.0,
            other => panic!("expected four-box forcing, got {other:?}"),
        }
    }

    #[test]
    fn solve_takes_internal_efficacy_from_start_of_step() {
        let c = with_climate(vec![agent(false, 1.5, 1.0)]);
        let expected = |ecs: FloatValue, q: FloatValue| {
            let forcing = [c.agent_forcing(&c.parameters().agents[0], &[vec![1.0]])];
            c.aggregate_with(&forcing, &c.internal_efficacies(ecs, q).unwrap())
                .0
                 .1
        };

        let adjusted = solve_adjusted(&c, [2.0, 4.5]);
        assert_eq!(adjusted, expected(2.0, 3.9));
        assert!((adjusted[0] - expected(4.5, 4.2)[0]).abs() > 1e-4);

        // Before the climate model reports a sensitivity the initial climate is used
        let climate = c.parameters().climate.clone().unwrap();
        let adjusted = solve_adjusted(&c, [FloatValue::NAN, 4.5]);
        assert_eq!(adjusted, expected(climate.ecs, climate.erf_2xco2));
    }

    fn named(variable: &str, regional: bool) -> ForcingAgent {
        ForcingAgent {
            variable: variable.into(),
            ..agent(regional, 1.0, 1.0)
        }
    }

    #[test]
    fn summation_adds_nested_terms_first() {
        // 1 + (1e-16 + 1e-16) keeps the small terms that a left-to-right sum loses
        let agents = vec![named("a", false), named("b", false), named("c", false)];
        let flat = component(agents.clone(), EfficacyMethod::None);
        let values = [vec![1.0], vec![1e-16], vec![1e-16]];
        assert_eq!(flat.aggregate(&values).0, 1.0);
        let nested = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents,
            efficacy_method: EfficacyMethod::None,
            summation: vec![
                ForcingTerm::Agent("a".into()),
                ForcingTerm::Sum(vec![
                    ForcingTerm::Agent("b".into()),
                    ForcingTerm::Agent("c".into()),
                ]),
            ],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(nested.aggregate(&values).0, 1.0 + 2e-16);
    }

    #[test]
    fn rejects_summation_missing_an_agent() {
        let result = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![named("a", false), named("b", false)],
            summation: vec![ForcingTerm::Agent("a".into())],
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn sum_of_adds_variables_before_efficacy() {
        let mut biomass = named("biomass", true);
        biomass.sum_of = vec!["p".into(), "q".into()];
        biomass.efficacy = 2.0;
        let c = component(vec![biomass], EfficacyMethod::Ar6);
        let items = [four_box_item("p", 1.0), four_box_item("q", 2.0)];
        let input_state = InputState::build(items.iter().collect(), 2000.0);
        let output = c.solve(2000.0, 2001.0, &input_state).unwrap();
        match &output[EFFICACY_ADJUSTED_ERF] {
            StateValue::FourBox(slice) => assert_eq!(slice.0, [6.0; 4]),
            other => panic!("expected four-box forcing, got {other:?}"),
        }
    }

    #[test]
    fn subtotals_sum_global_erf_by_category() {
        let categorised = |variable, category| ForcingAgent {
            category,
            efficacy: 2.0,
            ..named(variable, false)
        };
        let c = component(
            vec![
                categorised("co2", ForcingCategory::GreenhouseGas),
                categorised("aer", ForcingCategory::Aerosol),
                categorised("o3", ForcingCategory::Anthropogenic),
                categorised("solar", ForcingCategory::Natural),
            ],
            EfficacyMethod::Ar6,
        );
        let items = [
            scalar_item("co2", [2.0, 2.0]),
            scalar_item("aer", [-0.5, -0.5]),
            scalar_item("o3", [0.25, 0.25]),
            scalar_item("solar", [0.125, 0.125]),
        ];
        let input_state = InputState::build(items.iter().collect(), 2000.0);
        let output = c.solve(2000.0, 2001.0, &input_state).unwrap();
        let scalar = |name: &str| match &output[name] {
            StateValue::Scalar(value) => *value,
            other => panic!("expected a scalar, got {other:?}"),
        };
        assert_eq!(scalar(rscm_core::names::ERF_GREENHOUSE_GASES), 2.0);
        assert_eq!(scalar(rscm_core::names::ERF_AEROSOLS), -0.5);
        assert_eq!(scalar(rscm_core::names::ERF_ANTHROPOGENIC), 1.75);
        assert_eq!(scalar(ERF), 1.875);
    }

    fn four_box_item(name: &str, value: FloatValue) -> TimeseriesItem {
        let grid = rscm_core::spatial::FourBoxGrid::magicc_standard();
        TimeseriesItem {
            data: TimeseriesData::FourBox(rscm_core::timeseries::GridTimeseries::new(
                ndarray::Array2::from_elem((2, 4), value),
                std::sync::Arc::new(rscm_core::timeseries::TimeAxis::from_values(
                    ndarray::array![2000.0, 2001.0],
                )),
                grid,
                "W/m^2".to_string(),
                rscm_core::interpolate::strategies::InterpolationStrategy::from(
                    rscm_core::interpolate::strategies::LinearSplineStrategy::new(true),
                ),
            )),
            name: name.to_string(),
        }
    }

    #[test]
    fn rejects_nonpositive_internal_efficacy() {
        let result = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![agent(false, 1.0, 0.0)],
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
