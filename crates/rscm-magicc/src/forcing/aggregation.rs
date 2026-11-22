//! Forcing Aggregation Component
//!
//! Combines the ERF of each forcing agent into the four-box efficacy-adjusted
//! ERF that drives `ClimateUDEB`, following MAGICC7's DELTAQ.
//!
//! Each agent is either global, spread over the regions with its regional
//! pattern, or already regional. Its regional ERF is then scaled by
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
//! The `agents` parameter names the variables it reads and has no default agents.
//! `rscm.magicc.full.forcing_agents` builds MAGICC7's list.
//!
//! # Inputs
//!
//! - `Effective Radiative Forcing|{agent}` (W/m^2): the ERF of each agent, named by its `variable`,
//!   or each variable in its `sum_of`. It is on the four-box grid when the agent is `regional`.
//! - `Equilibrium Climate Sensitivity` (K), only with the `climate` parameter:
//!   the previous year's value, for the internal efficacies
//! - `2xCO2 Effective Radiative Forcing` (W/m^2), only with the `climate` parameter:
//!   the previous year's value, for the internal efficacies
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing` (W/m^2): area-weighted sum of the agent ERF
//! - `Effective Radiative Forcing|{Greenhouse Gases,Aerosols,Anthropogenic}` (W/m^2):
//!   the same sum over the agents of each [`ForcingCategory`](rscm_core::forcing::ForcingCategory)
//! - `Efficacy Adjusted Effective Radiative Forcing` (W/m^2, four-box): the ERF that drives the climate

use super::REGIONS;
use crate::climate::lamcalc::{
    calc_internal_efficacy, compute_qfrac, hemispheric_qfrac, lamcalc, LAMCALCParameters,
};
use crate::parameters::{EfficacyMethod, ForcingAgent, ForcingAggregationParameters, ForcingTerm};
use rscm_core::component::{
    Component, ComponentState, InputState, OutputWriter, VariableDefinition,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::forcing::ForcingSubtotals;
use rscm_core::names::{EFFICACY_ADJUSTED_ERF, EQUILIBRIUM_CLIMATE_SENSITIVITY, ERF, ERF_2XCO2};
use rscm_core::state::{FourBoxSlice, VariableSource};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::OnceLock;

/// Aggregates agent ERF into four-box efficacy-adjusted forcing.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["forcing", "aggregation", "magicc"], category = "Radiative Forcing")]
pub struct ForcingAggregation {
    parameters: ForcingAggregationParameters,
}

impl ForcingAggregation {
    /// Create the component, validating the agent list.
    pub fn from_parameters(parameters: ForcingAggregationParameters) -> RSCMResult<Self> {
        for agent in &parameters.agents {
            if !(agent.efficacy.is_finite() && agent.internal_efficacy.is_finite())
                || agent.internal_efficacy <= 0.0
            {
                return Err(RSCMError::Error(format!(
                    "invalid efficacy for {}: prescribed {}, internal {}",
                    agent.variable, agent.efficacy, agent.internal_efficacy
                )));
            }
        }
        validate_summation(&parameters)?;
        validate_scaling(&parameters)?;
        let component = Self { parameters };
        if let Some(climate) = &component.parameters.climate {
            component
                .internal_efficacies(climate.ecs, climate.erf_2xco2)
                .ok_or_else(|| {
                    RSCMError::Error(format!(
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
        let patterns: Vec<_> = self
            .parameters
            .agents
            .iter()
            .map(|a| a.regional_pattern)
            .collect();
        self.internal_efficacies_with(ecs, erf_2xco2, &patterns)
    }

    /// Internal efficacies as `internal_efficacies`,
    /// with the patterns of agents with `pattern_terms` from the ERF `lookup` reads.
    pub fn internal_efficacies_from(
        &self,
        ecs: FloatValue,
        erf_2xco2: FloatValue,
        lookup: &Lookup,
    ) -> RSCMResult<Option<Vec<FloatValue>>> {
        let k = self.constants();
        let patterns = self.patterns_with(lookup, &k)?;
        Ok(self.internal_efficacies_with(ecs, erf_2xco2, patterns))
    }

    /// Pattern of each agent: its `pattern_terms` summed in `pattern_year`, or `regional_pattern`.
    pub fn patterns(&self, lookup: &Lookup) -> RSCMResult<Vec<[FloatValue; 4]>> {
        Ok(self.patterns_with(lookup, &self.constants())?.clone())
    }

    fn constants(&self) -> Constants {
        let n = self
            .parameters
            .agents
            .iter()
            .map(|a| agent_inputs(a).len())
            .sum();
        Constants {
            held: (0..n).map(|_| OnceLock::new()).collect(),
            base: (0..n).map(|_| OnceLock::new()).collect(),
            ..Default::default()
        }
    }

    fn patterns_with<'k>(
        &self,
        lookup: &Lookup,
        k: &'k Constants,
    ) -> RSCMResult<&'k Vec<[FloatValue; 4]>> {
        memo_ref(&k.patterns, || self.compute_patterns(lookup, k))
    }

    fn compute_patterns(&self, lookup: &Lookup, k: &Constants) -> RSCMResult<Vec<[FloatValue; 4]>> {
        let agents = &self.parameters.agents;
        let mut wanted: Vec<(&str, Time)> = Vec::new();
        for agent in agents {
            let mut names = Vec::new();
            term_names(&agent.pattern_terms, &mut names);
            if let Some(year) = agent.pattern_year {
                wanted.extend(names.into_iter().map(|name| (name, year)));
            }
        }
        let mut boxes: Vec<((String, Time), [FloatValue; 4])> = Vec::new();
        let mut years: Vec<Time> = wanted.iter().map(|(_, year)| *year).collect();
        years.sort_by(FloatValue::total_cmp);
        years.dedup();
        for year in years {
            let has = |name: &str| wanted.contains(&(name, year));
            for (name, values) in self.variable_box_forcing_with(lookup, &has, &[year], k)? {
                boxes.push(((name, year), values[0]));
            }
        }
        let at = |name: &str, year: Time| {
            boxes
                .iter()
                .find(|((n, y), _)| n == name && *y == year)
                .map_or([FloatValue::NAN; 4], |(_, b)| *b)
        };
        Ok(agents
            .iter()
            .map(|agent| match agent.pattern_year {
                Some(year) => fold_terms(&agent.pattern_terms, &|name| at(name, year), &|a, b| {
                    std::array::from_fn(|i| a[i] + b[i])
                })
                .unwrap_or(agent.regional_pattern),
                None => agent.regional_pattern,
            })
            .collect())
    }

    fn internal_efficacies_with(
        &self,
        ecs: FloatValue,
        erf_2xco2: FloatValue,
        patterns: &[[FloatValue; 4]],
    ) -> Option<Vec<FloatValue>> {
        let agents = &self.parameters.agents;
        let Some(climate) = &self.parameters.climate else {
            return Some(agents.iter().map(|a| a.internal_efficacy).collect());
        };
        let grid = self.parameters.grid();
        let result = lamcalc(&LAMCALCParameters {
            erf_2xco2,
            land_ocean_heat_exchange: climate.land_ocean_heat_exchange,
            north_south_heat_exchange: climate.north_south_heat_exchange,
            ecs,
            land_ocean_warming_ratio: climate.land_ocean_warming_ratio,
            amplify_ocean_to_land: climate.amplify_ocean_to_land,
            grid: grid.clone(),
            co2_regional_pattern: climate.co2_regional_pattern,
            include_unset_lamcalc_error: climate.include_unset_lamcalc_error,
        })?;
        Some(
            agents
                .iter()
                .zip(patterns)
                .map(|(a, pattern)| {
                    if a.use_co2_internal_efficacy {
                        return result.co2_internal_efficacy;
                    }
                    calc_internal_efficacy(
                        erf_2xco2,
                        &result.matrix_inverse,
                        grid.weights(),
                        pattern,
                        ecs,
                    )
                })
                .collect(),
        )
    }

    fn initial_internal_efficacies<'k>(
        &self,
        lookup: &Lookup,
        k: &'k Constants,
    ) -> RSCMResult<&'k Vec<FloatValue>> {
        memo_ref(&k.initial_internal, || {
            let (ecs, q) = self
                .parameters
                .climate
                .as_ref()
                .map_or((FloatValue::NAN, FloatValue::NAN), |c| (c.ecs, c.erf_2xco2));
            let patterns = self.patterns_with(lookup, k)?;
            Ok(self
                .internal_efficacies_with(ecs, q, patterns)
                .unwrap_or_else(|| {
                    self.parameters
                        .agents
                        .iter()
                        .map(|a| a.internal_efficacy)
                        .collect()
                }))
        })
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &ForcingAggregationParameters {
        &self.parameters
    }

    /// Box ERF and efficacy-adjusted box forcing at time `t` from each agent variable's value,
    /// with the internal efficacies before the climate model has run.
    ///
    /// `values` holds each agent's variables in turn (`sum_of`, or `variable` itself),
    /// four entries for a regional agent and one otherwise.
    /// `lookup` gives an agent's prescribed variable at other times,
    /// which harmonisation and `constant_after_year` read.
    pub fn aggregate(
        &self,
        values: &[Vec<FloatValue>],
        lookup: &Lookup,
        t: Time,
    ) -> RSCMResult<(FloatValue, [FloatValue; 4])> {
        let k = self.constants();
        let forcing = self.variable_forcing(values, lookup, t, &k)?;
        Ok(self
            .aggregate_with(
                &forcing,
                &|agent| agent_inputs(agent).len(),
                self.initial_internal_efficacies(lookup, &k)?,
            )
            .0)
    }

    /// Scale of each agent's variables, in the order of `aggregate`'s values.
    ///
    /// A harmonised agent's scale comes from its prescribed variable, which `lookup` reads.
    pub fn variable_scales(&self, lookup: &Lookup) -> RSCMResult<Vec<FloatValue>> {
        Ok(self.scales_with(lookup, &self.constants())?.clone())
    }

    fn scales_with<'k>(
        &self,
        lookup: &Lookup,
        k: &'k Constants,
    ) -> RSCMResult<&'k Vec<FloatValue>> {
        memo_ref(&k.scales, || self.compute_scales(lookup))
    }

    fn compute_scales(&self, lookup: &Lookup) -> RSCMResult<Vec<FloatValue>> {
        let agents = &self.parameters.agents;
        let own = agents
            .iter()
            .map(|agent| self.agent_scale(agent, lookup))
            .collect::<RSCMResult<Vec<_>>>()?;
        let scale_of = |name: &str, fallback: FloatValue| {
            if name.is_empty() {
                return fallback;
            }
            let index = agents.iter().position(|a| a.variable == name).unwrap();
            own[index]
        };
        Ok(agents
            .iter()
            .zip(&own)
            .flat_map(|(agent, &scale)| {
                let from = &agent.sum_of_scale_from;
                (0..agent_inputs(agent).len())
                    .map(move |j| from.get(j).map_or(scale, |name| scale_of(name, scale)))
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    fn agent_scale(&self, agent: &ForcingAgent, lookup: &Lookup) -> RSCMResult<FloatValue> {
        let Some(h) = &agent.harmonisation else {
            return Ok(agent.scale);
        };
        let global = |t| -> RSCMResult<FloatValue> {
            let value = lookup(&agent.variable, agent.regional, t)?;
            Ok(self.unscaled_global(agent, &value))
        };
        let change = global(h.reference_year)? - global(h.preindustrial_reference_year)?;
        let scale = h.target / change;
        if !scale.is_finite() {
            return Err(RSCMError::Error(format!(
                "cannot harmonise {}: its ERF changes by {change} from {} to {}",
                agent.variable, h.preindustrial_reference_year, h.reference_year
            )));
        }
        Ok(scale)
    }

    fn unscaled_global(&self, agent: &ForcingAgent, value: &[FloatValue]) -> FloatValue {
        if agent.regional {
            self.fold(value)
        } else {
            value[0]
        }
    }

    /// Area-weighted global of four boxes, added box by box as MAGICC7 does.
    fn fold(&self, boxes: &[FloatValue]) -> FloatValue {
        let area = *self.parameters.grid().weights();
        (0..4).fold(0.0, |sum, i| sum + boxes[i] * area[i])
    }

    /// Scaled global and box ERF of each agent variable at time `t`.
    fn variable_forcing(
        &self,
        values: &[impl AsRef<[FloatValue]>],
        lookup: &Lookup,
        t: Time,
        k: &Constants,
    ) -> RSCMResult<Vec<Forcing>> {
        let scales = self.scales_with(lookup, k)?;
        let variables: Vec<_> = self
            .parameters
            .agents
            .iter()
            .flat_map(|agent| agent_inputs(agent).iter().map(move |name| (agent, name)))
            .collect();
        if values.len() != variables.len() {
            return Err(RSCMError::Error(format!(
                "expected {} agent variable values, got {}",
                variables.len(),
                values.len()
            )));
        }
        variables
            .into_iter()
            .zip(values)
            .zip(scales)
            .enumerate()
            .map(|(j, (((agent, name), value), &scale))| {
                self.held_forcing(agent, name, value.as_ref(), scale, lookup, t, k.slot(j))
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn held_forcing(
        &self,
        agent: &ForcingAgent,
        name: &str,
        value: &[FloatValue],
        scale: FloatValue,
        lookup: &Lookup,
        t: Time,
        (held, base): Slot,
    ) -> RSCMResult<Forcing> {
        let value = match agent.constant_after_year {
            Some(year) if t > year => memo_ref(held, || lookup(name, agent.regional, year))?,
            _ => value,
        };
        let Some(year) = agent.relative_to_year else {
            return Ok(self.scaled_forcing(agent, value, scale));
        };
        let base = memo_ref(base, || lookup(name, agent.regional, year))?;
        let boxes: [FloatValue; 4] = std::array::from_fn(|i| value[i] * scale - base[i] * scale);
        Ok((self.fold(&boxes), boxes))
    }

    /// Scaled box ERF at each of `times`, before efficacies, of each agent variable `lookup` has.
    ///
    /// Variables are in the order of `aggregate`'s values.
    pub fn variable_box_forcing(
        &self,
        lookup: &Lookup,
        has: &dyn Fn(&str) -> bool,
        times: &[Time],
    ) -> RSCMResult<Vec<(String, Vec<[FloatValue; 4]>)>> {
        self.variable_box_forcing_with(lookup, has, times, &self.constants())
    }

    fn variable_box_forcing_with(
        &self,
        lookup: &Lookup,
        has: &dyn Fn(&str) -> bool,
        times: &[Time],
        k: &Constants,
    ) -> RSCMResult<Vec<(String, Vec<[FloatValue; 4]>)>> {
        let scales = self.scales_with(lookup, k)?;
        let variables = self
            .parameters
            .agents
            .iter()
            .flat_map(|agent| agent_inputs(agent).iter().map(move |name| (agent, name)));
        variables
            .zip(scales)
            .enumerate()
            .filter(|(_, ((_, name), _))| has(name))
            .map(|(j, ((agent, name), &scale))| {
                let boxes = times
                    .iter()
                    .map(|&t| {
                        let value = lookup(name, agent.regional, t)?;
                        Ok(self
                            .held_forcing(agent, name, &value, scale, lookup, t, k.slot(j))?
                            .1)
                    })
                    .collect::<RSCMResult<Vec<_>>>()?;
                Ok((name.clone(), boxes))
            })
            .collect()
    }

    /// Global and box ERF of one of an agent's variables,
    /// from four box values for a regional agent and a global value otherwise.
    #[cfg(test)]
    fn agent_forcing(&self, agent: &ForcingAgent, value: &[FloatValue]) -> Forcing {
        self.scaled_forcing(agent, value, agent.scale)
    }

    /// `agent_forcing` with `scale` in place of the agent's own.
    fn scaled_forcing(
        &self,
        agent: &ForcingAgent,
        value: &[FloatValue],
        scale: FloatValue,
    ) -> Forcing {
        let area = *self.parameters.grid().weights();
        if agent.regional {
            let boxes: [FloatValue; 4] = std::array::from_fn(|i| value[i] * scale);
            (self.fold(&boxes), boxes)
        } else if let Some(pattern) = &agent.spreading_pattern {
            let scaled = value[0] * scale;
            let mean = self.fold(pattern);
            let boxes: [FloatValue; 4] = std::array::from_fn(|i| (scaled * pattern[i]) / mean);
            (self.fold(&boxes), boxes)
        } else {
            let pattern = &agent.regional_pattern;
            let qfrac = if agent.hemispheric {
                hemispheric_qfrac(pattern, &area)
            } else {
                compute_qfrac(pattern, &area)
            };
            let global = value[0] * scale;
            (global, qfrac.map(|q| global * q))
        }
    }

    /// Total forcing and global subtotals, both added in the order of `summation`.
    ///
    /// `forcing` holds each agent's `parts(agent)` variables in turn.
    fn aggregate_with(
        &self,
        forcing: &[Forcing],
        parts: &dyn Fn(&ForcingAgent) -> usize,
        internal: &[FloatValue],
    ) -> (Forcing, ForcingSubtotals) {
        let mut rest = forcing;
        let per_agent: Vec<(Forcing, ForcingSubtotals)> = self
            .parameters
            .agents
            .iter()
            .zip(internal)
            .map(|(agent, internal)| {
                let (parts, tail) = rest.split_at(parts(agent));
                rest = tail;
                let scale = match self.parameters.efficacy_method {
                    EfficacyMethod::None => 1.0,
                    EfficacyMethod::Unit => 1.0 / internal,
                    EfficacyMethod::AR6 => agent.efficacy / internal,
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

/// A variable's value in a step: its first `.1` entries.
struct StepValue([FloatValue; 4], usize);

impl AsRef<[FloatValue]> for StepValue {
    fn as_ref(&self) -> &[FloatValue] {
        &self.0[..self.1]
    }
}

/// A variable's `constant_after_year` and `relative_to_year` values.
type Slot<'k> = (&'k OnceLock<Vec<FloatValue>>, &'k OnceLock<Vec<FloatValue>>);

/// Values that are the same at every step of a run, each computed when first needed.
#[derive(Debug, Default)]
struct Constants {
    scales: OnceLock<Vec<FloatValue>>,
    patterns: OnceLock<Vec<[FloatValue; 4]>>,
    initial_internal: OnceLock<Vec<FloatValue>>,
    held: Vec<OnceLock<Vec<FloatValue>>>,
    base: Vec<OnceLock<Vec<FloatValue>>>,
}

impl Constants {
    fn slot(&self, variable: usize) -> Slot<'_> {
        (&self.held[variable], &self.base[variable])
    }
}

fn memo_ref<T>(cell: &OnceLock<T>, f: impl FnOnce() -> RSCMResult<T>) -> RSCMResult<&T> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    let value = f()?;
    Ok(cell.get_or_init(|| value))
}

/// Internal state of [`ForcingAggregation`]: the run constants of its prescribed inputs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ForcingAggregationState {
    #[serde(skip)]
    constants: Option<Constants>,
}

#[typetag::serde(name = "magicc.ForcingAggregationState")]
impl ComponentState for ForcingAggregationState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn sum(terms: impl Iterator<Item = Forcing>) -> Forcing {
    terms
        .reduce(|(global, boxes), (g, b)| (global + g, std::array::from_fn(|i| boxes[i] + b[i])))
        .unwrap_or((0.0, [0.0; 4]))
}

fn term_names<'a>(terms: &'a [ForcingTerm], out: &mut Vec<&'a str>) {
    for term in terms {
        match term {
            ForcingTerm::Agent(name) => out.push(name),
            ForcingTerm::Sum(nested) => term_names(nested, out),
        }
    }
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
fn validate_summation(parameters: &ForcingAggregationParameters) -> RSCMResult<()> {
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
        return Err(RSCMError::Error(format!(
            "summation must name each agent once: {listed:?} against agents {agents:?}"
        )));
    }
    Ok(())
}

/// An agent variable at any time: four values for a regional agent and one otherwise.
pub type Lookup<'a> = dyn Fn(&str, bool, Time) -> RSCMResult<Vec<FloatValue>> + 'a;

fn prescribed_only(name: &str, source: VariableSource) -> RSCMResult<()> {
    if source == VariableSource::Prescribed {
        return Ok(());
    }
    Err(RSCMError::Error(format!(
        "{name} must be prescribed, because its agent reads it at other times"
    )))
}

/// Check the scaling fields each agent sets refer to agents and variables it can read.
fn validate_scaling(parameters: &ForcingAggregationParameters) -> RSCMResult<()> {
    let agents = &parameters.agents;
    let variables: Vec<&String> = agents.iter().flat_map(agent_inputs).collect();
    for agent in agents {
        let fail = |reason: &str| Err(RSCMError::Error(format!("{}: {reason}", agent.variable)));
        let single = agent.sum_of.is_empty();
        if agent.harmonisation.is_some() && !single {
            return fail("harmonisation needs an agent without sum_of");
        }
        if agent.constant_after_year.is_some() && !single {
            return fail("constant_after_year needs an agent without sum_of");
        }
        if agent.relative_to_year.is_some() && !(single && agent.regional) {
            return fail("relative_to_year needs a regional agent without sum_of");
        }
        let mut names = Vec::new();
        term_names(&agent.pattern_terms, &mut names);
        if names.is_empty() != agent.pattern_year.is_none() {
            return fail("pattern_terms and pattern_year need each other");
        }
        if let Some(name) = names.iter().find(|n| !variables.iter().any(|v| v == *n)) {
            return fail(&format!("pattern_terms names no agent variable {name}"));
        }
        let from = &agent.sum_of_scale_from;
        if !from.is_empty() && from.len() != agent.sum_of.len() {
            return fail("sum_of_scale_from needs one entry per sum_of variable");
        }
        for name in from.iter().filter(|name| !name.is_empty()) {
            match agents.iter().find(|a| &a.variable == name) {
                None => return fail(&format!("sum_of_scale_from names no agent {name}")),
                Some(source) if !source.sum_of_scale_from.is_empty() => {
                    return fail(&format!("{name} takes its own scale from another agent"))
                }
                Some(_) => {}
            }
        }
    }
    Ok(())
}

/// Variables read for an agent.
fn agent_inputs(agent: &ForcingAgent) -> &[String] {
    if agent.sum_of.is_empty() {
        std::slice::from_ref(&agent.variable)
    } else {
        &agent.sum_of
    }
}

#[typetag::serde(name = "magicc.ForcingAggregation")]
impl Component for ForcingAggregation {
    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(ForcingAggregationState::default())
    }

    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions: Vec<_> = self
            .parameters
            .agents
            .iter()
            .flat_map(|agent| {
                agent_inputs(agent).iter().map(|name| {
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

    fn solve_into(
        &self,
        _t_current: Time,
        t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
        out: &mut OutputWriter,
    ) -> RSCMResult<()> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<ForcingAggregationState>()
            .ok_or_else(|| {
                RSCMError::Error("Internal state should be ForcingAggregationState".into())
            })?;
        let k = &*state.constants.get_or_insert_with(|| self.constants());
        let values: Vec<StepValue> = self
            .parameters
            .agents
            .iter()
            .flat_map(|agent| agent_inputs(agent).iter().map(move |name| (agent, name)))
            .map(|(agent, name)| {
                if agent.regional {
                    let window = input_state.get_four_box_window(name);
                    StepValue(REGIONS.map(|region| window.get(region)), 4)
                } else {
                    StepValue([input_state.get_scalar_window(name).get(); 4], 1)
                }
            })
            .collect();
        let lookup = |name: &str, regional: bool, t: Time| -> RSCMResult<Vec<FloatValue>> {
            if regional {
                let window = input_state.get_four_box_window(name);
                prescribed_only(name, window.variable_source())?;
                REGIONS
                    .iter()
                    .map(|&region| window.interpolate(t, region))
                    .collect()
            } else {
                let window = input_state.get_scalar_window(name);
                prescribed_only(name, window.variable_source())?;
                Ok(vec![window.interpolate(t)?])
            }
        };
        let forcing = self.variable_forcing(&values, &lookup, t_next, k)?;
        let drifting;
        let internal = match &self.parameters.climate {
            None => self.initial_internal_efficacies(&lookup, k)?,
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
                let patterns = self.patterns_with(&lookup, k)?;
                drifting = self
                    .internal_efficacies_with(ecs, q, patterns)
                    .ok_or_else(|| {
                        RSCMError::Error(format!("LAMCALC failed to converge for ECS={ecs}"))
                    })?;
                &drifting
            }
        };
        let ((total, adjusted), subtotals) =
            self.aggregate_with(&forcing, &|agent| agent_inputs(agent).len(), internal);
        out.set(ERF, total);
        subtotals.write_to(out);
        out.set(EFFICACY_ADJUSTED_ERF, FourBoxSlice::from_array(adjusted));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::{EfficacyClimate, ForcingAgent, Harmonisation};
    use rscm_core::forcing::ForcingCategory;
    use rscm_core::model::resolve_standalone;
    use rscm_core::state::StateValue;
    use rscm_core::state::VariableSource;
    use rscm_core::timeseries_collection::{TimeseriesCollection, TimeseriesData, TimeseriesItem};

    fn no_lookup(name: &str, _: bool, _: Time) -> RSCMResult<Vec<FloatValue>> {
        Err(RSCMError::Error(format!("no series for {name}")))
    }

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
            scale: 1.0,
            efficacy,
            internal_efficacy: internal,
            harmonisation: None,
            sum_of_scale_from: Vec::new(),
            spreading_pattern: None,
            constant_after_year: None,
            relative_to_year: None,
            pattern_terms: Vec::new(),
            pattern_year: None,
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
        let c = component(vec![agent(false, 1.0, 1.0)], EfficacyMethod::AR6);
        let (total, boxes) = c.aggregate(&[vec![2.0]], &no_lookup, 2000.0).unwrap();
        let area = *c.parameters().grid().weights();
        let mean: FloatValue = (0..4).map(|i| boxes[i] * area[i]).sum();
        assert!((total - 2.0).abs() < 1e-12);
        assert!((mean - 2.0).abs() < 1e-12);
        assert!((boxes[1] / boxes[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn scale_multiplies_the_input_before_the_efficacy() {
        let scaled = |regional, scale, value: Vec<FloatValue>| {
            let mut a = agent(regional, 1.5, 1.2);
            a.scale = scale;
            component(vec![a], EfficacyMethod::AR6)
                .aggregate(&[value], &no_lookup, 2000.0)
                .unwrap()
        };
        let boxes = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            scaled(true, 0.66, boxes.clone()),
            scaled(true, 1.0, boxes.iter().map(|v| v * 0.66).collect())
        );
        assert_eq!(
            scaled(false, 0.66, vec![2.0]),
            scaled(false, 1.0, vec![2.0 * 0.66])
        );
        assert_ne!(
            scaled(false, 0.66, vec![2.0]),
            scaled(false, 1.0, vec![2.0])
        );
    }

    #[test]
    fn hemispheric_agent_keeps_hemispheric_means() {
        let mut tropoz = agent(false, 1.0, 1.0);
        tropoz.regional_pattern = [0.46565, 0.51646, 0.17687, 0.23793];
        tropoz.hemispheric = true;
        let c = component(vec![tropoz], EfficacyMethod::None);
        let (total, boxes) = c.aggregate(&[vec![1.0]], &no_lookup, 2000.0).unwrap();
        let expected = [0.95618, 1.06051, 0.9324, 1.25429];
        for i in 0..4 {
            assert!((boxes[i] - expected[i]).abs() < 1e-5, "{boxes:?}");
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn efficacy_methods_scale_box_forcing() {
        let values = [vec![1.0, 2.0, 3.0, 4.0]];
        let adjusted = |method| {
            component(vec![agent(true, 1.5, 1.2)], method)
                .aggregate(&values, &no_lookup, 2000.0)
                .unwrap()
        };
        let (total, none) = adjusted(EfficacyMethod::None);
        assert_eq!(none, [1.0, 2.0, 3.0, 4.0]);
        assert!((adjusted(EfficacyMethod::Unit).1[3] - 4.0 / 1.2).abs() < 1e-12);
        assert!((adjusted(EfficacyMethod::AR6).1[3] - 5.0).abs() < 1e-12);
        assert_eq!(adjusted(EfficacyMethod::AR6).0, total);
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
        let udeb = crate::climate::ClimateUDEB::new();
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
        let fixed = component(vec![agent(false, 1.0, 1.3)], EfficacyMethod::AR6);
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
        let mut collection = TimeseriesCollection::new();
        for item in [
            scalar_item("x", [1.0, 1.0]),
            scalar_item(EQUILIBRIUM_CLIMATE_SENSITIVITY, ecs),
            scalar_item(ERF_2XCO2, [3.9, 4.2]),
        ] {
            let TimeseriesData::Scalar(ts) = item.data else {
                unreachable!()
            };
            collection.add_timeseries(item.name, ts);
        }
        // MAGICC7 lags the sensitivity into the aggregation
        let mut step =
            resolve_standalone("ForcingAggregation", &collection, &c.definitions()).unwrap();
        for read in &mut step.reads {
            if read.definition.name == EQUILIBRIUM_CLIMATE_SENSITIVITY {
                read.source = VariableSource::Lagged;
                read.plain = false;
            }
        }
        let input_state = InputState::from_step(&collection, &step, 2000.0);
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
            let forcing = [c.agent_forcing(&c.parameters().agents[0], &[1.0])];
            c.aggregate_with(&forcing, &|_| 1, &c.internal_efficacies(ecs, q).unwrap())
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
        assert_eq!(flat.aggregate(&values, &no_lookup, 2000.0).unwrap().0, 1.0);
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
        assert_eq!(
            nested.aggregate(&values, &no_lookup, 2000.0).unwrap().0,
            1.0 + 2e-16
        );
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
        let c = component(vec![biomass], EfficacyMethod::AR6);
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
            EfficacyMethod::AR6,
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

    /// Series of `x` in 2000 to 2004, rising by one each year from `start`.
    fn rising(start: FloatValue) -> impl Fn(&str, bool, Time) -> RSCMResult<Vec<FloatValue>> {
        move |_, regional, t| {
            let value = start + (t.clamp(2000.0, 2004.0) - 2000.0);
            Ok(vec![value; if regional { 4 } else { 1 }])
        }
    }

    #[test]
    fn harmonisation_scales_change_to_target() {
        let mut harmonised = agent(true, 1.0, 1.0);
        harmonised.scale = 100.0;
        harmonised.harmonisation = Some(Harmonisation {
            target: 0.5,
            reference_year: 2003.0,
            preindustrial_reference_year: 2000.0,
        });
        let c = component(vec![harmonised], EfficacyMethod::None);
        let scales = c.variable_scales(&rising(1.0)).unwrap();
        assert_eq!(scales, vec![0.5 / 3.0]);
        let (total, _) = c.aggregate(&[vec![4.0; 4]], &rising(1.0), 2003.0).unwrap();
        assert!((total - 4.0 * 0.5 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn sum_of_scale_from_takes_another_agents_scale() {
        let mut industrial = named("industrial", true);
        industrial.scale = 2.0;
        let mut biomass = named("biomass", true);
        biomass.sum_of = vec!["p".into(), "q".into()];
        biomass.sum_of_scale_from = vec!["industrial".into(), String::new()];
        biomass.scale = 5.0;
        let c = component(vec![industrial, biomass], EfficacyMethod::None);
        assert_eq!(c.variable_scales(&no_lookup).unwrap(), vec![2.0, 2.0, 5.0]);
        let values = [vec![0.0; 4], vec![1.0; 4], vec![1.0; 4]];
        assert_eq!(
            c.aggregate(&values, &no_lookup, 2000.0).unwrap().1,
            [7.0; 4]
        );
    }

    #[test]
    fn spreading_pattern_spreads_but_leaves_internal_efficacy() {
        let mut dust = agent(false, 1.0, 1.0);
        dust.spreading_pattern = Some([2.0, 1.0, 2.0, 1.0]);
        let plain = component(vec![dust.clone()], EfficacyMethod::None);
        let (_, boxes) = plain.aggregate(&[vec![1.0]], &no_lookup, 2000.0).unwrap();
        let area = *plain.parameters().grid().weights();
        let mean = (0..4).fold(0.0, |sum, i| sum + [2.0, 1.0, 2.0, 1.0][i] * area[i]);
        assert_eq!(boxes, [2.0, 1.0, 2.0, 1.0].map(|p| p / mean));
        let c = with_climate(vec![dust.clone()]);
        dust.spreading_pattern = None;
        let climate = c.parameters().climate.clone().unwrap();
        assert_eq!(
            c.internal_efficacies(climate.ecs, climate.erf_2xco2),
            with_climate(vec![dust]).internal_efficacies(climate.ecs, climate.erf_2xco2)
        );
    }

    #[test]
    fn constant_after_year_holds_value() {
        let mut held = agent(false, 1.0, 1.0);
        held.constant_after_year = Some(2001.0);
        let c = component(vec![held], EfficacyMethod::None);
        let at = |t| c.aggregate(&[vec![t - 1999.0]], &rising(1.0), t).unwrap().0;
        assert_eq!(at(2001.0), 2.0);
        assert_eq!(at(2003.0), 2.0);
    }

    #[test]
    fn relative_to_year_subtracts_scaled_value_in_that_year() {
        let mut snow = agent(true, 1.0, 1.0);
        snow.scale = 0.1;
        snow.relative_to_year = Some(2000.0);
        let c = component(vec![snow], EfficacyMethod::None);
        let (_, boxes) = c.aggregate(&[vec![4.0; 4]], &rising(1.0), 2003.0).unwrap();
        assert_eq!(boxes, [4.0 * 0.1 - 0.1; 4]);
    }

    #[test]
    fn pattern_terms_sum_scaled_boxes_in_pattern_year() {
        let mut fossil = named("fossil", true);
        fossil.scale = 2.0;
        let mut biomass = named("biomass", true);
        biomass.sum_of = vec!["b".into()];
        biomass.sum_of_scale_from = vec!["fossil".into()];
        let mut summed = named("summed", true);
        summed.pattern_terms = vec![ForcingTerm::Sum(vec![
            ForcingTerm::Agent("fossil".into()),
            ForcingTerm::Agent("b".into()),
        ])];
        summed.pattern_year = Some(2003.0);
        let c = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![fossil, biomass, summed.clone()],
            climate: Some(EfficacyClimate::default()),
            ..Default::default()
        })
        .unwrap();
        let patterns = c.patterns(&rising(1.0)).unwrap();
        assert_eq!(patterns[2], [16.0; 4]);
        let climate = EfficacyClimate::default();
        let from_terms = c
            .internal_efficacies_from(climate.ecs, climate.erf_2xco2, &rising(1.0))
            .unwrap()
            .unwrap();
        summed.pattern_terms = Vec::new();
        summed.pattern_year = None;
        summed.regional_pattern = [16.0; 4];
        let named_pattern = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![named("fossil", true), named("b", true), summed],
            climate: Some(climate.clone()),
            ..Default::default()
        })
        .unwrap()
        .internal_efficacies(climate.ecs, climate.erf_2xco2)
        .unwrap();
        assert_eq!(from_terms[2], named_pattern[2]);
    }

    #[test]
    fn pattern_terms_need_pattern_year() {
        let mut agent = named("a", true);
        agent.pattern_terms = vec![ForcingTerm::Agent("a".into())];
        let result = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![agent],
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn rejects_unknown_scale_source() {
        let mut biomass = named("biomass", true);
        biomass.sum_of = vec!["p".into()];
        biomass.sum_of_scale_from = vec!["missing".into()];
        let result = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![biomass],
            ..Default::default()
        });
        assert!(result.is_err());
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
