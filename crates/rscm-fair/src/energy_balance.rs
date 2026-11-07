//! n-layer energy balance model (FaIR step 16 to 18)
//!
//! The state $x = (F_s, T_1, \ldots, T_n)$ holds the forcing seen by the climate and the
//! temperature of each layer, surface first. Over one step of length $\Delta t$,
//!
//! $$x' = e^{A} x + A^{-1} (e^{A} - I)\, b\, F$$
//!
//! where $A$ is the energy balance matrix of Cummins et al. (2020) with heat capacities
//! divided by $\Delta t$, $b = (\gamma, 0, \ldots, 0)$ and $F$ the efficacy-adjusted forcing.
//! The top of atmosphere imbalance is
//!
//! $$N = F_{tot} - \kappa_1 T_1 + (1 - \epsilon) \kappa_n (T_{n-1} - T_n)$$
//!
//! from the unadjusted forcing, and the ocean heat content change integrates $N$ over the Earth.
//! Stochastic internal variability is not implemented.

use crate::constants::{
    efficacy_adjusted_forcing, forcing_unit, get, surface_temperature, temperature_unit,
    total_forcing, EARTH_RADIUS, SECONDS_PER_YEAR,
};
use nalgebra::{DMatrix, DVector};
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::names::{
    ocean_layer_temperature, NET_ENERGY_IMBALANCE, OCEAN_HEAT_CONTENT, STOCHASTIC_FORCING,
};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Parameters of [`EnergyBalance`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnergyBalanceParameters {
    /// Heat capacity of each layer, surface first (W yr m-2 K-1).
    pub ocean_heat_capacity: Vec<FloatValue>,
    /// Heat transfer coefficient of each layer, surface first (W m-2 K-1).
    /// The first is the climate feedback parameter.
    pub ocean_heat_transfer: Vec<FloatValue>,
    /// Efficacy of heat transfer into the deepest layer.
    pub deep_ocean_efficacy: FloatValue,
    /// Autocorrelation of the forcing seen by the climate (yr-1).
    #[serde(default = "default_gamma")]
    pub gamma_autocorrelation: FloatValue,
    /// Surface temperature state.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
    /// Total forcing input, for the energy imbalance.
    #[serde(default = "total_forcing")]
    pub forcing: String,
    /// Efficacy-adjusted forcing input, which drives the temperature.
    #[serde(default = "efficacy_adjusted_forcing")]
    pub efficacy_adjusted_forcing: String,
}

fn default_gamma() -> FloatValue {
    2.0
}

/// Discretised matrices for one step length.
#[derive(Debug, Clone)]
struct Discretised {
    dt: FloatValue,
    eb_matrix_d: DMatrix<FloatValue>,
    forcing_vector_d: DVector<FloatValue>,
}

/// FaIR's n-layer energy balance model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Unchecked")]
pub struct EnergyBalance {
    parameters: EnergyBalanceParameters,
    #[serde(skip)]
    discretised: OnceLock<Discretised>,
}

#[derive(Deserialize)]
struct Unchecked {
    parameters: EnergyBalanceParameters,
}

impl TryFrom<Unchecked> for EnergyBalance {
    type Error = RSCMError;

    fn try_from(unchecked: Unchecked) -> RSCMResult<Self> {
        Self::from_parameters(unchecked.parameters)
    }
}

impl EnergyBalance {
    /// Create the component, checking the layer parameters.
    pub fn from_parameters(parameters: EnergyBalanceParameters) -> RSCMResult<Self> {
        let n = parameters.ocean_heat_capacity.len();
        if n < 2 {
            return Err(RSCMError::Error(
                "EnergyBalance needs at least two layers".into(),
            ));
        }
        if parameters.ocean_heat_transfer.len() != n {
            return Err(RSCMError::Error(
                "ocean_heat_capacity and ocean_heat_transfer must have the same length".into(),
            ));
        }
        let component = Self {
            parameters,
            discretised: OnceLock::new(),
        };
        // The step scales whole rows, so it cannot make the matrix singular.
        if !component.eb_matrix(1.0).lu().is_invertible() {
            return Err(RSCMError::Error(
                "The energy balance matrix is singular: check the heat transfers and gamma".into(),
            ));
        }
        Ok(component)
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &EnergyBalanceParameters {
        &self.parameters
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.parameters.ocean_heat_capacity.len()
    }

    /// Temperature state of layer `layer`, counting the surface as layer 0.
    pub fn layer_temperature(&self, layer: usize) -> String {
        if layer == 0 {
            self.parameters.surface_temperature.clone()
        } else {
            ocean_layer_temperature(layer + 1)
        }
    }

    /// Energy balance matrix with the forcing row and column, for step `dt`.
    pub fn eb_matrix(&self, dt: FloatValue) -> DMatrix<FloatValue> {
        let p = &self.parameters;
        let n = self.n_layers();
        let c: Vec<FloatValue> = p.ocean_heat_capacity.iter().map(|c| c / dt).collect();
        let k = &p.ocean_heat_transfer;
        let mut epsilon = vec![1.0; n];
        epsilon[n - 2] = p.deep_ocean_efficacy;

        let mut m = DMatrix::zeros(n + 1, n + 1);
        m[(1, 1)] = -(k[0] + epsilon[0] * k[1]) / c[0];
        m[(1, 2)] = epsilon[0] * k[1] / c[0];
        m[(n, n - 1)] = k[n - 1] / c[n - 1];
        m[(n, n)] = -k[n - 1] / c[n - 1];
        for row in 1..n - 1 {
            m[(row + 1, row)] = k[row] / c[row];
            m[(row + 1, row + 1)] = -(k[row] + epsilon[row] * k[row + 1]) / c[row];
            m[(row + 1, row + 2)] = epsilon[row] * k[row + 1] / c[row];
        }
        m[(0, 0)] = -p.gamma_autocorrelation;
        m[(1, 0)] = 1.0 / c[0];
        m
    }

    fn discretise(&self, dt: FloatValue) -> Discretised {
        let n = self.n_layers();
        let eb_matrix = self.eb_matrix(dt);
        let eb_matrix_d = eb_matrix.exp();
        let mut forcing_vector = DVector::zeros(n + 1);
        forcing_vector[0] = self.parameters.gamma_autocorrelation;
        let rhs = (&eb_matrix_d - DMatrix::identity(n + 1, n + 1)) * forcing_vector;
        let forcing_vector_d = eb_matrix
            .lu()
            .solve(&rhs)
            .expect("from_parameters rejects a singular energy balance matrix");
        Discretised {
            dt,
            eb_matrix_d,
            forcing_vector_d,
        }
    }

    /// Advance the state one step under efficacy-adjusted forcing `forcing`.
    pub fn step(
        &self,
        state: &DVector<FloatValue>,
        forcing: FloatValue,
        dt: FloatValue,
    ) -> DVector<FloatValue> {
        let cached = self.discretised.get_or_init(|| self.discretise(dt));
        let fresh;
        let d = if cached.dt == dt {
            cached
        } else {
            fresh = self.discretise(dt);
            &fresh
        };
        &d.eb_matrix_d * state + &d.forcing_vector_d * forcing
    }

    /// Top of atmosphere energy imbalance for layer temperatures `t` and total forcing.
    pub fn energy_imbalance(&self, forcing: FloatValue, t: &[FloatValue]) -> FloatValue {
        let p = &self.parameters;
        let n = t.len();
        forcing - p.ocean_heat_transfer[0] * t[0]
            + (1.0 - p.deep_ocean_efficacy) * p.ocean_heat_transfer[n - 1] * (t[n - 2] - t[n - 1])
    }
}

#[typetag::serde(name = "fair.EnergyBalance")]
impl Component for EnergyBalance {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let p = &self.parameters;
        let mut definitions = vec![
            VariableDefinition::scalar_input(&p.forcing, &forcing_unit()),
            VariableDefinition::scalar_input(&p.efficacy_adjusted_forcing, &forcing_unit()),
            VariableDefinition::scalar_state(STOCHASTIC_FORCING, &forcing_unit()),
        ];
        for layer in 0..self.n_layers() {
            definitions.push(VariableDefinition::scalar_state(
                &self.layer_temperature(layer),
                &temperature_unit(),
            ));
        }
        definitions.push(VariableDefinition::scalar_state(OCEAN_HEAT_CONTENT, "J"));
        definitions.push(VariableDefinition::scalar_output(
            NET_ENERGY_IMBALANCE,
            &forcing_unit(),
        ));
        definitions
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let dt = t_next - t_current;
        let n = self.n_layers();
        let p = &self.parameters;
        let get = |name: &str| get(input_state, name);

        let mut state = DVector::zeros(n + 1);
        state[0] = get(STOCHASTIC_FORCING);
        for layer in 0..n {
            state[layer + 1] = get(&self.layer_temperature(layer));
        }
        let state = self.step(&state, get(&p.efficacy_adjusted_forcing), dt);
        let temperatures: Vec<FloatValue> = state.iter().skip(1).copied().collect();
        let imbalance = self.energy_imbalance(get(&p.forcing), &temperatures);
        let heat = get(OCEAN_HEAT_CONTENT)
            + imbalance
                * dt
                * EARTH_RADIUS
                * EARTH_RADIUS
                * 4.0
                * std::f64::consts::PI
                * SECONDS_PER_YEAR;

        let mut output = OutputState::with_capacity(n + 3);
        output.insert(STOCHASTIC_FORCING.to_string(), StateValue::Scalar(state[0]));
        for (layer, t) in temperatures.iter().enumerate() {
            output.insert(self.layer_temperature(layer), StateValue::Scalar(*t));
        }
        output.insert(
            NET_ENERGY_IMBALANCE.to_string(),
            StateValue::Scalar(imbalance),
        );
        output.insert(OCEAN_HEAT_CONTENT.to_string(), StateValue::Scalar(heat));
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn component() -> EnergyBalance {
        EnergyBalance::from_parameters(EnergyBalanceParameters {
            ocean_heat_capacity: vec![8.0, 14.0, 100.0],
            ocean_heat_transfer: vec![1.1, 1.6, 0.9],
            deep_ocean_efficacy: 1.1,
            gamma_autocorrelation: 2.0,
            surface_temperature: surface_temperature(),
            forcing: total_forcing(),
            efficacy_adjusted_forcing: efficacy_adjusted_forcing(),
        })
        .unwrap()
    }

    #[test]
    fn rejects_one_layer() {
        let mut p = component().parameters.clone();
        p.ocean_heat_capacity = vec![8.0];
        p.ocean_heat_transfer = vec![1.1];
        assert!(EnergyBalance::from_parameters(p).is_err());
    }

    #[test]
    fn rejects_a_singular_matrix() {
        let mut p = component().parameters.clone();
        p.ocean_heat_transfer[2] = 0.0;
        assert!(EnergyBalance::from_parameters(p.clone()).is_err());
        p.ocean_heat_transfer[2] = 0.9;
        p.gamma_autocorrelation = 0.0;
        assert!(EnergyBalance::from_parameters(p).is_err());
    }

    #[test]
    fn deserialising_checks_the_parameters() {
        let mut p = component().parameters.clone();
        p.ocean_heat_transfer[2] = 0.0;
        let json = format!(
            r#"{{"parameters": {}}}"#,
            serde_json::to_string(&p).unwrap()
        );
        assert!(serde_json::from_str::<EnergyBalance>(&json).is_err());
        let valid = serde_json::to_string(&component()).unwrap();
        assert!(serde_json::from_str::<EnergyBalance>(&valid).is_ok());
    }

    #[test]
    fn equilibrium_is_forcing_over_feedback() {
        let c = component();
        let mut state = DVector::zeros(4);
        for _ in 0..20_000 {
            state = c.step(&state, 4.0, 1.0);
        }
        assert_relative_eq!(state[0], 4.0, max_relative = 1e-10);
        // With efficacy the surface equilibrates at F / lambda only once the deep ocean has
        // warmed to match, which this run reaches.
        for layer in 1..4 {
            assert_relative_eq!(state[layer], 4.0 / 1.1, max_relative = 1e-8);
        }
        let t: Vec<_> = state.iter().skip(1).copied().collect();
        assert!(c.energy_imbalance(4.0, &t).abs() < 1e-8);
    }

    #[test]
    fn step_length_changes_the_discretisation() {
        let c = component();
        let one = c.step(&DVector::zeros(4), 1.0, 1.0);
        let five = c.step(&DVector::zeros(4), 1.0, 5.0);
        assert!(five[1] > one[1]);
    }
}
