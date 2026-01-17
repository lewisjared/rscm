//! Two-layer energy balance model component implementation.
//!
//! This module contains the core [`TwoLayer`] component and its associated
//! [`TwoLayerParameters`] configuration. The component integrates with the
//! RSCM framework via the [`Component`] trait.
//!
//! See the [crate-level documentation](crate) for scientific background and usage examples.

use ode_solvers::*;
use std::sync::Arc;

use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    TimeseriesWindow,
};
use rscm_core::errors::RSCMResult;
use rscm_core::ivp::{get_last_step, IVPBuilder, IVP};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

// Define some types that are used by OdeSolvers
type ModelState = Vector3<FloatValue>;

/// Parameters for the two-layer energy balance model
///
/// This parameterisation follows Held et al. (2010) and represents the climate
/// system as two coupled thermal reservoirs: a surface layer (mixed layer ocean
/// + atmosphere) and a deep ocean layer.
///
/// # References
///
/// Held, I. M., Winton, M., Takahashi, K., Delworth, T., Zeng, F., & Vallis, G. K. (2010).
/// Probing the fast and slow components of global warming by returning abruptly to
/// preindustrial forcing. Journal of Climate, 23(9), 2418-2427.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwoLayerParameters {
    /// Climate feedback parameter at zero warming
    ///
    /// Units: W/(m² K)
    ///
    /// Controls the strength of radiative feedback. Higher values mean stronger
    /// negative feedback and lower climate sensitivity. Typical values: 0.8-1.5
    pub lambda0: FloatValue,

    /// Nonlinear feedback coefficient
    ///
    /// Units: W/(m² K²)
    ///
    /// Represents state-dependence of climate feedbacks. Positive values indicate
    /// that feedback weakens (sensitivity increases) as temperature rises.
    /// Set to 0 for a linear model. Typical values: 0-0.1
    pub a: FloatValue,

    /// Ocean heat uptake efficacy
    ///
    /// Units: dimensionless
    ///
    /// Ratio of the feedback parameter for ocean heat uptake to the equilibrium
    /// feedback parameter. Values > 1 indicate that ocean heat uptake is more
    /// effective at reducing surface warming than the equilibrium response suggests.
    /// Typical values: 1.0-1.8
    pub efficacy: FloatValue,

    /// Heat exchange coefficient between surface and deep layers
    ///
    /// Units: W/(m² K)
    ///
    /// Controls the rate of heat transfer from the surface to the deep ocean.
    /// Higher values mean faster equilibration between layers.
    /// Typical values: 0.5-1.0
    pub eta: FloatValue,

    /// Heat capacity of the surface layer (mixed layer ocean + atmosphere)
    ///
    /// Units: W yr/(m² K)
    ///
    /// Determines the thermal inertia of the fast-responding surface layer.
    /// Typical values: 5-15 (corresponding to ~50-150m mixed layer depth)
    pub heat_capacity_surface: FloatValue,

    /// Heat capacity of the deep ocean layer
    ///
    /// Units: W yr/(m² K)
    ///
    /// Determines the thermal inertia of the slow-responding deep ocean.
    /// Typical values: 50-200 (much larger than surface layer)
    pub heat_capacity_deep: FloatValue,
}

/// Two-layer energy balance climate model component
///
/// Implements a two-layer energy balance model following Held et al. (2010).
/// The model represents the climate system as two coupled thermal reservoirs:
///
/// - **Surface layer**: Fast-responding mixed layer ocean + atmosphere
/// - **Deep layer**: Slow-responding deep ocean
///
/// # Governing Equations
///
/// The model solves the following coupled ODEs:
///
/// $$
/// C_s \frac{dT_s}{dt} = F - \lambda(T_s) T_s - \varepsilon \eta (T_s - T_d)
/// $$
///
/// $$
/// C_d \frac{dT_d}{dt} = \eta (T_s - T_d)
/// $$
///
/// where:
///
/// | Symbol | Description | Units |
/// |--------|-------------|-------|
/// | $T_s$ | Surface temperature anomaly | K |
/// | $T_d$ | Deep ocean temperature anomaly | K |
/// | $F$ | Effective radiative forcing | W/m² |
/// | $\lambda(T_s) = \lambda_0 - a T_s$ | State-dependent feedback | W/(m² K) |
/// | $\varepsilon$ | Ocean heat uptake efficacy | dimensionless |
/// | $\eta$ | Heat exchange coefficient | W/(m² K) |
/// | $C_s$ | Surface layer heat capacity | W yr/(m² K) |
/// | $C_d$ | Deep ocean heat capacity | W yr/(m² K) |
///
/// # Example
///
/// ```rust
/// use rscm_two_layer::{TwoLayer, TwoLayerParameters};
///
/// let component = TwoLayer::from_parameters(TwoLayerParameters {
///     lambda0: 1.0,              // Climate feedback parameter
///     a: 0.0,                    // Linear model (no state-dependence)
///     efficacy: 1.0,             // Standard efficacy
///     eta: 0.7,                  // Heat exchange coefficient
///     heat_capacity_surface: 8.0, // ~80m mixed layer
///     heat_capacity_deep: 100.0,  // Deep ocean
/// });
/// ```
///
/// # References
///
/// Held, I. M., Winton, M., Takahashi, K., Delworth, T., Zeng, F., & Vallis, G. K. (2010).
/// Probing the fast and slow components of global warming by returning abruptly to
/// preindustrial forcing. Journal of Climate, 23(9), 2418-2427.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["temperature", "ocean", "two-layer", "stable"], category = "Temperature")]
#[parameters(TwoLayerParameters)]
#[inputs(
    erf { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[outputs(
    surface_temperature { name = "Surface Temperature", unit = "K" },
)]
pub struct TwoLayer {
    parameters: TwoLayerParameters,
}

// Create the set of ODEs to represent the two layer model
impl IVP<Time, ModelState> for TwoLayer {
    fn calculate_dy_dt(
        &self,
        _t: Time,
        input_state: &InputState,
        y: &ModelState,
        dy_dt: &mut ModelState,
    ) {
        let temperature_surface = y[0];
        let temperature_deep = y[1];
        let inputs = TwoLayerInputs::from_input_state(input_state);
        let erf = inputs.erf.current();

        let temperature_difference = temperature_surface - temperature_deep;

        let lambda_eff = self.parameters.lambda0 - self.parameters.a * temperature_surface;
        let heat_exchange_surface =
            self.parameters.efficacy * self.parameters.eta * temperature_difference;
        let dtemperature_surface_dt =
            (erf - lambda_eff * temperature_surface - heat_exchange_surface)
                / self.parameters.heat_capacity_surface;

        let heat_exchange_deep = self.parameters.eta * temperature_difference;
        let dtemperature_deep_dt = heat_exchange_deep / self.parameters.heat_capacity_deep;

        dy_dt[0] = dtemperature_surface_dt;
        dy_dt[1] = dtemperature_deep_dt;
        dy_dt[2] = self.parameters.heat_capacity_surface * dtemperature_surface_dt
            + self.parameters.heat_capacity_deep * dtemperature_deep_dt;
    }
}

impl TwoLayer {
    /// Creates a new two-layer model component from the given parameters.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Physical parameters controlling the model behaviour
    ///
    /// # Example
    ///
    /// ```rust
    /// use rscm_two_layer::{TwoLayer, TwoLayerParameters};
    ///
    /// let component = TwoLayer::from_parameters(TwoLayerParameters {
    ///     lambda0: 1.0,
    ///     a: 0.0,
    ///     efficacy: 1.2,
    ///     eta: 0.7,
    ///     heat_capacity_surface: 8.0,
    ///     heat_capacity_deep: 100.0,
    /// });
    /// ```
    pub fn from_parameters(parameters: TwoLayerParameters) -> Self {
        Self { parameters }
    }
}

#[typetag::serde]
impl Component for TwoLayer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let y0 = ModelState::new(0.0, 0.0, 0.0);

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), input_state, y0);

        let mut solver = solver.to_rk4(t_current, t_next, 0.1);
        solver.integrate().expect("Failed solving");

        let results = get_last_step(solver.results(), t_next);

        let outputs = TwoLayerOutputs {
            surface_temperature: results[0],
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;
    use rscm_core::model::extract_state;
    use rscm_core::state::StateValue;
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::{TimeseriesCollection, VariableType};

    fn create_component() -> TwoLayer {
        TwoLayer::from_parameters(TwoLayerParameters {
            lambda0: 1.0,               // W/(m^2 K) - climate feedback parameter
            a: 0.0,                     // No nonlinear feedback for simpler testing
            efficacy: 1.0,              // Ocean heat uptake efficacy
            eta: 0.7,                   // W/(m^2 K) - heat exchange coefficient
            heat_capacity_surface: 8.0, // W yr / (m^2 K) - realistic ocean mixed layer
            heat_capacity_deep: 100.0,  // W yr / (m^2 K) - deep ocean
        })
    }

    fn create_input_state_with_erf(
        erf_value: FloatValue,
        t_start: Time,
        t_end: Time,
    ) -> TimeseriesCollection {
        let mut ts_collection = TimeseriesCollection::new();
        ts_collection.add_timeseries(
            "Effective Radiative Forcing".to_string(),
            Timeseries::from_values(array![erf_value, erf_value], array![t_start, t_end]),
            VariableType::Exogenous,
        );
        ts_collection
    }

    #[test]
    fn test_positive_erf_causes_warming() {
        let component = create_component();
        let ts_collection = create_input_state_with_erf(4.0, 2000.0, 2001.0);
        let input_state = extract_state(&ts_collection, component.input_names(), 2000.0);

        let output_state = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let temperature = match output_state.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Positive ERF should cause warming (T > 0)
        assert!(
            temperature > 0.0,
            "Positive ERF should cause warming, got T = {}",
            temperature
        );

        // Temperature should be less than equilibrium value (ERF/lambda0 = 4.0/1.0 = 4.0 K)
        // since we're only integrating for 1 year
        assert!(
            temperature < 4.0,
            "Temperature {} should be below equilibrium (4.0 K)",
            temperature
        );
    }

    #[test]
    fn test_zero_erf_no_warming() {
        let component = create_component();
        let ts_collection = create_input_state_with_erf(0.0, 2000.0, 2001.0);
        let input_state = extract_state(&ts_collection, component.input_names(), 2000.0);

        let output_state = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let temperature = match output_state.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Zero ERF from zero initial state should stay at zero
        assert!(
            temperature.abs() < 1e-10,
            "Zero ERF should cause no warming, got T = {}",
            temperature
        );
    }

    #[test]
    fn test_negative_erf_causes_cooling() {
        let component = create_component();
        let ts_collection = create_input_state_with_erf(-2.0, 2000.0, 2001.0);
        let input_state = extract_state(&ts_collection, component.input_names(), 2000.0);

        let output_state = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let temperature = match output_state.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Negative ERF should cause cooling (T < 0)
        assert!(
            temperature < 0.0,
            "Negative ERF should cause cooling, got T = {}",
            temperature
        );
    }

    #[test]
    fn test_larger_erf_causes_more_warming() {
        let component = create_component();

        // Integrate with ERF = 2.0
        let ts_collection_small = create_input_state_with_erf(2.0, 2000.0, 2001.0);
        let input_state_small =
            extract_state(&ts_collection_small, component.input_names(), 2000.0);
        let output_small = component.solve(2000.0, 2001.0, &input_state_small).unwrap();
        let temp_small = match output_small.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Integrate with ERF = 4.0
        let ts_collection_large = create_input_state_with_erf(4.0, 2000.0, 2001.0);
        let input_state_large =
            extract_state(&ts_collection_large, component.input_names(), 2000.0);
        let output_large = component.solve(2000.0, 2001.0, &input_state_large).unwrap();
        let temp_large = match output_large.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Larger ERF should cause more warming
        assert!(
            temp_large > temp_small,
            "Larger ERF ({}) should cause more warming than smaller ERF ({})",
            temp_large,
            temp_small
        );

        // For linear system (a=0), doubling ERF should approximately double the response
        let ratio = temp_large / temp_small;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "Doubling ERF should approximately double temperature response, got ratio = {}",
            ratio
        );
    }
}
