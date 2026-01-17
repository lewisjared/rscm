//! Initial Value Problem (IVP) solving infrastructure for ODE-based components.
//!
//! Many climate model components are formulated as systems of Ordinary Differential
//! Equations (ODEs). This module provides the infrastructure to solve these ODEs
//! as initial value problems using numerical integration methods.
//!
//! # Overview
//!
//! The IVP module wraps the `ode_solvers` crate, providing:
//! - The [`IVP`] trait for components that define ODE systems
//! - The [`IVPBuilder`] struct for constructing and solving IVPs
//! - Integration with the component state system via [`InputState`]
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use nalgebra::Vector2;
//! use rscm_core::ivp::{IVP, IVPBuilder, get_last_step};
//! use rscm_core::component::InputState;
//!
//! // Define a component that implements IVP
//! struct TwoBoxModel {
//!     heat_capacity_1: f64,  // J / (m^2 K)
//!     heat_capacity_2: f64,
//!     heat_exchange: f64,    // W / (m^2 K)
//! }
//!
//! impl IVP<f64, Vector2<f64>> for TwoBoxModel {
//!     fn calculate_dy_dt(
//!         &self,
//!         t: f64,
//!         input_state: &InputState,
//!         y: &Vector2<f64>,
//!         dy_dt: &mut Vector2<f64>,
//!     ) {
//!         // Get forcing from input state
//!         let erf = input_state.get_scalar_window("Effective Radiative Forcing").at(t);
//!
//!         // Calculate temperature derivatives
//!         let (t1, t2) = (y[0], y[1]);
//!         dy_dt[0] = (erf - self.heat_exchange * (t1 - t2)) / self.heat_capacity_1;
//!         dy_dt[1] = self.heat_exchange * (t1 - t2) / self.heat_capacity_2;
//!     }
//! }
//!
//! // Solve the IVP
//! fn solve_component(
//!     component: Arc<TwoBoxModel>,
//!     input_state: &InputState,
//!     y0: Vector2<f64>,
//!     t_start: f64,
//!     t_end: f64,
//! ) -> Vector2<f64> {
//!     let builder = IVPBuilder::new(component, input_state, y0);
//!     let mut solver = builder.to_rk4(t_start, t_end, 0.1);
//!     let result = solver.integrate();
//!     get_last_step(&result, t_end).clone()
//! }
//! ```

use crate::component::InputState;
use crate::timeseries::Time;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim};
use ode_solvers::dop_shared::{FloatNumber, SolverResult};
use ode_solvers::*;
use std::sync::Arc;

/// Threshold for validating that the solver reached the expected end time.
/// If the final time differs from the expected time by more than this value,
/// the solver result is considered invalid.
const T_THRESHOLD: Time = 5e-3;

/// Extracts the final state vector from an ODE solver result.
///
/// This function validates that the solver reached the expected end time
/// (within a small threshold) and returns the state vector at the final timestep.
///
/// # Arguments
///
/// * `results` - The solver result containing time and state history
/// * `t_expected` - The expected end time of the integration
///
/// # Panics
///
/// Panics if:
/// - The result contains fewer than 2 timesteps
/// - The final time differs from `t_expected` by more than the threshold (5e-3)
pub fn get_last_step<V>(results: &SolverResult<Time, V>, t_expected: Time) -> &V {
    let (t, y) = results.get();
    assert!(y.len() > 1);

    let t_distance = (t.last().unwrap() - t_expected).abs();

    // I couldn't figure out how to make this value a constant that worked with generics
    assert!(t_distance < T_THRESHOLD);

    let last_timestep = y.last().unwrap();

    last_timestep
}

/// Trait for components that define an ODE system.
///
/// Implement this trait to define the differential equations for a component
/// that will be solved numerically. The trait is generic over the time type `T`
/// and the state vector type `S`.
///
/// # Type Parameters
///
/// * `T` - The time type (typically `f64`)
/// * `S` - The state vector type (e.g., `nalgebra::Vector2<f64>` for a 2-variable system)
///
/// # Example
///
/// ```ignore
/// use nalgebra::Vector1;
/// use rscm_core::ivp::IVP;
/// use rscm_core::component::InputState;
///
/// struct SimpleDecay {
///     decay_rate: f64,  // per year
/// }
///
/// impl IVP<f64, Vector1<f64>> for SimpleDecay {
///     fn calculate_dy_dt(
///         &self,
///         _t: f64,
///         _input_state: &InputState,
///         y: &Vector1<f64>,
///         dy_dt: &mut Vector1<f64>,
///     ) {
///         // Exponential decay: dy/dt = -k * y
///         dy_dt[0] = -self.decay_rate * y[0];
///     }
/// }
/// ```
pub trait IVP<T, S> {
    /// Calculate the time derivative of the state vector.
    ///
    /// This method defines the ODE system: dy/dt = f(t, y).
    ///
    /// # Arguments
    ///
    /// * `t` - Current time
    /// * `input_state` - Access to input variables (e.g., forcing timeseries)
    /// * `y` - Current state vector
    /// * `dy_dt` - Output: time derivative of the state vector (mutated in place)
    fn calculate_dy_dt(&self, t: T, input_state: &InputState, y: &S, dy_dt: &mut S);
}

/// Builder for constructing and solving initial value problems.
///
/// `IVPBuilder` wraps a component that implements [`IVP`] and provides methods
/// to construct ODE solvers. It connects the component's differential equations
/// to the numerical integration routines from the `ode_solvers` crate.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of the input state reference
/// * `C` - The component type (must implement [`IVP`])
/// * `S` - The state vector type
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use nalgebra::Vector2;
/// use rscm_core::ivp::{IVPBuilder, get_last_step};
///
/// // Create the builder
/// let builder = IVPBuilder::new(
///     Arc::new(my_component),
///     &input_state,
///     Vector2::new(initial_temp_1, initial_temp_2),
/// );
///
/// // Create an RK4 solver and integrate
/// let mut solver = builder.to_rk4(t_start, t_end, step_size);
/// let result = solver.integrate();
/// let final_state = get_last_step(&result, t_end);
/// ```
#[derive(Clone)]
pub struct IVPBuilder<'a, C, S> {
    /// The model component containing the ODE system definition
    // This needs to be a box/arc-like data type as the size of C is not known at compile time.
    component: Arc<C>,
    /// Initial state vector (y0)
    y0: S,
    /// Reference to the input state providing access to forcing data
    input_state: &'a InputState<'a>,
}

impl<T, D: Dim, C> System<T, OVector<T, D>> for IVPBuilder<'_, C, OVector<T, D>>
where
    T: FloatNumber,
    C: IVP<T, OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<T, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<T, D>,
{
    fn system(&self, t: T, y: &OVector<T, D>, dy: &mut OVector<T, D>) {
        self.component.calculate_dy_dt(t, self.input_state, y, dy)
    }
}

impl<'a, T, D: Dim, C> IVPBuilder<'a, C, OVector<T, D>>
where
    T: FloatNumber,
    C: IVP<T, OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<T, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new IVP builder.
    ///
    /// # Arguments
    ///
    /// * `component` - The component implementing the ODE system (wrapped in `Arc`)
    /// * `input_state` - Reference to the input state for accessing forcing data
    /// * `y0` - Initial state vector at time t0
    pub fn new(component: Arc<C>, input_state: &'a InputState<'a>, y0: OVector<T, D>) -> Self {
        Self {
            component,
            y0,
            input_state,
        }
    }

    /// Create a fourth-order Runge-Kutta (RK4) solver.
    ///
    /// RK4 is a fixed-step explicit method with good accuracy for most climate
    /// model applications. It provides a balance between computational cost
    /// and accuracy.
    ///
    /// # Arguments
    ///
    /// * `t0` - Start time
    /// * `t1` - End time
    /// * `step` - Fixed time step size (typically ~0.1 for annual models)
    ///
    /// # Returns
    ///
    /// An `Rk4` solver ready to be integrated via `.integrate()`.
    #[allow(clippy::type_complexity)]
    pub fn to_rk4(
        self,
        t0: T,
        t1: T,
        step: T,
    ) -> Rk4<T, OVector<T, D>, IVPBuilder<'a, C, OVector<T, D>>> {
        let y0 = self.y0.clone();
        Rk4::new(self, t0, y0, t1, step)
    }
}
