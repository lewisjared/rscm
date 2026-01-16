use crate::component::InputState;
use crate::timeseries::Time;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim};
use ode_solvers::dop_shared::{FloatNumber, SolverResult};
use ode_solvers::*;
use std::sync::Arc;

const T_THRESHOLD: Time = 5e-3;

pub fn get_last_step<V>(results: &SolverResult<Time, V>, t_expected: Time) -> &V {
    let (t, y) = results.get();
    assert!(y.len() > 1);

    let t_distance = (t.last().unwrap() - t_expected).abs();

    // I couldn't figure out how to make this value a constant that worked with generics
    assert!(t_distance < T_THRESHOLD);

    let last_timestep = y.last().unwrap();

    last_timestep
}

pub trait IVP<T, S> {
    fn calculate_dy_dt(&self, t: T, input_state: &InputState, y: &S, dy_dt: &mut S);
}

/// Builds a solver for an initial value problem
#[derive(Clone)]
pub struct IVPBuilder<'a, C, S> {
    /// Model component to be solving
    // This needs to be a box/arc-like data type as the size of C is not known at compile time.
    component: Arc<C>,
    /// Initial
    y0: S,
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
    pub fn new(component: Arc<C>, input_state: &'a InputState<'a>, y0: OVector<T, D>) -> Self {
        Self {
            component,
            y0,
            input_state,
        }
    }

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
