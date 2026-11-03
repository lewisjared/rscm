//! Nitrogen limitation of NPP (MAGICC7 module 13)
//!
//! Each year the plant-available nitrogen pool is the steady state of
//!
//! $$F_{dep} + F_{recyc} + F_{fix} = F_{up} + N / \tau$$
//!
//! found by bisection as MAGICC7 does.
//! Before `fertilization_yrstart` plant uptake is a fixed fraction of NPP and NPP
//! is not limited. From then on uptake and NPP scale with the nitrogen availability
//! $\phi(N, NPP) = (1 + A) / (A + e^{-C (N^2 / (N + S) - k \, NPP)})$
//! relative to its value in the last year before `fertilization_yrstart`.

use crate::parameters::NitrogenParameters;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Nitrogen state carried between years.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NitrogenState {
    /// Nitrogen pool in the last year before limitation (Gt N)
    pub pool_reference: FloatValue,
    /// Unlimited NPP in the last year before limitation (GtC/yr)
    pub npp_reference: FloatValue,
}

/// Nitrogen pool, fluxes and the resulting NPP limitation for one year.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NitrogenBudget {
    /// Factor applied to NPP
    pub limitation: FloatValue,
    /// Plant-available nitrogen pool (Gt N)
    pub pool: FloatValue,
    /// Plant uptake (Gt N/yr)
    pub uptake: FloatValue,
    /// Recycling from decomposition (Gt N/yr)
    pub recycling: FloatValue,
    /// Leaching and gaseous losses (Gt N/yr)
    pub losses: FloatValue,
    /// Biological fixation (Gt N/yr)
    pub fixation: FloatValue,
    /// Atmospheric deposition (Gt N/yr)
    pub deposition: FloatValue,
}

/// Drivers of the nitrogen budget for one year.
#[derive(Debug, Clone, Copy)]
pub struct NitrogenDrivers {
    /// NPP before nitrogen limitation (GtC/yr)
    pub npp: FloatValue,
    /// Heterotrophic respiration of the previous year (GtC/yr)
    pub respiration: FloatValue,
    /// CO2 fertilisation factor of NPP
    pub fertilization: FloatValue,
    /// Temperature factor of NPP
    pub temperature_factor: FloatValue,
    /// Deposition (Gt N/yr)
    pub deposition: FloatValue,
    /// Prescribed fixation (Gt N/yr), unused with prognostic fixation
    pub fixation: FloatValue,
}

impl NitrogenParameters {
    fn availability(&self, pool: FloatValue, npp: FloatValue) -> FloatValue {
        let a = self.availability_factor_a;
        let supply = pool * pool / (pool + self.uptake_supply);
        (1.0 + a) / (a + (-self.availability_factor_c * (supply - self.ncratio_uptake * npp)).exp())
    }

    fn fixation(&self, drivers: &NitrogenDrivers, ratio: FloatValue) -> FloatValue {
        match &self.prognostic_fixation {
            None => drivers.fixation,
            Some(f) => {
                let npp = drivers.npp * ratio;
                1e-3 * f.vmax * npp / (f.km + npp)
            }
        }
    }

    /// Solve the nitrogen pool for one year.
    ///
    /// `limited` is false before `fertilization_yrstart`, when the year also becomes
    /// the reference for limitation.
    pub fn solve(
        &self,
        state: &mut NitrogenState,
        drivers: &NitrogenDrivers,
        limited: bool,
    ) -> NitrogenBudget {
        let f = self.recycling_cn_factor;
        let recycling = self.ncratio_recycling * drivers.respiration * (1.0 + f)
            / (f + drivers.fertilization)
            * (1.0 + f)
            / (f + drivers.temperature_factor);
        let ratio = |pool: FloatValue| {
            if limited {
                self.availability(pool, drivers.npp)
                    / self.availability(state.pool_reference, state.npp_reference)
            } else {
                1.0
            }
        };
        let balance = |pool: FloatValue| {
            let r = ratio(pool);
            drivers.deposition + recycling + self.fixation(drivers, r)
                - self.ncratio_uptake * drivers.npp * r
                - pool / self.loss_turnover_time
        };
        let pool = bisect(balance);
        let limitation = ratio(pool);
        if !limited {
            state.pool_reference = pool;
            state.npp_reference = drivers.npp;
        }
        NitrogenBudget {
            limitation,
            pool,
            uptake: self.ncratio_uptake * drivers.npp * limitation,
            recycling,
            losses: pool / self.loss_turnover_time,
            fixation: self.fixation(drivers, limitation),
            deposition: drivers.deposition,
        }
    }
}

/// MAGICC7's bisection over [-100, 1000] Gt N to a 1e-5 Gt N bracket.
fn bisect(f: impl Fn(FloatValue) -> FloatValue) -> FloatValue {
    let (mut lower, mut upper) = (-100.0, 1000.0);
    let mut f_lower = f(lower);
    for _ in 0..200 {
        let mid = 0.5 * (lower + upper);
        let f_mid = f(mid);
        if f_lower * f_mid <= 0.0 {
            upper = mid;
        } else {
            lower = mid;
            f_lower = f_mid;
        }
        if (upper - lower).abs() <= 1e-5 {
            break;
        }
    }
    0.5 * (lower + upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drivers(npp: FloatValue) -> NitrogenDrivers {
        NitrogenDrivers {
            npp,
            respiration: npp,
            fertilization: 1.0,
            temperature_factor: 1.0,
            deposition: 0.0,
            fixation: 0.1694,
        }
    }

    #[test]
    fn test_steady_state_before_limitation() {
        let p = NitrogenParameters::default();
        let mut state = NitrogenState::default();
        let budget = p.solve(&mut state, &drivers(66.2716), false);
        // Uptake and recycling cancel, so losses balance fixation.
        assert!((budget.pool - 2.0 * 0.1694).abs() < 1e-5);
        assert_eq!(budget.limitation, 1.0);
        assert_eq!(state.pool_reference, budget.pool);
    }

    #[test]
    fn test_unchanged_drivers_do_not_limit() {
        let p = NitrogenParameters::default();
        let mut state = NitrogenState::default();
        p.solve(&mut state, &drivers(66.2716), false);
        let budget = p.solve(&mut state, &drivers(66.2716), true);
        assert!((budget.limitation - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_higher_npp_is_limited() {
        let p = NitrogenParameters::default();
        let mut state = NitrogenState::default();
        p.solve(&mut state, &drivers(66.2716), false);
        let mut d = drivers(80.0);
        d.respiration = 66.2716;
        let budget = p.solve(&mut state, &d, true);
        assert!(budget.limitation < 1.0);
        let residual =
            budget.deposition + budget.recycling + budget.fixation - budget.uptake - budget.losses;
        assert!(residual.abs() < 1e-4);
    }
}
