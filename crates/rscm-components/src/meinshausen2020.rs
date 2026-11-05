//! Meinshausen et al. (2020) forcing for CO2, CH4 and N2O
//!
//! MAGICC7 calls this method OLBL. FaIR v2 calls it `meinshausen2020`.
//! Each function multiplies by `scale`, the rapid adjustment or forcing scaling,
//! between the coefficient and the concentration term, as MAGICC7 does.
//! Coupled MAGICC7 runs amplify a one-ulp change in this order beyond the parity tolerance.
//!
//! $$\alpha_{CO2} = c_1 \sqrt{N} + \begin{cases}
//!     d_1 & C \le C_0 \\
//!     a_1 (C - C_0)^2 + b_1 (C - C_0) + d_1 & C_0 < C < C_{max} \\
//!     d_1 - b_1^2 / (4 a_1) & C \ge C_{max}
//! \end{cases}$$
//! $$F_{CO2} = \alpha_{CO2} \ln(C / C_0)$$
//! $$F_{CH4} = (a_3 \sqrt{M} + b_3 \sqrt{N} + d_3) (\sqrt{M} - \sqrt{M_0})$$
//! $$F_{N2O} = (a_2 \sqrt{C} + b_2 \sqrt{N} + c_2 \sqrt{M} + d_2) (\sqrt{N} - \sqrt{N_0})$$
//!
//! $C$ is CO2 in ppm, $M$ and $N$ are CH4 and N2O in ppb.

use rscm_core::timeseries::FloatValue;

/// Coefficients of the Meinshausen et al. (2020) formulae.
///
/// The default is the published set, which FaIR uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meinshausen2020 {
    /// CO2 quadratic term (W m-2 ppm-2).
    pub a1: FloatValue,
    /// CO2 linear term (W m-2 ppm-1).
    pub b1: FloatValue,
    /// N2O overlap in the CO2 coefficient (W m-2 ppb-0.5).
    pub c1: FloatValue,
    /// CO2 constant (W m-2).
    pub d1: FloatValue,
    /// CO2 overlap in the N2O coefficient (W m-2 ppm-0.5 ppb-0.5).
    pub a2: FloatValue,
    /// N2O term (W m-2 ppb-1).
    pub b2: FloatValue,
    /// CH4 overlap in the N2O coefficient (W m-2 ppb-1).
    pub c2: FloatValue,
    /// N2O constant (W m-2 ppb-0.5).
    pub d2: FloatValue,
    /// CH4 term (W m-2 ppb-1).
    pub a3: FloatValue,
    /// N2O overlap in the CH4 coefficient (W m-2 ppb-1).
    pub b3: FloatValue,
    /// CH4 constant (W m-2 ppb-0.5).
    pub d3: FloatValue,
}

impl Default for Meinshausen2020 {
    fn default() -> Self {
        Self {
            a1: -2.4785e-07,
            b1: 0.00075906,
            c1: -0.0021492,
            d1: 5.2488,
            a2: -0.00034197,
            b2: 0.00025455,
            c2: -0.00024357,
            d2: 0.12173,
            a3: -8.9603e-05,
            b3: -0.00012462,
            d3: 0.045194,
        }
    }
}

impl Meinshausen2020 {
    /// CO2 forcing per unit $\ln(C / C_0)$.
    pub fn co2_alpha(&self, co2: FloatValue, n2o: FloatValue, co2_ref: FloatValue) -> FloatValue {
        let n2o_overlap = self.c1 * n2o.sqrt();
        // Vertex of the quadratic, in MAGICC7's form
        let c_max = (2.0 * self.a1 * co2_ref - self.b1) / (2.0 * self.a1);

        if co2 >= c_max {
            -self.b1 * self.b1 / (4.0 * self.a1) + self.d1 + n2o_overlap
        } else if co2 <= co2_ref {
            self.d1 + n2o_overlap
        } else {
            let delta = co2 - co2_ref;
            self.b1 * delta + self.a1 * (delta * delta) + self.d1 + n2o_overlap
        }
    }

    /// CO2 forcing (W m-2).
    pub fn co2(
        &self,
        co2: FloatValue,
        n2o: FloatValue,
        co2_ref: FloatValue,
        scale: FloatValue,
    ) -> FloatValue {
        self.co2_alpha(co2, n2o, co2_ref) * scale * (co2 / co2_ref).ln()
    }

    /// CH4 forcing (W m-2).
    pub fn ch4(
        &self,
        ch4: FloatValue,
        n2o: FloatValue,
        ch4_ref: FloatValue,
        scale: FloatValue,
    ) -> FloatValue {
        let coeff = self.a3 * ch4.sqrt() + self.b3 * n2o.sqrt() + self.d3;
        coeff * scale * (ch4.sqrt() - ch4_ref.sqrt())
    }

    /// N2O forcing (W m-2).
    pub fn n2o(
        &self,
        co2: FloatValue,
        ch4: FloatValue,
        n2o: FloatValue,
        n2o_ref: FloatValue,
        scale: FloatValue,
    ) -> FloatValue {
        let coeff = self.a2 * co2.sqrt() + self.b2 * n2o.sqrt() + self.c2 * ch4.sqrt() + self.d2;
        coeff * scale * (n2o.sqrt() - n2o_ref.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn zero_at_reference() {
        let m = Meinshausen2020::default();
        assert_relative_eq!(m.co2(278.0, 270.0, 278.0, 1.0), 0.0);
        assert_relative_eq!(m.ch4(722.0, 270.0, 722.0, 1.0), 0.0);
        assert_relative_eq!(m.n2o(278.0, 722.0, 270.0, 270.0, 1.0), 0.0);
    }

    #[test]
    fn co2_alpha_is_continuous_at_the_vertex() {
        let m = Meinshausen2020::default();
        let c_max = 278.0 - m.b1 / (2.0 * m.a1);
        let below = m.co2_alpha(c_max * (1.0 - 1e-12), 270.0, 278.0);
        let above = m.co2_alpha(c_max * (1.0 + 1e-12), 270.0, 278.0);
        assert_relative_eq!(below, above, max_relative = 1e-12);
    }

    #[test]
    fn co2_alpha_is_constant_below_reference() {
        let m = Meinshausen2020::default();
        assert_relative_eq!(
            m.co2_alpha(200.0, 270.0, 278.0),
            m.d1 + m.c1 * 270.0_f64.sqrt()
        );
    }
}
