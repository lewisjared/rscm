//! Physical dimensions for unit validation.
//!
//! This module provides a dimension system based on the SI base quantities.
//! Dimensions are represented as integer exponents of base dimensions,
//! enabling compile-time and runtime dimensional analysis.
//!
//! # Base Dimensions
//!
//! The system tracks seven base dimensions, following SI conventions:
//! - Mass (M)
//! - Length (L)
//! - Time (T)
//! - Temperature (Θ)
//! - Amount of substance (N)
//! - Electric current (I)
//! - Luminous intensity (J)
//!
//! Derived dimensions (like force, energy, power) are represented as
//! combinations of these base dimensions.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Represents the physical dimension of a quantity.
///
/// Dimensions are stored as integer exponents of the base SI dimensions.
/// For example:
/// - Velocity has dimensions L·T⁻¹ (length = 1, time = -1)
/// - Force has dimensions M·L·T⁻² (mass = 1, length = 1, time = -2)
/// - Power has dimensions M·L²·T⁻³
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Dimension {
    /// Mass exponent (M)
    pub mass: i8,
    /// Length exponent (L)
    pub length: i8,
    /// Time exponent (T)
    pub time: i8,
    /// Temperature exponent (Θ)
    pub temperature: i8,
    /// Amount of substance exponent (N)
    pub amount: i8,
    /// Electric current exponent (I)
    pub current: i8,
    /// Luminous intensity exponent (J)
    pub luminosity: i8,
}

impl Dimension {
    /// Creates a new dimension with all exponents set to zero (dimensionless).
    #[must_use]
    pub const fn dimensionless() -> Self {
        Self {
            mass: 0,
            length: 0,
            time: 0,
            temperature: 0,
            amount: 0,
            current: 0,
            luminosity: 0,
        }
    }

    /// Creates a dimension with the specified exponents.
    #[must_use]
    pub const fn new(
        mass: i8,
        length: i8,
        time: i8,
        temperature: i8,
        amount: i8,
        current: i8,
        luminosity: i8,
    ) -> Self {
        Self {
            mass,
            length,
            time,
            temperature,
            amount,
            current,
            luminosity,
        }
    }

    /// Mass dimension (M¹).
    pub const MASS: Self = Self::new(1, 0, 0, 0, 0, 0, 0);

    /// Length dimension (L¹).
    pub const LENGTH: Self = Self::new(0, 1, 0, 0, 0, 0, 0);

    /// Time dimension (T¹).
    pub const TIME: Self = Self::new(0, 0, 1, 0, 0, 0, 0);

    /// Temperature dimension (Θ¹).
    pub const TEMPERATURE: Self = Self::new(0, 0, 0, 1, 0, 0, 0);

    /// Amount of substance dimension (N¹).
    pub const AMOUNT: Self = Self::new(0, 0, 0, 0, 1, 0, 0);

    /// Electric current dimension (I¹).
    pub const CURRENT: Self = Self::new(0, 0, 0, 0, 0, 1, 0);

    /// Luminous intensity dimension (J¹).
    pub const LUMINOSITY: Self = Self::new(0, 0, 0, 0, 0, 0, 1);

    // Derived dimensions commonly used in climate science

    /// Area dimension (L²).
    pub const AREA: Self = Self::new(0, 2, 0, 0, 0, 0, 0);

    /// Volume dimension (L³).
    pub const VOLUME: Self = Self::new(0, 3, 0, 0, 0, 0, 0);

    /// Force dimension (M·L·T⁻²).
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);

    /// Energy dimension (M·L²·T⁻²).
    pub const ENERGY: Self = Self::new(1, 2, -2, 0, 0, 0, 0);

    /// Power dimension (M·L²·T⁻³).
    pub const POWER: Self = Self::new(1, 2, -3, 0, 0, 0, 0);

    /// Radiative flux dimension (M·T⁻³), equivalent to W/m².
    pub const RADIATIVE_FLUX: Self = Self::new(1, 0, -3, 0, 0, 0, 0);

    /// Returns true if this dimension is dimensionless.
    #[must_use]
    pub const fn is_dimensionless(&self) -> bool {
        self.mass == 0
            && self.length == 0
            && self.time == 0
            && self.temperature == 0
            && self.amount == 0
            && self.current == 0
            && self.luminosity == 0
    }

    /// Returns true if this dimension is compatible with another for conversion.
    ///
    /// Two dimensions are compatible if they are identical.
    #[must_use]
    pub const fn is_compatible(&self, other: &Self) -> bool {
        self.mass == other.mass
            && self.length == other.length
            && self.time == other.time
            && self.temperature == other.temperature
            && self.amount == other.amount
            && self.current == other.current
            && self.luminosity == other.luminosity
    }

    /// Raises this dimension to an integer power.
    #[must_use]
    pub const fn pow(&self, exp: i8) -> Self {
        Self {
            mass: self.mass * exp,
            length: self.length * exp,
            time: self.time * exp,
            temperature: self.temperature * exp,
            amount: self.amount * exp,
            current: self.current * exp,
            luminosity: self.luminosity * exp,
        }
    }
}

impl Mul for Dimension {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            mass: self.mass + rhs.mass,
            length: self.length + rhs.length,
            time: self.time + rhs.time,
            temperature: self.temperature + rhs.temperature,
            amount: self.amount + rhs.amount,
            current: self.current + rhs.current,
            luminosity: self.luminosity + rhs.luminosity,
        }
    }
}

impl Add for Dimension {
    type Output = Self;

    /// Adding dimensions is the same as multiplying them (adding exponents).
    /// This is intentional: when combining physical quantities, their dimensions multiply.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl Sub for Dimension {
    type Output = Self;

    /// Subtracting dimensions represents division (subtracting exponents).
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            mass: self.mass - rhs.mass,
            length: self.length - rhs.length,
            time: self.time - rhs.time,
            temperature: self.temperature - rhs.temperature,
            amount: self.amount - rhs.amount,
            current: self.current - rhs.current,
            luminosity: self.luminosity - rhs.luminosity,
        }
    }
}

impl Neg for Dimension {
    type Output = Self;

    /// Negating a dimension represents taking its reciprocal.
    fn neg(self) -> Self::Output {
        Self {
            mass: -self.mass,
            length: -self.length,
            time: -self.time,
            temperature: -self.temperature,
            amount: -self.amount,
            current: -self.current,
            luminosity: -self.luminosity,
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "dimensionless");
        }

        let mut parts = Vec::new();
        let symbols = [
            (self.mass, "M"),
            (self.length, "L"),
            (self.time, "T"),
            (self.temperature, "Θ"),
            (self.amount, "N"),
            (self.current, "I"),
            (self.luminosity, "J"),
        ];

        for (exp, sym) in symbols {
            if exp == 1 {
                parts.push(sym.to_string());
            } else if exp != 0 {
                parts.push(format!("{sym}^{exp}"));
            }
        }

        write!(f, "{}", parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let dim = Dimension::dimensionless();
        assert!(dim.is_dimensionless());
        assert_eq!(format!("{}", dim), "dimensionless");
    }

    #[test]
    fn test_base_dimensions() {
        assert_eq!(Dimension::MASS.mass, 1);
        assert_eq!(Dimension::LENGTH.length, 1);
        assert_eq!(Dimension::TIME.time, 1);
        assert_eq!(Dimension::TEMPERATURE.temperature, 1);
    }

    #[test]
    fn test_dimension_multiplication() {
        // Force = Mass * Acceleration = M * L * T^-2
        let force = Dimension::MASS * Dimension::LENGTH * Dimension::TIME.pow(-2);
        assert_eq!(force, Dimension::FORCE);
    }

    #[test]
    fn test_dimension_division() {
        // Velocity = Length / Time = L * T^-1
        let velocity = Dimension::LENGTH - Dimension::TIME;
        assert_eq!(velocity.length, 1);
        assert_eq!(velocity.time, -1);
    }

    #[test]
    fn test_power_derived_dimension() {
        // Power = Energy / Time = M L^2 T^-2 / T = M L^2 T^-3
        let power = Dimension::ENERGY - Dimension::TIME;
        assert_eq!(power, Dimension::POWER);
    }

    #[test]
    fn test_radiative_flux() {
        // W/m^2 = Power / Area = M L^2 T^-3 / L^2 = M T^-3
        let flux = Dimension::POWER - Dimension::AREA;
        assert_eq!(flux, Dimension::RADIATIVE_FLUX);
    }

    #[test]
    fn test_dimension_pow() {
        let length_squared = Dimension::LENGTH.pow(2);
        assert_eq!(length_squared, Dimension::AREA);
    }

    #[test]
    fn test_is_compatible() {
        assert!(Dimension::MASS.is_compatible(&Dimension::MASS));
        assert!(!Dimension::MASS.is_compatible(&Dimension::LENGTH));
        assert!(Dimension::POWER.is_compatible(&Dimension::POWER));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Dimension::MASS), "M");
        assert_eq!(format!("{}", Dimension::AREA), "L^2");
        assert_eq!(format!("{}", Dimension::FORCE), "M L T^-2");
    }
}
