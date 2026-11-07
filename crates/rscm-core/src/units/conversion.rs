//! Unit conversion calculations.
//!
//! This module provides the high-level [`Unit`] type that combines parsing,
//! normalisation, and conversion into a single ergonomic API.
//!
//! # Example
//!
//! ```
//! use rscm_core::units::Unit;
//!
//! // Parse and compare units (normalised comparison)
//! let u1 = Unit::parse("W/m^2").unwrap();
//! let u2 = Unit::parse("W / m ^ 2").unwrap();
//! assert_eq!(u1, u2);
//!
//! // Convert between compatible units
//! let gtc = Unit::parse("GtC/yr").unwrap();
//! let mtco2 = Unit::parse("MtCO2/yr").unwrap();
//! let factor = gtc.conversion_factor(&mtco2).unwrap();
//! // 1 GtC/yr = ~3666.67 MtCO2/yr
//! ```

use super::dimension::Dimension;
use super::parser::{ParseError, ParsedUnit};
use super::registry::UNIT_REGISTRY;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Error type for unit conversion failures.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionError {
    /// Units have incompatible dimensions.
    IncompatibleDimensions {
        from: Dimension,
        to: Dimension,
        from_unit: String,
        to_unit: String,
    },
    /// Units have the same dimension but count species of different families.
    IncompatibleSpecies { from_unit: String, to_unit: String },
    /// Failed to parse one of the units.
    ParseError(ParseError),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleDimensions {
                from,
                to,
                from_unit,
                to_unit,
            } => {
                write!(
                    f,
                    "cannot convert from '{from_unit}' to '{to_unit}': \
                     incompatible dimensions ({from} vs {to})"
                )
            }
            Self::IncompatibleSpecies { from_unit, to_unit } => write!(
                f,
                "cannot convert from '{from_unit}' to '{to_unit}': they count different species"
            ),
            Self::ParseError(e) => write!(f, "unit parse error: {e}"),
        }
    }
}

impl std::error::Error for ConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ParseError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ParseError> for ConversionError {
    fn from(e: ParseError) -> Self {
        Self::ParseError(e)
    }
}

/// A parsed and validated unit.
///
/// This is the primary type for working with units. It provides:
/// - Parsing from strings with flexible syntax
/// - Automatic normalisation for comparison
/// - Conversion factor calculation between compatible units
/// - Dimensional analysis
///
/// # Equality
///
/// Two units are equal if they have the same normalised representation.
/// This means `Unit::parse("W/m^2") == Unit::parse("W / m ^ 2")`
/// and `Unit::parse("Gt C/yr") == Unit::parse("GtC/yr")`.
///
/// # Example
///
/// ```
/// use rscm_core::units::Unit;
///
/// let gtc_yr = Unit::parse("GtC/yr").unwrap();
/// let mtco2_yr = Unit::parse("MtCO2/yr").unwrap();
///
/// // Check if conversion is possible
/// assert!(gtc_yr.is_compatible(&mtco2_yr));
///
/// // Get conversion factor
/// let factor = gtc_yr.conversion_factor(&mtco2_yr).unwrap();
/// println!("1 GtC/yr = {factor} MtCO2/yr");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unit {
    /// The original input string (preserved for display).
    original: String,
    /// The parsed unit representation.
    parsed: ParsedUnit,
}

impl Unit {
    /// Parses a unit string.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_core::units::Unit;
    ///
    /// // All these parse to equivalent units
    /// let u1 = Unit::parse("W/m^2").unwrap();
    /// let u2 = Unit::parse("W / m ^ 2").unwrap();
    /// let u3 = Unit::parse("W m^-2").unwrap();
    /// assert_eq!(u1, u2);
    /// assert_eq!(u2, u3);
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let parsed = ParsedUnit::parse(input)?;
        Ok(Self {
            original: input.to_string(),
            parsed,
        })
    }

    /// Returns the original input string.
    #[must_use]
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Returns the normalised string representation.
    #[must_use]
    pub fn normalised(&self) -> String {
        self.parsed.normalised()
    }

    /// Returns true if this unit is physically dimensionless.
    ///
    /// Units like "ppm", "ppb", and "1" are dimensionless.
    /// Returns false if the dimension cannot be computed (unknown unit).
    #[must_use]
    pub fn is_dimensionless(&self) -> bool {
        self.parsed.is_dimensionless().unwrap_or(false)
    }

    /// Returns the physical dimension of this unit.
    pub fn dimension(&self) -> Result<Dimension, ParseError> {
        self.parsed.dimension()
    }

    /// Returns the conversion factor to SI base units.
    pub fn to_si_factor(&self) -> Result<f64, ParseError> {
        self.parsed.to_si_factor()
    }

    /// Returns true if this unit can be converted to the target unit.
    ///
    /// Units are compatible if they have the same physical dimension
    /// and count species of the same families.
    /// `GtC/yr` and `Mt CO2/yr` are compatible, `Mt CH4/yr` and `Mt CO2/yr` are not.
    pub fn is_compatible(&self, other: &Self) -> bool {
        match (self.dimension(), other.dimension()) {
            (Ok(d1), Ok(d2)) => d1.is_compatible(&d2) && self.same_species_families(other),
            _ => false,
        }
    }

    fn same_species_families(&self, other: &Self) -> bool {
        self.parsed.species_families() == other.parsed.species_families()
    }

    /// Returns true if this unit divides by a time unit, such as `GtC / yr`.
    ///
    /// `W / m^2` is not per time, although its dimension includes time.
    pub fn is_per_time(&self) -> bool {
        self.parsed.components().iter().any(|(factor, &exp)| {
            exp < 0
                && UNIT_REGISTRY
                    .lookup(&factor.symbol)
                    .is_some_and(|info| info.dimension == Dimension::TIME)
        })
    }

    /// Calculates the conversion factor from this unit to the target unit.
    ///
    /// The factor is the multiplier to convert a value in `self` to a value
    /// in `other`. For example, if `self` is `GtC/yr` and `other` is `MtCO2/yr`,
    /// the factor is approximately 3666.67.
    ///
    /// # Errors
    ///
    /// Returns an error if the units have incompatible dimensions
    /// or count species of different families.
    ///
    /// # Example
    ///
    /// ```
    /// use rscm_core::units::Unit;
    ///
    /// let gtc = Unit::parse("GtC/yr").unwrap();
    /// let mtco2 = Unit::parse("MtCO2/yr").unwrap();
    /// let factor = gtc.conversion_factor(&mtco2).unwrap();
    ///
    /// // 10 GtC/yr in MtCO2/yr
    /// let value_gtc = 10.0;
    /// let value_mtco2 = value_gtc * factor;
    /// ```
    pub fn conversion_factor(&self, other: &Self) -> Result<f64, ConversionError> {
        let dim_self = self.dimension()?;
        let dim_other = other.dimension()?;

        if !dim_self.is_compatible(&dim_other) {
            return Err(ConversionError::IncompatibleDimensions {
                from: dim_self,
                to: dim_other,
                from_unit: self.original.clone(),
                to_unit: other.original.clone(),
            });
        }
        if !self.same_species_families(other) {
            return Err(ConversionError::IncompatibleSpecies {
                from_unit: self.original.clone(),
                to_unit: other.original.clone(),
            });
        }

        let factor_self = self.to_si_factor()?;
        let factor_other = other.to_si_factor()?;

        // To convert from self to other:
        // value_self * factor_self = SI_value
        // SI_value / factor_other = value_other
        // So: value_other = value_self * (factor_self / factor_other)
        Ok(factor_self / factor_other)
    }

    /// Converts a value from this unit to the target unit.
    ///
    /// This is a convenience method equivalent to:
    /// `value * self.conversion_factor(other)?`
    pub fn convert_to(&self, value: f64, other: &Self) -> Result<f64, ConversionError> {
        let factor = self.conversion_factor(other)?;
        Ok(value * factor)
    }
}

impl PartialEq for Unit {
    fn eq(&self, other: &Self) -> bool {
        // Compare by normalised representation
        self.parsed == other.parsed
    }
}

impl Eq for Unit {}

impl std::hash::Hash for Unit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the normalised string for consistency with PartialEq
        self.normalised().hash(state);
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.normalised())
    }
}

/// Checks if two unit strings are equivalent after normalisation.
///
/// This is a convenience function for quick comparisons without
/// keeping the parsed `Unit` around.
///
/// # Example
///
/// ```
/// use rscm_core::units::units_equal;
///
/// assert!(units_equal("W/m^2", "W / m ^ 2").unwrap());
/// assert!(!units_equal("W/m^2", "W/m").unwrap());
/// ```
pub fn units_equal(a: &str, b: &str) -> Result<bool, ParseError> {
    let unit_a = Unit::parse(a)?;
    let unit_b = Unit::parse(b)?;
    Ok(unit_a == unit_b)
}

/// Calculates the conversion factor between two unit strings.
///
/// # Example
///
/// ```
/// use rscm_core::units::conversion_factor;
///
/// let factor = conversion_factor("GtC/yr", "MtCO2/yr").unwrap();
/// // 1 GtC/yr ≈ 3666.67 MtCO2/yr
/// assert!((factor - 3666.67).abs() < 0.01);
/// ```
pub fn conversion_factor(from: &str, to: &str) -> Result<f64, ConversionError> {
    let from_unit = Unit::parse(from)?;
    let to_unit = Unit::parse(to)?;
    from_unit.conversion_factor(&to_unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CO2_TO_C_RATIO: f64 = 44.0 / 12.0;

    #[test]
    fn test_is_per_time() {
        for unit in [
            "GtC / yr",
            "kt / yr",
            "Gt N / yr",
            "Mt CO2 / yr",
            "GtC/yr",
            "kg / s",
        ] {
            assert!(Unit::parse(unit).unwrap().is_per_time(), "{unit}");
        }
        for unit in ["W / m^2", "K", "ppm", "yr", "1", "GtC", "W yr / m^2"] {
            assert!(!Unit::parse(unit).unwrap().is_per_time(), "{unit}");
        }
    }

    #[test]
    fn test_unit_parse() {
        let unit = Unit::parse("W/m^2").unwrap();
        assert_eq!(unit.normalised(), "W/m^2");
    }

    #[test]
    fn test_unit_equality() {
        let u1 = Unit::parse("W/m^2").unwrap();
        let u2 = Unit::parse("W / m ^ 2").unwrap();
        let u3 = Unit::parse("W m^-2").unwrap();
        assert_eq!(u1, u2);
        assert_eq!(u2, u3);
    }

    #[test]
    fn test_unit_inequality() {
        let u1 = Unit::parse("W/m^2").unwrap();
        let u2 = Unit::parse("W/m").unwrap();
        assert_ne!(u1, u2);
    }

    #[test]
    fn test_is_compatible() {
        let gtc = Unit::parse("GtC/yr").unwrap();
        let mtco2 = Unit::parse("MtCO2/yr").unwrap();
        assert!(gtc.is_compatible(&mtco2));

        let flux = Unit::parse("W/m^2").unwrap();
        assert!(!gtc.is_compatible(&flux));
    }

    #[test]
    fn test_conversion_factor_identity() {
        let u1 = Unit::parse("W/m^2").unwrap();
        let u2 = Unit::parse("W / m^2").unwrap();
        let factor = u1.conversion_factor(&u2).unwrap();
        assert!((factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gtc_to_mtco2() {
        let gtc = Unit::parse("GtC/yr").unwrap();
        let mtco2 = Unit::parse("MtCO2/yr").unwrap();
        let factor = gtc.conversion_factor(&mtco2).unwrap();

        // Expected: 1000 * (44/12) = 3666.67
        let expected = 1000.0 * CO2_TO_C_RATIO;
        assert!(
            (factor - expected).abs() < 0.01,
            "GtC/yr to MtCO2/yr: {factor} != {expected}"
        );
    }

    #[test]
    fn test_conversion_gtco2_to_gtc() {
        let gtco2 = Unit::parse("GtCO2").unwrap();
        let gtc = Unit::parse("GtC").unwrap();
        let factor = gtco2.conversion_factor(&gtc).unwrap();

        // Expected: 12/44 = 0.2727...
        let expected = 12.0 / 44.0;
        assert!(
            (factor - expected).abs() < 0.001,
            "GtCO2 to GtC: {factor} != {expected}"
        );
    }

    #[test]
    fn test_incompatible_dimensions_error() {
        let gtc = Unit::parse("GtC").unwrap();
        let flux = Unit::parse("W/m^2").unwrap();
        let result = gtc.conversion_factor(&flux);

        assert!(matches!(
            result,
            Err(ConversionError::IncompatibleDimensions { .. })
        ));
    }

    #[test]
    fn test_conversion_with_time() {
        let gtc_yr = Unit::parse("GtC/yr").unwrap();
        let gtc_s = Unit::parse("GtC/s").unwrap();
        let factor = gtc_yr.conversion_factor(&gtc_s).unwrap();

        // 1 GtC/yr = 1/(365.25*24*3600) GtC/s
        let expected = 1.0 / (365.25 * 24.0 * 3600.0);
        assert!(
            (factor - expected).abs() < 1e-15,
            "GtC/yr to GtC/s: {factor} != {expected}"
        );
    }

    #[test]
    fn test_km_to_m() {
        let km = Unit::parse("km").unwrap();
        let m = Unit::parse("m").unwrap();
        let factor = km.conversion_factor(&m).unwrap();
        assert!((factor - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_gw_to_w() {
        let gw = Unit::parse("GW").unwrap();
        let w = Unit::parse("W").unwrap();
        let factor = gw.conversion_factor(&w).unwrap();
        assert!((factor - 1e9).abs() < 1e-10);
    }

    #[test]
    fn test_convert_to() {
        let gtc = Unit::parse("GtC/yr").unwrap();
        let mtco2 = Unit::parse("MtCO2/yr").unwrap();

        let value = 10.0; // 10 GtC/yr
        let converted = gtc.convert_to(value, &mtco2).unwrap();

        let expected = 10.0 * 1000.0 * CO2_TO_C_RATIO;
        assert!(
            (converted - expected).abs() < 0.1,
            "10 GtC/yr = {converted} MtCO2/yr (expected {expected})"
        );
    }

    #[test]
    fn test_units_equal_helper() {
        assert!(units_equal("W/m^2", "W / m ^ 2").unwrap());
        assert!(!units_equal("W/m^2", "W/m").unwrap());
    }

    #[test]
    fn test_conversion_factor_helper() {
        let factor = conversion_factor("km", "m").unwrap();
        assert!((factor - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_ppm_dimensionless() {
        let ppm = Unit::parse("ppm").unwrap();
        assert!(ppm.is_dimensionless());
    }

    #[test]
    fn test_original_preserved() {
        let unit = Unit::parse("W / m ^ 2").unwrap();
        assert_eq!(unit.original(), "W / m ^ 2");
        assert_eq!(unit.normalised(), "W/m^2");
    }

    fn factor(from: &str, to: &str) -> f64 {
        conversion_factor(from, to).unwrap_or_else(|e| panic!("{from} -> {to}: {e}"))
    }

    fn assert_factor(from: &str, to: &str, expected: f64) {
        let actual = factor(from, to);
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs(),
            "{from} -> {to}: {actual} != {expected}"
        );
        let back = factor(to, from);
        assert!(
            (back * expected - 1.0).abs() < 1e-12,
            "{to} -> {from}: {back} != {}",
            1.0 / expected
        );
    }

    #[test]
    fn test_carbon_family() {
        assert_factor("Gt C", "GtC", 1.0);
        assert_factor("Gt C/yr", "Mt CO2/yr", 1000.0 * 44.0 / 12.0);
        assert_factor("GtC/yr", "Gt CO2/yr", 44.0 / 12.0);
        assert_factor("PgC/yr", "GtC/yr", 1.0);
    }

    #[test]
    fn test_sulfur_family() {
        assert_factor("Mt S/yr", "Mt SO2/yr", 64.0 / 32.0);
        assert_factor("Tg S/yr", "kt SO2/yr", 1000.0 * 64.0 / 32.0);
    }

    #[test]
    fn test_nitrogen_oxides_family() {
        assert_factor("Mt N/yr", "Mt NO2/yr", 46.0 / 14.0);
        assert_factor("Mt N/yr", "Mt NOx/yr", 46.0 / 14.0);
        assert_factor("Mt NOx/yr", "Mt NO2/yr", 1.0);
    }

    #[test]
    fn test_nitrous_oxide_family() {
        assert_factor("Mt N2ON/yr", "Mt N2O/yr", 44.0 / 28.0);
        assert_factor("Mt N2ON/yr", "kt N2O/yr", 1000.0 * 44.0 / 28.0);
    }

    #[test]
    fn test_standalone_species_convert_to_themselves() {
        assert_factor("Mt CH4/yr", "kt CH4/yr", 1000.0);
        assert_factor("Mt NMVOC/yr", "Mt VOC/yr", 1.0);
        assert_factor("kt HFC4310/yr", "t HFC4310mee/yr", 1000.0);
        assert_factor("kt SF6/yr", "kt SF6 / yr", 1.0);
    }

    #[test]
    fn test_different_species_never_convert() {
        for (from, to) in [
            ("Mt CH4/yr", "Mt CO2/yr"),
            ("Mt N/yr", "Mt N2ON/yr"),
            ("Mt NO2/yr", "kt N2O/yr"),
            ("Mt S/yr", "Mt C/yr"),
            ("Mt NH3/yr", "Mt N/yr"),
            ("kt CFC11/yr", "kt CFC12/yr"),
            ("Mt BC/yr", "Mt OC/yr"),
            ("Mt CH4/yr", "Mt/yr"),
        ] {
            let from_unit = Unit::parse(from).unwrap();
            let to_unit = Unit::parse(to).unwrap();
            assert!(!from_unit.is_compatible(&to_unit), "{from} -> {to}");
            assert!(
                matches!(
                    from_unit.conversion_factor(&to_unit),
                    Err(ConversionError::IncompatibleSpecies { .. })
                ),
                "{from} -> {to}"
            );
        }
    }

    #[test]
    fn test_species_ratio_unit_converts_within_family() {
        // 1 kg CO2 counts 12/44 kg C.
        assert_factor("kg CO2 / kg CH4", "kg C / kg CH4", 12.0 / 44.0);
    }

    #[test]
    fn test_spaced_carbon_equals_fused() {
        assert_eq!(Unit::parse("Gt C").unwrap(), Unit::parse("GtC").unwrap());
        assert_eq!(
            Unit::parse("Mt CO2/yr").unwrap(),
            Unit::parse("MtCO2/yr").unwrap()
        );
    }
}
