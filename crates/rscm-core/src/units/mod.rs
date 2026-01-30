//! Unit parsing, normalization, and conversion for climate model variables.
//!
//! This module provides comprehensive support for working with physical units
//! in climate models. It handles parsing of unit strings with flexible syntax,
//! dimensional analysis, and conversion factor calculation.
//!
//! # Features
//!
//! - **Flexible parsing**: Handles various notations for the same unit
//!   (`W/m^2`, `W / m ^ 2`, `W m^-2` are all equivalent)
//! - **Dimensional analysis**: Validates that conversions are physically meaningful
//! - **Climate-specific units**: Carbon (C, CO2), concentrations (ppm, ppb),
//!   radiative forcing (W/m^2), and emissions rates (GtC/yr)
//! - **Conversion factors**: Automatically calculates multipliers between
//!   compatible units including the CO2-C molecular weight ratio
//!
//! # Quick Start
//!
//! ```
//! use rscm_core::units::Unit;
//!
//! // Parse units with flexible syntax
//! let u1 = Unit::parse("W/m^2").unwrap();
//! let u2 = Unit::parse("W / m ^ 2").unwrap();
//! assert_eq!(u1, u2);  // Normalized comparison
//!
//! // Convert between compatible units
//! let gtc = Unit::parse("GtC/yr").unwrap();
//! let mtco2 = Unit::parse("MtCO2/yr").unwrap();
//!
//! // Check compatibility (same physical dimension)
//! assert!(gtc.is_compatible(&mtco2));
//!
//! // Get conversion factor
//! let factor = gtc.conversion_factor(&mtco2).unwrap();
//! // 1 GtC/yr ≈ 3666.67 MtCO2/yr
//!
//! // Incompatible units produce errors
//! let flux = Unit::parse("W/m^2").unwrap();
//! assert!(!gtc.is_compatible(&flux));
//! assert!(gtc.conversion_factor(&flux).is_err());
//! ```
//!
//! # Supported Syntax
//!
//! The parser accepts several equivalent notations:
//!
//! | Notation | Meaning |
//! |----------|---------|
//! | `m^2`, `m**2`, `m2` | Square metres |
//! | `W/m^2`, `W m^-2`, `W per m^2` | Watts per square metre |
//! | `kg m`, `kg*m`, `kg·m` | Kilogram-metres |
//! | `GtC / yr` | Gigatonnes of carbon per year |
//!
//! Whitespace is normalized automatically.
//!
//! # Climate-Specific Units
//!
//! The module includes units commonly used in climate modelling:
//!
//! - **Carbon**: `C`, `CO2` with automatic molecular weight conversion (44/12)
//! - **Mass prefixes**: `Gt` (giga-tonne), `Mt` (mega-tonne), `kt`, `t`, etc.
//! - **Concentrations**: `ppm`, `ppb`, `ppt`
//! - **Time**: `yr` (365.25 days), `day`, `h`, `min`, `s`
//! - **Energy/Power**: `W`, `J` with SI prefixes
//! - **Temperature**: `K`, `degC` (for differences)
//!
//! # Conversion Examples
//!
//! | From | To | Factor |
//! |------|-----|--------|
//! | GtC | MtCO2 | 3666.67 (1000 * 44/12) |
//! | GtCO2 | GtC | 0.273 (12/44) |
//! | GtC/yr | GtC/s | 3.17e-8 |
//! | km | m | 1000 |
//! | ppm | ppb | 1000 |
//!
//! # Module Structure
//!
//! - [`dimension`]: Physical dimension types (M, L, T, Θ, etc.)
//! - [`registry`]: Known units with conversion factors
//! - [`parser`]: Unit string parsing with normalization
//! - [`conversion`]: High-level [`Unit`] type and conversion API

pub mod conversion;
pub mod dimension;
pub mod parser;
pub mod python;
pub mod registry;

// Re-export the main types for convenient access
pub use conversion::{conversion_factor, units_equal, ConversionError, Unit};
pub use dimension::Dimension;
pub use parser::{ParseError, ParsedUnit};
pub use registry::{UnitInfo, UnitRegistry, UNIT_REGISTRY};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the main example from the task works correctly.
    #[test]
    fn test_main_api() {
        // Parsing and normalization
        let u1 = Unit::parse("W/m^2").unwrap();
        let u2 = Unit::parse("W / m ^ 2").unwrap();
        assert_eq!(u1, u2);

        // Conversion
        let gtc = Unit::parse("GtC/yr").unwrap();
        let mtco2 = Unit::parse("MtCO2/yr").unwrap();
        let factor = gtc.conversion_factor(&mtco2).unwrap();

        // Expected: 1000 * (44/12) = 3666.67
        assert!(
            (factor - 3666.67).abs() < 0.01,
            "Conversion factor was {factor}"
        );
    }

    /// Test that incompatible units produce errors.
    #[test]
    fn test_incompatible_error() {
        let gtc = Unit::parse("GtC").unwrap();
        let flux = Unit::parse("W/m^2").unwrap();

        assert!(!gtc.is_compatible(&flux));
        assert!(gtc.conversion_factor(&flux).is_err());
    }

    /// Test various equivalent notations.
    #[test]
    fn test_equivalent_notations() {
        // Exponent notations
        let a = Unit::parse("m^2").unwrap();
        let b = Unit::parse("m**2").unwrap();
        let c = Unit::parse("m2").unwrap();
        assert_eq!(a, b);
        assert_eq!(b, c);

        // Division notations
        let d = Unit::parse("W/m^2").unwrap();
        let e = Unit::parse("W m^-2").unwrap();
        let f = Unit::parse("W per m^2").unwrap();
        assert_eq!(d, e);
        assert_eq!(e, f);

        // Whitespace variations
        let g = Unit::parse("GtC/yr").unwrap();
        let h = Unit::parse("GtC / yr").unwrap();
        let i = Unit::parse("  GtC  /  yr  ").unwrap();
        assert_eq!(g, h);
        assert_eq!(h, i);
    }

    /// Test carbon-CO2 conversions in both directions.
    #[test]
    fn test_carbon_conversions() {
        // C to CO2
        let gtc = Unit::parse("GtC").unwrap();
        let gtco2 = Unit::parse("GtCO2").unwrap();
        let factor_c_to_co2 = gtc.conversion_factor(&gtco2).unwrap();
        assert!(
            (factor_c_to_co2 - 44.0 / 12.0).abs() < 0.001,
            "C to CO2: {factor_c_to_co2}"
        );

        // CO2 to C
        let factor_co2_to_c = gtco2.conversion_factor(&gtc).unwrap();
        assert!(
            (factor_co2_to_c - 12.0 / 44.0).abs() < 0.001,
            "CO2 to C: {factor_co2_to_c}"
        );

        // Round-trip
        let round_trip = factor_c_to_co2 * factor_co2_to_c;
        assert!((round_trip - 1.0).abs() < 1e-10, "Round trip: {round_trip}");
    }

    /// Test time unit conversions.
    #[test]
    fn test_time_conversions() {
        let yr = Unit::parse("yr").unwrap();
        let s = Unit::parse("s").unwrap();
        let factor = yr.conversion_factor(&s).unwrap();

        let expected = 365.25 * 24.0 * 3600.0;
        assert!(
            (factor - expected).abs() < 1.0,
            "yr to s: {factor} vs {expected}"
        );
    }

    /// Test concentration units are dimensionless.
    #[test]
    fn test_concentration_dimensionless() {
        let ppm = Unit::parse("ppm").unwrap();
        let ppb = Unit::parse("ppb").unwrap();

        assert!(ppm.is_dimensionless());
        assert!(ppb.is_dimensionless());
        assert!(ppm.is_compatible(&ppb));

        let factor = ppm.conversion_factor(&ppb).unwrap();
        assert!((factor - 1000.0).abs() < 1e-10, "ppm to ppb: {factor}");
    }

    /// Test SI prefix handling.
    #[test]
    fn test_si_prefixes() {
        let gw = Unit::parse("GW").unwrap();
        let mw = Unit::parse("MW").unwrap();
        let kw = Unit::parse("kW").unwrap();
        let w = Unit::parse("W").unwrap();

        // GW to MW
        let factor = gw.conversion_factor(&mw).unwrap();
        assert!((factor - 1000.0).abs() < 1e-10);

        // MW to kW
        let factor = mw.conversion_factor(&kw).unwrap();
        assert!((factor - 1000.0).abs() < 1e-10);

        // GW to W
        let factor = gw.conversion_factor(&w).unwrap();
        assert!((factor - 1e9).abs() < 1e-10);
    }

    /// Integration test: typical climate model unit conversion.
    #[test]
    fn test_climate_scenario() {
        // A typical emissions pathway might be given in GtCO2/yr
        // but the carbon cycle model wants GtC/yr
        let emissions_input = Unit::parse("GtCO2/yr").unwrap();
        let model_unit = Unit::parse("GtC/yr").unwrap();

        let factor = emissions_input.conversion_factor(&model_unit).unwrap();

        // 1 GtCO2/yr = (12/44) GtC/yr
        let expected = 12.0 / 44.0;
        assert!(
            (factor - expected).abs() < 0.001,
            "GtCO2/yr to GtC/yr: {factor} vs {expected}"
        );

        // Convert 40 GtCO2/yr to GtC/yr
        let value = 40.0; // GtCO2/yr
        let converted = emissions_input.convert_to(value, &model_unit).unwrap();
        let expected_value = 40.0 * 12.0 / 44.0; // ~10.9 GtC/yr
        assert!(
            (converted - expected_value).abs() < 0.01,
            "40 GtCO2/yr = {converted} GtC/yr"
        );
    }
}
