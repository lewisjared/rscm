//! Unit registry with climate-specific units and conversion factors.
//!
//! This module provides a registry of known units with their dimensions
//! and conversion factors to SI base units. It includes:
//!
//! - SI base units and prefixes
//! - Climate-specific units (carbon, emissions, concentrations)
//! - Time units commonly used in climate modelling
//!
//! # Conversion Factor Convention
//!
//! All conversion factors are defined as the multiplier to convert FROM
//! the registered unit TO the SI base unit. For example:
//! - GtC has factor 1e12 (1 GtC = 1e12 kg of carbon)
//! - yr has factor 31557600 (1 yr = 31557600 s, using 365.25 days)

use super::dimension::Dimension;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Information about a known unit.
#[derive(Debug, Clone)]
pub struct UnitInfo {
    /// The canonical name of this unit.
    pub name: String,
    /// The physical dimension of this unit.
    pub dimension: Dimension,
    /// Conversion factor to SI base units.
    pub to_si_factor: f64,
    /// Optional: the base unit this is derived from (for compound units like GtC).
    pub base_unit: Option<String>,
}

impl UnitInfo {
    /// Creates a new unit info.
    fn new(name: &str, dimension: Dimension, to_si_factor: f64) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            to_si_factor,
            base_unit: None,
        }
    }

    /// Creates a new unit info with a base unit reference.
    fn with_base(name: &str, dimension: Dimension, to_si_factor: f64, base: &str) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            to_si_factor,
            base_unit: Some(base.to_string()),
        }
    }
}

/// SI prefix multipliers.
#[derive(Debug, Clone, Copy)]
pub struct SiPrefix {
    pub symbol: &'static str,
    pub factor: f64,
}

/// All SI prefixes from yocto to yotta.
pub static SI_PREFIXES: &[SiPrefix] = &[
    SiPrefix {
        symbol: "Y",
        factor: 1e24,
    },
    SiPrefix {
        symbol: "Z",
        factor: 1e21,
    },
    SiPrefix {
        symbol: "E",
        factor: 1e18,
    },
    SiPrefix {
        symbol: "P",
        factor: 1e15,
    },
    SiPrefix {
        symbol: "T",
        factor: 1e12,
    },
    SiPrefix {
        symbol: "G",
        factor: 1e9,
    },
    SiPrefix {
        symbol: "M",
        factor: 1e6,
    },
    SiPrefix {
        symbol: "k",
        factor: 1e3,
    },
    SiPrefix {
        symbol: "h",
        factor: 1e2,
    },
    SiPrefix {
        symbol: "da",
        factor: 1e1,
    },
    SiPrefix {
        symbol: "d",
        factor: 1e-1,
    },
    SiPrefix {
        symbol: "c",
        factor: 1e-2,
    },
    SiPrefix {
        symbol: "m",
        factor: 1e-3,
    },
    SiPrefix {
        symbol: "u",
        factor: 1e-6,
    }, // using 'u' for micro (μ)
    SiPrefix {
        symbol: "n",
        factor: 1e-9,
    },
    SiPrefix {
        symbol: "p",
        factor: 1e-12,
    },
    SiPrefix {
        symbol: "f",
        factor: 1e-15,
    },
    SiPrefix {
        symbol: "a",
        factor: 1e-18,
    },
    SiPrefix {
        symbol: "z",
        factor: 1e-21,
    },
    SiPrefix {
        symbol: "y",
        factor: 1e-24,
    },
];

// Constants for time conversions
/// Seconds per year (using 365.25 days for astronomical year).
pub const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 3600.0;
/// Seconds per day.
pub const SECONDS_PER_DAY: f64 = 24.0 * 3600.0;
/// Seconds per hour.
pub const SECONDS_PER_HOUR: f64 = 3600.0;
/// Seconds per minute.
pub const SECONDS_PER_MINUTE: f64 = 60.0;

// Constants for carbon conversions
/// Molecular weight ratio CO2/C = 44/12.
pub const CO2_TO_C_RATIO: f64 = 44.0 / 12.0;
/// Molecular weight ratio C/CO2 = 12/44.
pub const C_TO_CO2_RATIO: f64 = 12.0 / 44.0;

/// The global unit registry.
pub static UNIT_REGISTRY: LazyLock<UnitRegistry> = LazyLock::new(UnitRegistry::new);

/// Registry of known units with their dimensions and conversion factors.
#[derive(Debug)]
pub struct UnitRegistry {
    /// Map from unit symbol to unit info.
    units: HashMap<&'static str, UnitInfo>,
    /// Map from alias to canonical name.
    aliases: HashMap<&'static str, &'static str>,
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UnitRegistry {
    /// Creates a new unit registry populated with standard units.
    pub fn new() -> Self {
        let mut registry = Self {
            units: HashMap::new(),
            aliases: HashMap::new(),
        };
        registry.register_base_units();
        registry.register_time_units();
        registry.register_carbon_units();
        registry.register_concentration_units();
        registry.register_energy_units();
        registry.register_temperature_units();
        registry
    }

    /// Looks up a unit by symbol, handling prefixes and aliases.
    pub fn lookup(&self, symbol: &str) -> Option<UnitInfo> {
        // First check for exact match
        if let Some(info) = self.units.get(symbol) {
            return Some(info.clone());
        }

        // Check aliases
        if let Some(&canonical) = self.aliases.get(symbol) {
            if let Some(info) = self.units.get(canonical) {
                return Some(info.clone());
            }
        }

        // Try to parse as prefixed unit
        self.lookup_prefixed(symbol)
    }

    /// Attempts to parse a symbol as a prefixed version of a base unit.
    fn lookup_prefixed(&self, symbol: &str) -> Option<UnitInfo> {
        // Sort prefixes by length descending to match longer prefixes first
        let mut prefixes: Vec<_> = SI_PREFIXES.iter().collect();
        prefixes.sort_by(|a, b| b.symbol.len().cmp(&a.symbol.len()));

        for prefix in prefixes {
            if let Some(base_symbol) = symbol.strip_prefix(prefix.symbol) {
                // Check if the remainder is a known base unit
                if let Some(base_info) = self.units.get(base_symbol) {
                    return Some(UnitInfo {
                        name: symbol.to_string(),
                        dimension: base_info.dimension,
                        to_si_factor: base_info.to_si_factor * prefix.factor,
                        base_unit: Some(base_info.name.clone()),
                    });
                }
                // Check aliases for base
                if let Some(&canonical) = self.aliases.get(base_symbol) {
                    if let Some(base_info) = self.units.get(canonical) {
                        return Some(UnitInfo {
                            name: symbol.to_string(),
                            dimension: base_info.dimension,
                            to_si_factor: base_info.to_si_factor * prefix.factor,
                            base_unit: Some(canonical.to_string()),
                        });
                    }
                }
            }
        }
        None
    }

    /// Registers SI base units.
    fn register_base_units(&mut self) {
        // Mass
        self.units
            .insert("kg", UnitInfo::new("kg", Dimension::MASS, 1.0));
        self.units
            .insert("g", UnitInfo::new("g", Dimension::MASS, 1e-3));
        self.units
            .insert("t", UnitInfo::new("t", Dimension::MASS, 1e3)); // metric tonne

        // Length
        self.units
            .insert("m", UnitInfo::new("m", Dimension::LENGTH, 1.0));

        // Time (base unit: second)
        self.units
            .insert("s", UnitInfo::new("s", Dimension::TIME, 1.0));

        // Temperature
        self.units
            .insert("K", UnitInfo::new("K", Dimension::TEMPERATURE, 1.0));

        // Amount of substance
        self.units
            .insert("mol", UnitInfo::new("mol", Dimension::AMOUNT, 1.0));

        // Electric current
        self.units
            .insert("A", UnitInfo::new("A", Dimension::CURRENT, 1.0));

        // Dimensionless
        self.units
            .insert("1", UnitInfo::new("1", Dimension::dimensionless(), 1.0));
        self.aliases.insert("dimensionless", "1");
    }

    /// Registers time units.
    fn register_time_units(&mut self) {
        self.units
            .insert("yr", UnitInfo::new("yr", Dimension::TIME, SECONDS_PER_YEAR));
        self.units.insert(
            "day",
            UnitInfo::new("day", Dimension::TIME, SECONDS_PER_DAY),
        );
        self.units
            .insert("h", UnitInfo::new("h", Dimension::TIME, SECONDS_PER_HOUR));
        self.units.insert(
            "min",
            UnitInfo::new("min", Dimension::TIME, SECONDS_PER_MINUTE),
        );

        // Aliases
        self.aliases.insert("year", "yr");
        self.aliases.insert("years", "yr");
        self.aliases.insert("a", "yr"); // annum
        self.aliases.insert("days", "day");
        self.aliases.insert("hour", "h");
        self.aliases.insert("hours", "h");
        self.aliases.insert("minute", "min");
        self.aliases.insert("minutes", "min");
        self.aliases.insert("sec", "s");
        self.aliases.insert("second", "s");
        self.aliases.insert("seconds", "s");
    }

    /// Registers carbon and CO2 units.
    ///
    /// Carbon units use mass dimension. The conversion between C and CO2
    /// is handled by treating them as different "flavours" of mass that
    /// can be inter-converted using the molecular weight ratio.
    fn register_carbon_units(&mut self) {
        // Base carbon unit (kg of carbon)
        // C and CO2 are both mass, but with different interpretations
        // We'll track them separately and allow conversion between them

        // Carbon as mass (reference: kg of carbon)
        self.units
            .insert("C", UnitInfo::new("C", Dimension::MASS, 1.0));
        self.units
            .insert("tC", UnitInfo::with_base("tC", Dimension::MASS, 1e3, "C"));
        self.units
            .insert("ktC", UnitInfo::with_base("ktC", Dimension::MASS, 1e6, "C"));
        self.units
            .insert("MtC", UnitInfo::with_base("MtC", Dimension::MASS, 1e9, "C"));
        self.units.insert(
            "GtC",
            UnitInfo::with_base("GtC", Dimension::MASS, 1e12, "C"),
        );
        self.units.insert(
            "PgC",
            UnitInfo::with_base("PgC", Dimension::MASS, 1e12, "C"),
        ); // 1 Pg = 1 Gt

        // CO2 as mass (reference: kg of CO2)
        // Note: to_si_factor converts to kg, and we apply the C/CO2 ratio
        // so that CO2 and C units are interconvertible
        self.units
            .insert("CO2", UnitInfo::new("CO2", Dimension::MASS, C_TO_CO2_RATIO));
        self.units.insert(
            "tCO2",
            UnitInfo::with_base("tCO2", Dimension::MASS, 1e3 * C_TO_CO2_RATIO, "CO2"),
        );
        self.units.insert(
            "ktCO2",
            UnitInfo::with_base("ktCO2", Dimension::MASS, 1e6 * C_TO_CO2_RATIO, "CO2"),
        );
        self.units.insert(
            "MtCO2",
            UnitInfo::with_base("MtCO2", Dimension::MASS, 1e9 * C_TO_CO2_RATIO, "CO2"),
        );
        self.units.insert(
            "GtCO2",
            UnitInfo::with_base("GtCO2", Dimension::MASS, 1e12 * C_TO_CO2_RATIO, "CO2"),
        );
    }

    /// Registers concentration units.
    fn register_concentration_units(&mut self) {
        // Concentration units are dimensionless ratios
        self.units.insert(
            "ppm",
            UnitInfo::new("ppm", Dimension::dimensionless(), 1e-6),
        );
        self.units.insert(
            "ppb",
            UnitInfo::new("ppb", Dimension::dimensionless(), 1e-9),
        );
        self.units.insert(
            "ppt",
            UnitInfo::new("ppt", Dimension::dimensionless(), 1e-12),
        );
    }

    /// Registers energy and power units.
    fn register_energy_units(&mut self) {
        // Energy (base: Joule = kg m^2 s^-2)
        self.units
            .insert("J", UnitInfo::new("J", Dimension::ENERGY, 1.0));

        // Power (base: Watt = kg m^2 s^-3)
        self.units
            .insert("W", UnitInfo::new("W", Dimension::POWER, 1.0));
    }

    /// Registers temperature units.
    fn register_temperature_units(&mut self) {
        // Kelvin is the base unit
        // For temperature differences, degC = K
        self.units
            .insert("degC", UnitInfo::new("degC", Dimension::TEMPERATURE, 1.0));
        self.units.insert(
            "delta_degC",
            UnitInfo::new("delta_degC", Dimension::TEMPERATURE, 1.0),
        );

        // Aliases
        self.aliases.insert("celsius", "degC");
        self.aliases.insert("Celsius", "degC");
        self.aliases.insert("deg_C", "degC");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_unit_lookup() {
        let registry = UnitRegistry::new();

        let kg = registry.lookup("kg").expect("kg should exist");
        assert_eq!(kg.dimension, Dimension::MASS);
        assert!((kg.to_si_factor - 1.0).abs() < f64::EPSILON);

        let m = registry.lookup("m").expect("m should exist");
        assert_eq!(m.dimension, Dimension::LENGTH);
    }

    #[test]
    fn test_prefixed_unit_lookup() {
        let registry = UnitRegistry::new();

        let km = registry.lookup("km").expect("km should exist");
        assert_eq!(km.dimension, Dimension::LENGTH);
        assert!((km.to_si_factor - 1e3).abs() < f64::EPSILON);

        let gw = registry.lookup("GW").expect("GW should exist");
        assert_eq!(gw.dimension, Dimension::POWER);
        assert!((gw.to_si_factor - 1e9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_carbon_units() {
        let registry = UnitRegistry::new();

        let gtc = registry.lookup("GtC").expect("GtC should exist");
        assert_eq!(gtc.dimension, Dimension::MASS);
        assert!((gtc.to_si_factor - 1e12).abs() < f64::EPSILON);

        let gtco2 = registry.lookup("GtCO2").expect("GtCO2 should exist");
        assert_eq!(gtco2.dimension, Dimension::MASS);
        // GtCO2 factor should be 1e12 * (12/44) for conversion to carbon-equivalent kg
        let expected = 1e12 * C_TO_CO2_RATIO;
        assert!(
            (gtco2.to_si_factor - expected).abs() < 1e6,
            "GtCO2 factor {} != expected {}",
            gtco2.to_si_factor,
            expected
        );
    }

    #[test]
    fn test_time_units() {
        let registry = UnitRegistry::new();

        let yr = registry.lookup("yr").expect("yr should exist");
        assert_eq!(yr.dimension, Dimension::TIME);
        assert!((yr.to_si_factor - SECONDS_PER_YEAR).abs() < f64::EPSILON);

        // Test alias
        let year = registry.lookup("year").expect("year alias should work");
        assert_eq!(year.dimension, Dimension::TIME);
    }

    #[test]
    fn test_concentration_units() {
        let registry = UnitRegistry::new();

        let ppm = registry.lookup("ppm").expect("ppm should exist");
        assert!(ppm.dimension.is_dimensionless());
        assert!((ppm.to_si_factor - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_unit() {
        let registry = UnitRegistry::new();
        assert!(registry.lookup("unknown_unit").is_none());
    }

    #[test]
    fn test_carbon_conversion_factors() {
        let registry = UnitRegistry::new();

        let gtc = registry.lookup("GtC").unwrap();
        let mtco2 = registry.lookup("MtCO2").unwrap();

        // To convert GtC to MtCO2:
        // 1 GtC = 1e12 kg C
        // 1e12 kg C * (44/12) = 1e12 * 3.667 kg CO2
        // In MtCO2: 1e12 * 3.667 / 1e9 = 3666.67 MtCO2
        let gtc_to_mtco2 = gtc.to_si_factor / mtco2.to_si_factor;
        let expected = 1000.0 * CO2_TO_C_RATIO; // 3666.67
        assert!(
            (gtc_to_mtco2 - expected).abs() < 0.01,
            "GtC to MtCO2 factor {} != expected {}",
            gtc_to_mtco2,
            expected
        );
    }
}
