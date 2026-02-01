//! Unit string parser with normalization.
//!
//! This module parses unit strings into a structured representation,
//! handling various syntactic variations:
//!
//! - Exponents: `m^2`, `m**2`, `m2`
//! - Multiplication: `kg m`, `kg*m`, `kg·m`
//! - Division: `W/m^2`, `W m^-2`, `W per m^2`
//! - Whitespace: `W/m^2` == `W / m ^ 2`
//!
//! # Grammar
//!
//! ```text
//! unit_expr  = term (('/' | 'per') term)*
//! term       = factor (('*' | '·' | ' ') factor)*
//! factor     = base_unit ('^' | '**')? exponent?
//! base_unit  = [a-zA-Z_]+
//! exponent   = '-'? [0-9]+
//! ```

use super::dimension::Dimension;
use super::registry::UNIT_REGISTRY;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Error type for unit parsing failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Empty unit string.
    EmptyUnit,
    /// Unknown unit symbol.
    UnknownUnit(String),
    /// Invalid exponent format.
    InvalidExponent(String),
    /// Unexpected character in unit string.
    UnexpectedChar(char),
    /// General parse error.
    ParseFailed(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyUnit => write!(f, "empty unit string"),
            Self::UnknownUnit(u) => write!(f, "unknown unit: '{u}'"),
            Self::InvalidExponent(e) => write!(f, "invalid exponent: '{e}'"),
            Self::UnexpectedChar(c) => write!(f, "unexpected character: '{c}'"),
            Self::ParseFailed(msg) => write!(f, "parse failed: {msg}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// A parsed unit expression.
///
/// Represents a unit as a product of base units with integer exponents.
/// For example, `W/m^2` is represented as `{W: 1, m: -2}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedUnit {
    /// Map from base unit symbol to exponent.
    /// Using BTreeMap for deterministic ordering.
    components: BTreeMap<String, i32>,
}

impl ParsedUnit {
    /// Creates a new empty (dimensionless) parsed unit.
    #[must_use]
    pub fn dimensionless() -> Self {
        Self {
            components: BTreeMap::new(),
        }
    }

    /// Creates a parsed unit from components.
    #[must_use]
    pub fn from_components(components: BTreeMap<String, i32>) -> Self {
        // Filter out zero exponents
        let components = components
            .into_iter()
            .filter(|(_, exp)| *exp != 0)
            .collect();
        Self { components }
    }

    /// Parses a unit string into a `ParsedUnit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_core::units::parser::ParsedUnit;
    ///
    /// let unit = ParsedUnit::parse("W/m^2").unwrap();
    /// let unit2 = ParsedUnit::parse("W / m ^ 2").unwrap();
    /// assert_eq!(unit, unit2);
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::EmptyUnit);
        }

        // Handle special case: "1" or "dimensionless"
        if input == "1" || input.eq_ignore_ascii_case("dimensionless") {
            return Ok(Self::dimensionless());
        }

        let mut parser = UnitParser::new(input);
        parser.parse_expression()
    }

    /// Returns the components of this unit.
    #[must_use]
    pub fn components(&self) -> &BTreeMap<String, i32> {
        &self.components
    }

    /// Returns true if this unit has no components (explicit dimensionless).
    ///
    /// Note: A unit like "ppm" has components but is physically dimensionless.
    /// Use [`dimension()`] to check the physical dimension.
    #[must_use]
    pub fn has_no_components(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns true if this unit is physically dimensionless.
    ///
    /// This computes the actual dimension and checks if it's dimensionless.
    /// Units like "ppm" and "ppb" are dimensionless even though they have components.
    pub fn is_dimensionless(&self) -> Result<bool, ParseError> {
        Ok(self.dimension()?.is_dimensionless())
    }

    /// Computes the overall dimension of this unit.
    pub fn dimension(&self) -> Result<Dimension, ParseError> {
        let mut result = Dimension::dimensionless();

        for (symbol, &exp) in &self.components {
            let info = UNIT_REGISTRY
                .lookup(symbol)
                .ok_or_else(|| ParseError::UnknownUnit(symbol.clone()))?;
            result = result + info.dimension.pow(exp as i8);
        }

        Ok(result)
    }

    /// Computes the SI conversion factor for this unit.
    ///
    /// The factor is the multiplier to convert a value in this unit
    /// to the equivalent SI base units.
    pub fn to_si_factor(&self) -> Result<f64, ParseError> {
        let mut factor = 1.0;

        for (symbol, &exp) in &self.components {
            let info = UNIT_REGISTRY
                .lookup(symbol)
                .ok_or_else(|| ParseError::UnknownUnit(symbol.clone()))?;
            factor *= info.to_si_factor.powi(exp);
        }

        Ok(factor)
    }

    /// Multiplies this unit by another unit.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let mut components = self.components.clone();
        for (symbol, exp) in &other.components {
            *components.entry(symbol.clone()).or_insert(0) += exp;
        }
        Self::from_components(components)
    }

    /// Divides this unit by another unit.
    #[must_use]
    pub fn divide(&self, other: &Self) -> Self {
        let mut components = self.components.clone();
        for (symbol, exp) in &other.components {
            *components.entry(symbol.clone()).or_insert(0) -= exp;
        }
        Self::from_components(components)
    }

    /// Raises this unit to a power.
    #[must_use]
    pub fn pow(&self, exp: i32) -> Self {
        let components = self
            .components
            .iter()
            .map(|(k, v)| (k.clone(), v * exp))
            .collect();
        Self::from_components(components)
    }

    /// Returns a normalized string representation of this unit.
    ///
    /// The normalized form is canonical: units with positive exponents
    /// first (alphabetically), then `/`, then units with negative exponents.
    #[must_use]
    pub fn normalized(&self) -> String {
        if self.components.is_empty() {
            return "1".to_string();
        }

        let mut numerator: Vec<(&str, i32)> = Vec::new();
        let mut denominator: Vec<(&str, i32)> = Vec::new();

        for (symbol, &exp) in &self.components {
            if exp > 0 {
                numerator.push((symbol, exp));
            } else if exp < 0 {
                denominator.push((symbol, -exp));
            }
        }

        // Sort for deterministic output
        numerator.sort_by_key(|(s, _)| *s);
        denominator.sort_by_key(|(s, _)| *s);

        let format_part = |parts: &[(&str, i32)]| -> String {
            parts
                .iter()
                .map(|(s, e)| {
                    if *e == 1 {
                        s.to_string()
                    } else {
                        format!("{s}^{e}")
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let num_str = format_part(&numerator);
        let den_str = format_part(&denominator);

        match (num_str.is_empty(), den_str.is_empty()) {
            (true, true) => "1".to_string(),
            (false, true) => num_str,
            (true, false) => format!("1 / {den_str}"),
            (false, false) => format!("{num_str} / {den_str}"),
        }
    }
}

impl fmt::Display for ParsedUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.normalized())
    }
}

/// Internal parser for unit strings.
struct UnitParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> UnitParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse_expression(&mut self) -> Result<ParsedUnit, ParseError> {
        self.skip_whitespace();
        let mut result = self.parse_term()?;

        loop {
            self.skip_whitespace();
            if self.peek() == Some('/') || self.check_keyword("per") {
                if self.peek() == Some('/') {
                    self.advance();
                } else {
                    self.skip_keyword("per");
                }
                self.skip_whitespace();
                let divisor = self.parse_term()?;
                result = result.divide(&divisor);
            } else {
                break;
            }
        }

        Ok(result)
    }

    fn parse_term(&mut self) -> Result<ParsedUnit, ParseError> {
        let mut result = self.parse_factor()?;

        loop {
            self.skip_whitespace();

            // Check for multiplication operators or implicit multiplication
            let next = self.peek();
            if next == Some('*') || next == Some('\u{00B7}') {
                // · (middle dot)
                self.advance();
                self.skip_whitespace();
                let factor = self.parse_factor()?;
                result = result.multiply(&factor);
            } else if next.is_some()
                && next != Some('/')
                && !self.check_keyword("per")
                && self.is_unit_start(next.unwrap())
            {
                // Implicit multiplication (space-separated)
                let factor = self.parse_factor()?;
                result = result.multiply(&factor);
            } else {
                break;
            }
        }

        Ok(result)
    }

    fn parse_factor(&mut self) -> Result<ParsedUnit, ParseError> {
        self.skip_whitespace();

        // Check for parentheses
        if self.peek() == Some('(') {
            self.advance();
            let inner = self.parse_expression()?;
            self.skip_whitespace();
            if self.peek() != Some(')') {
                return Err(ParseError::ParseFailed(
                    "missing closing parenthesis".into(),
                ));
            }
            self.advance();

            // Check for exponent after parentheses
            let exp = self.parse_optional_exponent()?;
            return Ok(inner.pow(exp));
        }

        // Parse base unit
        let symbol = self.parse_symbol()?;
        let exp = self.parse_optional_exponent()?;

        let mut components = BTreeMap::new();
        components.insert(symbol, exp);
        Ok(ParsedUnit::from_components(components))
    }

    fn parse_symbol(&mut self) -> Result<String, ParseError> {
        self.skip_whitespace();
        let start = self.pos;

        // First, consume all alphanumeric characters to get the maximum possible symbol
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        if self.pos == start {
            return Err(ParseError::ParseFailed("expected unit symbol".into()));
        }

        let full_symbol = &self.input[start..self.pos];

        // Try to find the longest suffix that looks like an exponent
        // E.g., "m2" -> "m" with exponent 2, but "CO2" -> "CO2" (known unit)
        // We check from the end: if trailing digits are NOT part of a known unit,
        // they're an exponent
        if let Some(last_letter_idx) = full_symbol.rfind(|c: char| c.is_ascii_alphabetic()) {
            let base = &full_symbol[..=last_letter_idx];
            let trailing = &full_symbol[last_letter_idx + 1..];

            if !trailing.is_empty() && trailing.chars().all(|c| c.is_ascii_digit()) {
                // Check if full_symbol is a known unit
                if UNIT_REGISTRY.lookup(full_symbol).is_some() {
                    // It's a known unit like CO2, keep as-is
                    return Ok(full_symbol.to_string());
                }
                // Check if base is a known unit
                if UNIT_REGISTRY.lookup(base).is_some() {
                    // Treat trailing digits as exponent - rewind position
                    self.pos = start + last_letter_idx + 1;
                    return Ok(base.to_string());
                }
                // Neither is known, assume trailing is exponent (common case like m2)
                self.pos = start + last_letter_idx + 1;
                return Ok(base.to_string());
            }
        }

        Ok(full_symbol.to_string())
    }

    fn parse_optional_exponent(&mut self) -> Result<i32, ParseError> {
        self.skip_whitespace();

        // Check for explicit exponent markers
        let has_marker = if self.peek() == Some('^') {
            self.advance();
            // Also handle '**'
            if self.peek() == Some('*') {
                self.advance();
            }
            true
        } else if self.input[self.pos..].starts_with("**") {
            self.pos += 2;
            true
        } else {
            false
        };

        self.skip_whitespace();

        // Check for exponent value
        if let Some(c) = self.peek() {
            if c == '-' || c.is_ascii_digit() {
                return self.parse_exponent();
            }
        }

        if has_marker {
            return Err(ParseError::ParseFailed("expected exponent after ^".into()));
        }

        // Check for implicit exponent (digits immediately following unit)
        if let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                return self.parse_exponent();
            }
        }

        Ok(1) // Default exponent
    }

    fn parse_exponent(&mut self) -> Result<i32, ParseError> {
        let start = self.pos;
        let negative = if self.peek() == Some('-') {
            self.advance();
            true
        } else {
            false
        };

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        if self.pos == start || (negative && self.pos == start + 1) {
            return Err(ParseError::InvalidExponent(
                self.input[start..self.pos].to_string(),
            ));
        }

        let exp_str = &self.input[start..self.pos];
        exp_str
            .parse()
            .map_err(|_| ParseError::InvalidExponent(exp_str.to_string()))
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    fn is_unit_start(&self, c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_' || c == '('
    }

    fn check_keyword(&self, keyword: &str) -> bool {
        self.input[self.pos..].to_lowercase().starts_with(keyword)
            && self.input[self.pos + keyword.len()..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric())
    }

    fn skip_keyword(&mut self, keyword: &str) {
        if self.check_keyword(keyword) {
            self.pos += keyword.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_unit() {
        let unit = ParsedUnit::parse("W").unwrap();
        assert_eq!(unit.components().get("W"), Some(&1));
    }

    #[test]
    fn test_parse_unit_with_exponent() {
        let unit = ParsedUnit::parse("m^2").unwrap();
        assert_eq!(unit.components().get("m"), Some(&2));

        let unit2 = ParsedUnit::parse("m**2").unwrap();
        assert_eq!(unit2.components().get("m"), Some(&2));
    }

    #[test]
    fn test_parse_division() {
        let unit = ParsedUnit::parse("W/m^2").unwrap();
        assert_eq!(unit.components().get("W"), Some(&1));
        assert_eq!(unit.components().get("m"), Some(&-2));
    }

    #[test]
    fn test_parse_with_whitespace() {
        let unit1 = ParsedUnit::parse("W/m^2").unwrap();
        let unit2 = ParsedUnit::parse("W / m ^ 2").unwrap();
        let unit3 = ParsedUnit::parse("  W  /  m  ^  2  ").unwrap();
        assert_eq!(unit1, unit2);
        assert_eq!(unit2, unit3);
    }

    #[test]
    fn test_parse_multiplication() {
        let unit1 = ParsedUnit::parse("kg m").unwrap();
        let unit2 = ParsedUnit::parse("kg*m").unwrap();
        assert_eq!(unit1, unit2);
        assert_eq!(unit1.components().get("kg"), Some(&1));
        assert_eq!(unit1.components().get("m"), Some(&1));
    }

    #[test]
    fn test_parse_compound_unit() {
        let unit = ParsedUnit::parse("GtC/yr").unwrap();
        assert_eq!(unit.components().get("GtC"), Some(&1));
        assert_eq!(unit.components().get("yr"), Some(&-1));
    }

    #[test]
    fn test_parse_co2_unit() {
        let unit = ParsedUnit::parse("MtCO2/yr").unwrap();
        assert_eq!(unit.components().get("MtCO2"), Some(&1));
        assert_eq!(unit.components().get("yr"), Some(&-1));
    }

    #[test]
    fn test_parse_negative_exponent() {
        let unit = ParsedUnit::parse("W m^-2").unwrap();
        assert_eq!(unit.components().get("W"), Some(&1));
        assert_eq!(unit.components().get("m"), Some(&-2));
    }

    #[test]
    fn test_normalized_output() {
        let unit = ParsedUnit::parse("m^-2 W").unwrap();
        assert_eq!(unit.normalized(), "W / m^2");
    }

    #[test]
    fn test_dimensionless() {
        let unit = ParsedUnit::parse("1").unwrap();
        assert!(unit.has_no_components());
        assert!(unit.is_dimensionless().unwrap());

        let unit2 = ParsedUnit::parse("dimensionless").unwrap();
        assert!(unit2.has_no_components());
        assert!(unit2.is_dimensionless().unwrap());
    }

    #[test]
    fn test_dimension_calculation() {
        let unit = ParsedUnit::parse("W/m^2").unwrap();
        let dim = unit.dimension().unwrap();
        // W = M L^2 T^-3, so W/m^2 = M T^-3
        assert_eq!(dim, Dimension::RADIATIVE_FLUX);
    }

    #[test]
    fn test_to_si_factor() {
        let unit = ParsedUnit::parse("GtC").unwrap();
        let factor = unit.to_si_factor().unwrap();
        assert!((factor - 1e12).abs() < f64::EPSILON);

        let unit2 = ParsedUnit::parse("km").unwrap();
        let factor2 = unit2.to_si_factor().unwrap();
        assert!((factor2 - 1e3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiply_units() {
        let kg = ParsedUnit::parse("kg").unwrap();
        let m = ParsedUnit::parse("m").unwrap();
        let kg_m = kg.multiply(&m);
        assert_eq!(kg_m.components().get("kg"), Some(&1));
        assert_eq!(kg_m.components().get("m"), Some(&1));
    }

    #[test]
    fn test_divide_units() {
        let w = ParsedUnit::parse("W").unwrap();
        let m2 = ParsedUnit::parse("m^2").unwrap();
        let flux = w.divide(&m2);
        assert_eq!(flux.components().get("W"), Some(&1));
        assert_eq!(flux.components().get("m"), Some(&-2));
    }

    #[test]
    fn test_per_keyword() {
        let unit1 = ParsedUnit::parse("W/m^2").unwrap();
        let unit2 = ParsedUnit::parse("W per m^2").unwrap();
        assert_eq!(unit1, unit2);
    }

    #[test]
    fn test_empty_unit_error() {
        assert!(matches!(ParsedUnit::parse(""), Err(ParseError::EmptyUnit)));
        assert!(matches!(
            ParsedUnit::parse("   "),
            Err(ParseError::EmptyUnit)
        ));
    }

    #[test]
    fn test_complex_unit() {
        // kg m^2 s^-3 = Watt
        let unit = ParsedUnit::parse("kg m^2 s^-3").unwrap();
        let dim = unit.dimension().unwrap();
        assert_eq!(dim, Dimension::POWER);
    }

    #[test]
    fn test_implicit_exponent() {
        // Some systems write m2 instead of m^2
        let unit = ParsedUnit::parse("m2").unwrap();
        assert_eq!(unit.components().get("m"), Some(&2));
    }
}
