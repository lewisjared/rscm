//! Unit string parser with normalisation.
//!
//! This module parses unit strings into a structured representation,
//! handling various syntactic variations:
//!
//! - Exponents: `m^2`, `m**2`, `m2`
//! - Multiplication: `kg m`, `kg*m`, `kg·m`
//! - Division: `W/m^2`, `W m^-2`, `W per m^2`
//! - Whitespace: `W/m^2` == `W / m ^ 2`
//! - Species: `Mt CH4/yr`, `Gt C/yr` and the fused `GtC/yr`, `MtCO2/yr`
//!
//! # Grammar
//!
//! ```text
//! unit_expr  = term (('/' | 'per') term)*
//! term       = factor (('*' | '·' | ' ') factor)*
//! factor     = '(' unit_expr ')' exponent? | base_unit exponent? | mass ' ' species
//! base_unit  = [a-zA-Z_][a-zA-Z0-9_]*
//! exponent   = ('^' | '**')? '-'? [0-9]+
//! ```
//!
//! A species token follows a mass unit and names what the mass counts (see [`super::species`]).
//! The carbon family may also be written fused to its mass unit, as in `GtC` and `MtCO2`.
//! An unknown token after a mass unit is an [`ParseError::UnknownSpecies`].

use super::dimension::Dimension;
use super::registry::UNIT_REGISTRY;
use super::species::{self, Species, FUSED};
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
    /// A token after a mass unit that is neither a species nor a unit.
    UnknownSpecies(String),
    /// A species token that does not follow a mass unit.
    MisplacedSpecies(String),
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
            Self::UnknownSpecies(s) => write!(f, "unknown species: '{s}'"),
            Self::MisplacedSpecies(s) => {
                write!(f, "species '{s}' must follow a mass unit, as in 'Mt {s}'")
            }
            Self::InvalidExponent(e) => write!(f, "invalid exponent: '{e}'"),
            Self::UnexpectedChar(c) => write!(f, "unexpected character: '{c}'"),
            Self::ParseFailed(msg) => write!(f, "parse failed: {msg}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// One factor of a unit: a unit symbol, and the species it counts when it is a mass.
///
/// `Mt CH4` is the symbol `Mt` counting `CH4`, and `GtC` is the symbol `Gt` counting `C`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitFactor {
    /// The unit symbol, such as `Mt` or `W`.
    pub symbol: String,
    /// The canonical species token, such as `CH4`.
    pub species: Option<&'static str>,
}

impl UnitFactor {
    /// A factor without a species.
    #[must_use]
    pub fn unit(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            species: None,
        }
    }

    /// A mass factor counting a species.
    #[must_use]
    pub fn mass_of(symbol: &str, species: &Species) -> Self {
        Self {
            symbol: symbol.to_string(),
            species: Some(species.name),
        }
    }

    fn species_info(&self) -> Option<Species> {
        self.species.and_then(species::lookup)
    }

    /// The compact form: carbon fused (`GtC`), other species spaced (`Mt CH4`).
    fn written(&self) -> String {
        match self.species {
            None => self.symbol.clone(),
            Some("C") => format!("{}C", self.symbol),
            Some(species) => format!("{} {species}", self.symbol),
        }
    }

    fn written_with_exponent(&self, exp: i32) -> String {
        let written = self.written();
        match (exp, written.contains(' ')) {
            (1, _) => written,
            (_, true) => format!("({written})^{exp}"),
            (_, false) => format!("{written}^{exp}"),
        }
    }
}

impl fmt::Display for UnitFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.written())
    }
}

/// A parsed unit expression.
///
/// Represents a unit as a product of factors with integer exponents.
/// For example, `W/m^2` is represented as `{W: 1, m: -2}`
/// and `Mt CH4/yr` as `{Mt CH4: 1, yr: -1}`.
///
/// It serialises as its normalised string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct ParsedUnit {
    /// Map from factor to exponent, ordered for deterministic output.
    components: BTreeMap<UnitFactor, i32>,
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
    pub fn from_components(components: BTreeMap<UnitFactor, i32>) -> Self {
        let components = components
            .into_iter()
            .filter(|(_, exp)| *exp != 0)
            .collect();
        Self { components }
    }

    fn single(factor: UnitFactor, exp: i32) -> Self {
        Self::from_components(BTreeMap::from([(factor, exp)]))
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
    ///
    /// let carbon = ParsedUnit::parse("Gt C/yr").unwrap();
    /// assert_eq!(carbon, ParsedUnit::parse("GtC/yr").unwrap());
    /// assert_eq!(carbon.normalized(), "GtC/yr");
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError::EmptyUnit);
        }

        if input.eq_ignore_ascii_case("dimensionless") {
            return Ok(Self::dimensionless());
        }

        let mut parser = UnitParser::new(input);
        let unit = parser.parse_expression()?;
        parser.skip_whitespace();
        match parser.peek() {
            Some(c) => Err(ParseError::UnexpectedChar(c)),
            None => Ok(unit),
        }
    }

    /// Returns the components of this unit.
    #[must_use]
    pub fn components(&self) -> &BTreeMap<UnitFactor, i32> {
        &self.components
    }

    /// Returns the exponent of the factor with this symbol and no species.
    #[must_use]
    pub fn exponent_of(&self, symbol: &str) -> Option<i32> {
        self.components.get(&UnitFactor::unit(symbol)).copied()
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
    ///
    /// A species does not change the dimension: `Mt CH4` is a mass.
    pub fn dimension(&self) -> Result<Dimension, ParseError> {
        let mut result = Dimension::dimensionless();

        for (factor, &exp) in &self.components {
            let info = UNIT_REGISTRY
                .lookup(&factor.symbol)
                .ok_or_else(|| ParseError::UnknownUnit(factor.symbol.clone()))?;
            result = result + info.dimension.pow(exp as i8);
        }

        Ok(result)
    }

    /// Returns the net exponent of each species family, omitting zeros.
    ///
    /// Two units convert only when these match, so `Mt CH4` never converts to `Mt CO2`.
    #[must_use]
    pub fn species_families(&self) -> BTreeMap<&'static str, i32> {
        let mut families = BTreeMap::new();
        for (factor, &exp) in &self.components {
            if let Some(species) = factor.species_info() {
                *families.entry(species.family).or_insert(0) += exp;
            }
        }
        families.retain(|_, exp| *exp != 0);
        families
    }

    /// Computes the SI conversion factor for this unit.
    ///
    /// The factor is the multiplier to convert a value in this unit
    /// to the equivalent SI base units.
    /// A species mass is expressed in its family basis, so `Gt CO2` is 1e12 * 12/44 kg of carbon.
    pub fn to_si_factor(&self) -> Result<f64, ParseError> {
        let mut factor = 1.0;

        for (unit_factor, &exp) in &self.components {
            let info = UNIT_REGISTRY
                .lookup(&unit_factor.symbol)
                .ok_or_else(|| ParseError::UnknownUnit(unit_factor.symbol.clone()))?;
            factor *= info.to_si_factor.powi(exp);
            if let Some(species) = unit_factor.species_info() {
                factor *= species.basis_per_mass.powi(exp);
            }
        }

        Ok(factor)
    }

    /// Multiplies this unit by another unit.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let mut components = self.components.clone();
        for (factor, exp) in &other.components {
            *components.entry(factor.clone()).or_insert(0) += exp;
        }
        Self::from_components(components)
    }

    /// Divides this unit by another unit.
    #[must_use]
    pub fn divide(&self, other: &Self) -> Self {
        self.multiply(&other.pow(-1))
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

    /// Returns a normalised string representation of this unit.
    ///
    /// The normalised form is canonical and compact: factors with positive exponents
    /// first (alphabetically), then `/`, then factors with negative exponents.
    /// There are no spaces around `/` or `^`, and factors are separated by a space.
    /// Carbon is fused to its mass unit (`GtC/yr`) and other species are spaced (`Mt CH4/yr`).
    #[must_use]
    pub fn normalized(&self) -> String {
        let format_part = |parts: Vec<String>| parts.join(" ");

        let numerator: Vec<String> = self
            .components
            .iter()
            .filter(|(_, exp)| **exp > 0)
            .map(|(factor, &exp)| factor.written_with_exponent(exp))
            .collect();
        let denominator: Vec<String> = self
            .components
            .iter()
            .filter(|(_, exp)| **exp < 0)
            .map(|(factor, &exp)| factor.written_with_exponent(-exp))
            .collect();

        match (numerator.is_empty(), denominator.is_empty()) {
            (true, true) => "1".to_string(),
            (false, true) => format_part(numerator),
            (true, false) => format!("1/{}", format_part(denominator)),
            (false, false) => format!("{}/{}", format_part(numerator), format_part(denominator)),
        }
    }
}

impl fmt::Display for ParsedUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.normalized())
    }
}

impl From<ParsedUnit> for String {
    fn from(unit: ParsedUnit) -> Self {
        unit.normalized()
    }
}

impl TryFrom<String> for ParsedUnit {
    type Error = ParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

/// Returns true if `symbol` is a registered mass unit, such as `Mt` or `kg`.
fn is_mass_unit(symbol: &str) -> bool {
    UNIT_REGISTRY
        .lookup(symbol)
        .is_some_and(|info| info.dimension == Dimension::MASS)
}

/// Splits a fused carbon token such as `GtC` or `MtCO2` into its mass unit and species.
fn split_fused(token: &str) -> Option<(&str, Species)> {
    FUSED.iter().find_map(|suffix| {
        let mass = token.strip_suffix(suffix)?;
        is_mass_unit(mass).then(|| (mass, species::lookup(suffix).expect("fused species")))
    })
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
        let (mut result, mut bare_mass) = self.parse_factor()?;

        loop {
            self.skip_whitespace();

            let next = self.peek();
            if next == Some('*') || next == Some('\u{00B7}') {
                self.advance();
                self.skip_whitespace();
                let (factor, mass) = self.parse_factor()?;
                result = result.multiply(&factor);
                bare_mass = mass;
            } else if next.is_some_and(|c| self.is_unit_start(c)) && !self.check_keyword("per") {
                // Implicit multiplication, or a species naming what the preceding mass counts.
                let token = self.peek_token();
                if let Some(mass) = bare_mass.take().filter(|_| !token.is_empty()) {
                    if let Some(species) = species::lookup(token) {
                        self.pos += token.len();
                        result = result
                            .divide(&ParsedUnit::single(UnitFactor::unit(&mass), 1))
                            .multiply(&ParsedUnit::single(UnitFactor::mass_of(&mass, &species), 1));
                        continue;
                    }
                    if !Self::is_unit_token(token) {
                        return Err(ParseError::UnknownSpecies(token.to_string()));
                    }
                }
                let (factor, mass) = self.parse_factor()?;
                result = result.multiply(&factor);
                bare_mass = mass;
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Parses one factor, also returning its symbol when it is a bare mass a species may follow.
    fn parse_factor(&mut self) -> Result<(ParsedUnit, Option<String>), ParseError> {
        self.skip_whitespace();

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

            let exp = self.parse_optional_exponent()?;
            return Ok((inner.pow(exp), None));
        }

        let factor = self.parse_symbol()?;
        let exp = self.parse_optional_exponent()?;

        let bare_mass = (factor.species.is_none() && exp == 1 && is_mass_unit(&factor.symbol))
            .then(|| factor.symbol.clone());
        let unit = if factor.symbol == "1" {
            ParsedUnit::dimensionless()
        } else {
            ParsedUnit::single(factor, exp)
        };
        Ok((unit, bare_mass))
    }

    fn parse_symbol(&mut self) -> Result<UnitFactor, ParseError> {
        self.skip_whitespace();
        let start = self.pos;
        let full_symbol = self.peek_token();

        if full_symbol.is_empty() {
            return Err(ParseError::ParseFailed("expected unit symbol".into()));
        }
        self.pos += full_symbol.len();

        if species::lookup(full_symbol).is_some() {
            return Err(ParseError::MisplacedSpecies(full_symbol.to_string()));
        }
        if UNIT_REGISTRY.lookup(full_symbol).is_some() {
            return Ok(UnitFactor::unit(full_symbol));
        }
        if let Some((mass, species)) = split_fused(full_symbol) {
            return Ok(UnitFactor::mass_of(mass, &species));
        }

        // Trailing digits that are not part of a known unit are an exponent, as in `m2`.
        if let Some(last_letter_idx) = full_symbol.rfind(|c: char| c.is_ascii_alphabetic()) {
            let base = &full_symbol[..=last_letter_idx];
            let trailing = &full_symbol[last_letter_idx + 1..];

            if !trailing.is_empty() && trailing.chars().all(|c| c.is_ascii_digit()) {
                self.pos = start + last_letter_idx + 1;
                return Ok(UnitFactor::unit(base));
            }
        }

        Ok(UnitFactor::unit(full_symbol))
    }

    /// Returns true if `token` reads as a unit, possibly with an implicit exponent as in `m2`.
    fn is_unit_token(token: &str) -> bool {
        if UNIT_REGISTRY.lookup(token).is_some() || split_fused(token).is_some() {
            return true;
        }
        let base = token.trim_end_matches(|c: char| c.is_ascii_digit());
        base != token && UNIT_REGISTRY.lookup(base).is_some()
    }

    fn parse_optional_exponent(&mut self) -> Result<i32, ParseError> {
        self.skip_whitespace();

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

        if let Some(c) = self.peek() {
            if c == '-' || c.is_ascii_digit() {
                return self.parse_exponent();
            }
        }

        if has_marker {
            return Err(ParseError::ParseFailed("expected exponent after ^".into()));
        }

        Ok(1)
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

    /// The run of alphanumerics and underscores at the current position.
    fn peek_token(&self) -> &'a str {
        let rest = &self.input[self.pos..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        &rest[..end]
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

    fn mass_of(symbol: &str, species: &str) -> UnitFactor {
        UnitFactor::mass_of(symbol, &species::lookup(species).unwrap())
    }

    #[test]
    fn test_parse_simple_unit() {
        let unit = ParsedUnit::parse("W").unwrap();
        assert_eq!(unit.exponent_of("W"), Some(1));
    }

    #[test]
    fn test_parse_unit_with_exponent() {
        let unit = ParsedUnit::parse("m^2").unwrap();
        assert_eq!(unit.exponent_of("m"), Some(2));

        let unit2 = ParsedUnit::parse("m**2").unwrap();
        assert_eq!(unit2.exponent_of("m"), Some(2));
    }

    #[test]
    fn test_parse_division() {
        let unit = ParsedUnit::parse("W/m^2").unwrap();
        assert_eq!(unit.exponent_of("W"), Some(1));
        assert_eq!(unit.exponent_of("m"), Some(-2));
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
        assert_eq!(unit1.exponent_of("kg"), Some(1));
        assert_eq!(unit1.exponent_of("m"), Some(1));
    }

    #[test]
    fn test_parse_compound_unit() {
        let unit = ParsedUnit::parse("GtC/yr").unwrap();
        assert_eq!(unit.components().get(&mass_of("Gt", "C")), Some(&1));
        assert_eq!(unit.exponent_of("yr"), Some(-1));
    }

    #[test]
    fn test_parse_co2_unit() {
        let unit = ParsedUnit::parse("MtCO2/yr").unwrap();
        assert_eq!(unit.components().get(&mass_of("Mt", "CO2")), Some(&1));
        assert_eq!(unit.exponent_of("yr"), Some(-1));
    }

    #[test]
    fn test_parse_negative_exponent() {
        let unit = ParsedUnit::parse("W m^-2").unwrap();
        assert_eq!(unit.exponent_of("W"), Some(1));
        assert_eq!(unit.exponent_of("m"), Some(-2));
    }

    #[test]
    fn test_normalized_output() {
        let unit = ParsedUnit::parse("m^-2 W").unwrap();
        assert_eq!(unit.normalized(), "W/m^2");
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
        assert_eq!(kg_m.exponent_of("kg"), Some(1));
        assert_eq!(kg_m.exponent_of("m"), Some(1));
    }

    #[test]
    fn test_divide_units() {
        let w = ParsedUnit::parse("W").unwrap();
        let m2 = ParsedUnit::parse("m^2").unwrap();
        let flux = w.divide(&m2);
        assert_eq!(flux.exponent_of("W"), Some(1));
        assert_eq!(flux.exponent_of("m"), Some(-2));
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
        assert_eq!(unit.exponent_of("m"), Some(2));
    }

    #[test]
    fn test_spaced_and_fused_species_are_equal() {
        for (spaced, fused) in [
            ("Gt C", "GtC"),
            ("Gt C/yr", "GtC/yr"),
            ("Mt CO2/yr", "MtCO2/yr"),
            ("Pg C / yr", "PgC/yr"),
            ("t C", "tC"),
        ] {
            assert_eq!(
                ParsedUnit::parse(spaced).unwrap(),
                ParsedUnit::parse(fused).unwrap(),
                "{spaced} vs {fused}"
            );
        }
    }

    #[test]
    fn test_species_is_mass_not_product() {
        let unit = ParsedUnit::parse("Gt C").unwrap();
        assert_eq!(unit.dimension().unwrap(), Dimension::MASS);
        assert_eq!(unit.components().len(), 1);

        let unit = ParsedUnit::parse("Mt CH4/yr").unwrap();
        assert_eq!(unit.components().get(&mass_of("Mt", "CH4")), Some(&1));
        assert_eq!(unit.exponent_of("yr"), Some(-1));
        assert_eq!(unit.dimension().unwrap(), Dimension::MASS - Dimension::TIME);
    }

    #[test]
    fn test_every_species_parses() {
        for token in species::tokens() {
            let unit = ParsedUnit::parse(&format!("kt {token}/yr")).unwrap();
            assert_eq!(unit.components().len(), 2, "{token}");
            assert_eq!(unit.species_families().len(), 1, "{token}");
        }
    }

    #[test]
    fn test_synonyms_parse_to_canonical() {
        assert_eq!(
            ParsedUnit::parse("Mt NMVOC/yr").unwrap(),
            ParsedUnit::parse("Mt VOC/yr").unwrap()
        );
        assert_eq!(
            ParsedUnit::parse("kt HFC4310/yr").unwrap(),
            ParsedUnit::parse("kt HFC4310mee/yr").unwrap()
        );
    }

    #[test]
    fn test_species_in_denominator() {
        let unit = ParsedUnit::parse("K / TtC").unwrap();
        assert_eq!(unit.components().get(&mass_of("Tt", "C")), Some(&-1));
        assert_eq!(unit, ParsedUnit::parse("K/(Tt C)").unwrap());
        assert_eq!(unit, ParsedUnit::parse("K / Tt C").unwrap());
    }

    #[test]
    fn test_unknown_species_names_token() {
        assert_eq!(
            ParsedUnit::parse("Mt XYZ/yr"),
            Err(ParseError::UnknownSpecies("XYZ".into()))
        );
        assert_eq!(
            ParsedUnit::parse("kt SOx/yr"),
            Err(ParseError::UnknownSpecies("SOx".into()))
        );
        assert!(ParseError::UnknownSpecies("XYZ".into())
            .to_string()
            .contains("'XYZ'"));
    }

    #[test]
    fn test_misplaced_species() {
        for input in ["CO2", "C", "W CH4", "Mt^2 CO2", "Mt * CH4", "yr CH4"] {
            assert!(
                matches!(
                    ParsedUnit::parse(input),
                    Err(ParseError::MisplacedSpecies(_))
                ),
                "{input}: {:?}",
                ParsedUnit::parse(input)
            );
        }
    }

    #[test]
    fn test_units_still_follow_mass() {
        let unit = ParsedUnit::parse("kg m2").unwrap();
        assert_eq!(unit.exponent_of("kg"), Some(1));
        assert_eq!(unit.exponent_of("m"), Some(2));

        let unit = ParsedUnit::parse("kg (m/s)").unwrap();
        assert_eq!(unit.exponent_of("s"), Some(-1));

        let unit = ParsedUnit::parse("Mt per yr").unwrap();
        assert_eq!(unit.exponent_of("yr"), Some(-1));
    }

    #[test]
    fn test_trailing_input_rejected() {
        assert_eq!(
            ParsedUnit::parse("Mt CO2-eq/yr"),
            Err(ParseError::UnexpectedChar('-'))
        );
        assert!(ParsedUnit::parse("Mt CO2^2").is_err());
    }

    #[test]
    fn test_normalized_is_compact() {
        for (input, expected) in [
            ("W / m ^ 2", "W/m^2"),
            ("GtC / yr", "GtC/yr"),
            ("Gt C/yr", "GtC/yr"),
            ("MtCO2/yr", "Mt CO2/yr"),
            ("Mt CO2 / yr", "Mt CO2/yr"),
            ("Mt CH4 / yr", "Mt CH4/yr"),
            ("kt N2O/yr", "kt N2O/yr"),
            ("Mt NMVOC/yr", "Mt VOC/yr"),
            ("kt HFC4310/yr", "kt HFC4310mee/yr"),
            ("1 / yr", "1/yr"),
            ("W yr m^-2 K^-1", "W yr/K m^2"),
            ("K / TtC", "K/TtC"),
            ("(Mt CH4)^2", "(Mt CH4)^2"),
            ("GtC^2", "GtC^2"),
        ] {
            assert_eq!(
                ParsedUnit::parse(input).unwrap().normalized(),
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn test_normalized_round_trips() {
        for input in [
            "W / m ^ 2",
            "Gt C / yr",
            "MtCO2/yr",
            "Mt CH4/yr",
            "kg CO2 / kg CH4",
            "(Mt N2O)^2 / yr",
            "1/yr",
            "W yr m^-2 K^-1",
            "K / Tt C",
        ] {
            let unit = ParsedUnit::parse(input).unwrap();
            let normalized = unit.normalized();
            assert_eq!(ParsedUnit::parse(&normalized).unwrap(), unit, "{input}");
            assert_eq!(
                ParsedUnit::parse(&normalized).unwrap().normalized(),
                normalized
            );
        }
    }

    #[test]
    fn test_species_si_factor_in_family_basis() {
        let gtco2 = ParsedUnit::parse("Gt CO2").unwrap();
        let expected = 1e12 * 12.0 / 44.0;
        assert!((gtco2.to_si_factor().unwrap() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_serde_uses_normalized_string() {
        let unit = ParsedUnit::parse("Mt CH4 / yr").unwrap();
        let json = serde_json::to_string(&unit).unwrap();
        assert_eq!(json, "\"Mt CH4/yr\"");
        let back: ParsedUnit = serde_json::from_str(&json).unwrap();
        assert_eq!(back, unit);
    }
}
