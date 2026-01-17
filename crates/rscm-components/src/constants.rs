//! Physical constants for climate model components
//!
//! This module contains fundamental physical constants used across climate model components.
//! All constants are documented with their units, typical values, and usage context.

use rscm_core::timeseries::FloatValue;

/// Conversion factor from gigatonnes of carbon (GtC) to parts per million (ppm) of atmospheric CO2
///
/// This constant relates the mass of carbon in the atmosphere to its concentration:
/// - **Value**: 2.13 GtC/ppm
/// - **Units**: GtC / ppm
/// - **Derivation**: Based on the total mass of the atmosphere (~5.15 x 10^18 kg) and the
///   molecular weights of CO2 (44 g/mol) and C (12 g/mol)
///
/// # Usage
///
/// Used primarily in carbon cycle components to convert between:
/// - Emissions (typically in GtC/yr or GtCO2/yr) and concentration changes (ppm/yr)
/// - Cumulative carbon budgets (GtC) and concentration anomalies (ppm)
///
/// # Example
///
/// ```rust
/// use rscm_components::constants::GTC_PER_PPM;
///
/// // Convert 10 GtC emissions to ppm change
/// let emissions_gtc = 10.0;
/// let concentration_change_ppm = emissions_gtc / GTC_PER_PPM;
/// assert!((concentration_change_ppm - 4.695).abs() < 0.01);
/// ```
///
/// # References
///
/// - IPCC AR5 WG1 Chapter 6, Table 6.1
/// - Friedlingstein et al. (2019), Global Carbon Budget
pub const GTC_PER_PPM: FloatValue = 2.13;
