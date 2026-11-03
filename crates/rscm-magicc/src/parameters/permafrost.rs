//! Permafrost parameters
//!
//! Parameters for the MAGICC7 permafrost carbon feedback (module 12).

use rscm_core::timeseries::FloatValue;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

/// Parameters of one permafrost soil type (mineral soil or peat).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PermafrostSoilParameters {
    /// Fraction of the frozen area thawed per year and per K of summer warming
    /// above the melting threshold
    /// unit: 1/K/yr
    pub thaw_rate: FloatValue,
    /// Exponent applied to the summer warming above the melting threshold
    pub thaw_exponent: FloatValue,
    /// Aerobic decomposition rate relative to mineral soil aerobic decomposition
    pub decomposition_ratio: FloatValue,
    /// Q10 exponent of aerobic decomposition
    /// unit: K
    pub q10_aerobic_alpha: FloatValue,
    /// Q10 exponent of anaerobic decomposition
    /// unit: K
    pub q10_anaerobic_alpha: FloatValue,
    /// Anaerobic fraction of thawed area at zero moisture sensitivity
    pub anaerobic_initial_fraction: FloatValue,
    /// Upper bound of the anaerobic fraction of thawed area
    pub anaerobic_max_fraction: FloatValue,
    /// Sensitivity of the anaerobic fraction to soil moisture
    pub anaerobic_moisture_sensitivity: FloatValue,
    /// Fraction of anaerobically produced CH4 oxidised to CO2 in the soil
    pub ch4_oxidation_fraction: FloatValue,
}

impl PermafrostSoilParameters {
    /// MAGICC7 default mineral soil parameters.
    pub fn mineral_soil() -> Self {
        Self {
            thaw_rate: 0.1,
            thaw_exponent: 1.0,
            decomposition_ratio: 1.0,
            q10_aerobic_alpha: 308.56,
            q10_anaerobic_alpha: 308.56,
            anaerobic_initial_fraction: 0.05,
            anaerobic_max_fraction: 0.3,
            anaerobic_moisture_sensitivity: 0.0,
            ch4_oxidation_fraction: 0.25,
        }
    }

    /// MAGICC7 default peat parameters.
    pub fn peat() -> Self {
        Self {
            thaw_rate: 0.05,
            decomposition_ratio: 0.5,
            anaerobic_initial_fraction: 0.8,
            anaerobic_max_fraction: 0.9,
            ch4_oxidation_fraction: 0.6,
            ..Self::mineral_soil()
        }
    }
}

impl Default for PermafrostSoilParameters {
    fn default() -> Self {
        Self::mineral_soil()
    }
}

/// Parameters for the permafrost carbon feedback.
///
/// The permafrost region is split into zonal bands whose melting threshold rises
/// linearly from `melting_temp_min` in the south to `melting_temp_max` in the north.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PermafrostParameters {
    /// Number of zonal bands
    pub n_bands: usize,
    /// Arctic warming at which the southernmost band starts to thaw
    /// unit: K
    pub melting_temp_min: FloatValue,
    /// Arctic warming at which the northernmost band starts to thaw
    /// unit: K
    pub melting_temp_max: FloatValue,
    /// Ratio of Arctic to global mean warming
    pub arctic_amplification: FloatValue,
    /// Amplitude of the seasonal soil temperature cycle
    /// unit: K
    pub soil_annual_cycle_amplitude: FloatValue,
    /// Turnover time of aerobic mineral soil decomposition at the Q10 reference
    /// unit: yr
    pub aerobic_turnover_time: FloatValue,
    /// Anaerobic decomposition rate relative to aerobic
    pub anaerobic_decomposition_ratio: FloatValue,
    /// Q10 reference temperature (LPJ `T1`)
    /// unit: K
    pub q10_temp1: FloatValue,
    /// Q10 offset temperature (LPJ `T2`)
    /// unit: K
    pub q10_temp2: FloatValue,
    /// Slope of soil moisture against soil temperature
    /// unit: 1/K
    pub soil_water_slope: FloatValue,
    /// Soil moisture at the melting threshold
    pub soil_water_offset: FloatValue,
    /// Lower bound of soil moisture
    pub soil_water_min: FloatValue,
    /// Total carbon in the permafrost region
    /// unit: GtC
    pub total_pool: FloatValue,
    /// Mineral soil fraction of the carbon in the southernmost band
    pub mineral_soil_fraction_south: FloatValue,
    /// Mineral soil fraction of the carbon in the northernmost band
    pub mineral_soil_fraction_north: FloatValue,
    /// Shape of the zonal carbon distribution, from -1 (north heavy) to 1 (south heavy)
    pub zonal_pool_distribution: FloatValue,
    #[serde(deserialize_with = "mineral_soil_over_defaults")]
    pub mineral_soil: PermafrostSoilParameters,
    #[serde(deserialize_with = "peat_over_defaults")]
    pub peat: PermafrostSoilParameters,
}

/// Fields given for a soil type override that soil's own defaults.
fn over_defaults<'de, D: Deserializer<'de>>(
    deserializer: D,
    defaults: PermafrostSoilParameters,
) -> Result<PermafrostSoilParameters, D::Error> {
    let mut merged = serde_json::to_value(defaults).map_err(D::Error::custom)?;
    if let (Some(base), serde_json::Value::Object(given)) = (
        merged.as_object_mut(),
        serde_json::Value::deserialize(deserializer)?,
    ) {
        base.extend(given);
    }
    serde_json::from_value(merged).map_err(D::Error::custom)
}

fn mineral_soil_over_defaults<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<PermafrostSoilParameters, D::Error> {
    over_defaults(deserializer, PermafrostSoilParameters::mineral_soil())
}

fn peat_over_defaults<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<PermafrostSoilParameters, D::Error> {
    over_defaults(deserializer, PermafrostSoilParameters::peat())
}

impl Default for PermafrostParameters {
    fn default() -> Self {
        Self {
            n_bands: 50,
            melting_temp_min: 1.0,
            melting_temp_max: 12.5,
            arctic_amplification: 1.7,
            soil_annual_cycle_amplitude: 5.0,
            aerobic_turnover_time: 20.0,
            anaerobic_decomposition_ratio: 0.1,
            q10_temp1: 56.02,
            q10_temp2: 46.02,
            soil_water_slope: 0.02,
            soil_water_offset: 0.2,
            soil_water_min: 0.2,
            total_pool: 800.0,
            mineral_soil_fraction_south: 0.8,
            mineral_soil_fraction_north: 0.8,
            zonal_pool_distribution: 0.0,
            mineral_soil: PermafrostSoilParameters::mineral_soil(),
            peat: PermafrostSoilParameters::peat(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_soil_keeps_its_own_defaults() {
        let p: PermafrostParameters =
            serde_json::from_str(r#"{"peat": {"thaw_rate": 0.2}}"#).unwrap();
        assert_eq!(p.peat.thaw_rate, 0.2);
        assert_eq!(p.peat.ch4_oxidation_fraction, 0.6);
        assert_eq!(p.mineral_soil.ch4_oxidation_fraction, 0.25);
    }
}
