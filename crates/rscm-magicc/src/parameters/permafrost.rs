//! Permafrost parameters
//!
//! Parameters for the MAGICC7 permafrost carbon feedback.

use rscm_core::timeseries::FloatValue;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

/// Parameters of one permafrost soil type (mineral soil or peat).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PermafrostSoil {
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
    /// MAGICC7: PF_MS_ANAEROB_INITIAL_AREAFRACTION, PF_PEAT_ANAEROB_INITIAL_AREAFRACTION
    pub initial_anaerobic_fraction: FloatValue,
    /// Upper bound of the anaerobic fraction of thawed area
    /// MAGICC7: PF_MS_ANAEROB_MAX_AREAFRACTION, PF_PEAT_ANAEROB_MAX_AREAFRACTION
    pub maximum_anaerobic_fraction: FloatValue,
    /// Sensitivity of the anaerobic fraction to soil moisture
    pub anaerobic_moisture_sensitivity: FloatValue,
    /// Fraction of anaerobically produced CH4 oxidised to CO2 in the soil
    pub ch4_oxidation_fraction: FloatValue,
}

impl PermafrostSoil {
    /// MAGICC7 default mineral soil parameters.
    pub fn mineral_soil() -> Self {
        Self {
            thaw_rate: 0.1,
            thaw_exponent: 1.0,
            decomposition_ratio: 1.0,
            q10_aerobic_alpha: 308.56,
            q10_anaerobic_alpha: 308.56,
            initial_anaerobic_fraction: 0.05,
            maximum_anaerobic_fraction: 0.3,
            anaerobic_moisture_sensitivity: 0.0,
            ch4_oxidation_fraction: 0.25,
        }
    }

    /// MAGICC7 default peat parameters.
    pub fn peat() -> Self {
        Self {
            thaw_rate: 0.05,
            decomposition_ratio: 0.5,
            initial_anaerobic_fraction: 0.8,
            maximum_anaerobic_fraction: 0.9,
            ch4_oxidation_fraction: 0.6,
            ..Self::mineral_soil()
        }
    }
}

impl Default for PermafrostSoil {
    fn default() -> Self {
        Self::mineral_soil()
    }
}

/// Parameters for the permafrost carbon feedback.
///
/// The permafrost region is split into zonal bands whose melting threshold rises
/// linearly from `minimum_melting_temperature` in the south to `maximum_melting_temperature` in the north.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PermafrostParameters {
    /// Number of zonal bands
    pub n_bands: usize,
    /// Arctic warming at which the southernmost band starts to thaw
    /// unit: K
    /// MAGICC7: PF_MELTINGTEMP_MIN
    pub minimum_melting_temperature: FloatValue,
    /// Arctic warming at which the northernmost band starts to thaw
    /// unit: K
    /// MAGICC7: PF_MELTINGTEMP_MAX
    pub maximum_melting_temperature: FloatValue,
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
    /// MAGICC7: PF_Q10_TEMP1
    pub q10_temperature1: FloatValue,
    /// Q10 offset temperature (LPJ `T2`)
    /// unit: K
    /// MAGICC7: PF_Q10_TEMP2
    pub q10_temperature2: FloatValue,
    /// Slope of soil moisture against soil temperature
    /// unit: 1/K
    pub soil_water_slope: FloatValue,
    /// Soil moisture at the melting threshold
    pub soil_water_offset: FloatValue,
    /// Lower bound of soil moisture
    /// MAGICC7: PF_SOILWATER_MINW
    pub minimum_soil_water: FloatValue,
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
    pub mineral_soil: PermafrostSoil,
    #[serde(deserialize_with = "peat_over_defaults")]
    pub peat: PermafrostSoil,
}

/// Fields given for a soil type override that soil's own defaults.
fn over_defaults<'de, D: Deserializer<'de>>(
    deserializer: D,
    defaults: PermafrostSoil,
) -> Result<PermafrostSoil, D::Error> {
    let serde_json::Value::Object(given) = serde_json::Value::deserialize(deserializer)? else {
        return Err(D::Error::custom("soil parameters must be an object"));
    };
    let mut merged = serde_json::to_value(defaults).map_err(D::Error::custom)?;
    if let Some(base) = merged.as_object_mut() {
        base.extend(given);
    }
    serde_json::from_value(merged).map_err(D::Error::custom)
}

fn mineral_soil_over_defaults<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<PermafrostSoil, D::Error> {
    over_defaults(deserializer, PermafrostSoil::mineral_soil())
}

fn peat_over_defaults<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<PermafrostSoil, D::Error> {
    over_defaults(deserializer, PermafrostSoil::peat())
}

impl Default for PermafrostParameters {
    fn default() -> Self {
        Self {
            n_bands: 50,
            minimum_melting_temperature: 1.0,
            maximum_melting_temperature: 12.5,
            arctic_amplification: 1.7,
            soil_annual_cycle_amplitude: 5.0,
            aerobic_turnover_time: 20.0,
            anaerobic_decomposition_ratio: 0.1,
            q10_temperature1: 56.02,
            q10_temperature2: 46.02,
            soil_water_slope: 0.02,
            soil_water_offset: 0.2,
            minimum_soil_water: 0.2,
            total_pool: 800.0,
            mineral_soil_fraction_south: 0.8,
            mineral_soil_fraction_north: 0.8,
            zonal_pool_distribution: 0.0,
            mineral_soil: PermafrostSoil::mineral_soil(),
            peat: PermafrostSoil::peat(),
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

    #[test]
    fn test_non_object_soil_rejected() {
        for soil in [r#"{"peat": false}"#, r#"{"mineral_soil": []}"#] {
            assert!(serde_json::from_str::<PermafrostParameters>(soil).is_err());
        }
    }
}
