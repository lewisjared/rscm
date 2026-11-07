//! Sea level rise parameters
//!
//! Parameters for the MAGICC7 sea level rise module.
//! Defaults are the effective values of MAGICC7 v7.5.3's default configuration,
//! after its CMIP5 mean and tuning files are applied.

use rscm_core::timeseries::FloatValue;
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer, Serialize};

/// Fields given override `defaults`, the rest keep them.
fn over_defaults<'de, D: Deserializer<'de>, T: Serialize + DeserializeOwned>(
    deserializer: D,
    defaults: T,
) -> Result<T, D::Error> {
    let mut merged = serde_json::to_value(defaults).map_err(D::Error::custom)?;
    if let (Some(base), serde_json::Value::Object(given)) = (
        merged.as_object_mut(),
        serde_json::Value::deserialize(deserializer)?,
    ) {
        base.extend(given);
    }
    serde_json::from_value(merged).map_err(D::Error::custom)
}

/// Thermal expansion of the ocean, calculated by `ClimateUdeb`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThermalExpansionParameters {
    /// Coefficients of the expansion polynomial (`SLR_EXPANSION_PARAMS`)
    pub coefficients: [FloatValue; 6],
    /// Scaling of the ocean layer warming (`SLR_EXPANSION_SCALING`)
    pub scaling: FloatValue,
    /// First year whose warming expands the ocean
    pub start_year: FloatValue,
}

impl Default for ThermalExpansionParameters {
    fn default() -> Self {
        Self {
            coefficients: [52.2368, 0.9938, 1.0333, 0.6065, 24.2643, 0.7408],
            scaling: 0.8823828,
            start_year: 1850.0,
        }
    }
}

/// Mountain glaciers and small ice caps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GlacierParameters {
    /// Fraction of the gap to equilibrium closed per year at 1 K of disequilibrium
    /// unit: 1/yr
    pub sensitivity: FloatValue,
    /// Exponent applied to the temperature disequilibrium
    pub exponent: FloatValue,
    /// First year of glacier melt
    pub start_year: FloatValue,
    /// Warming at which `equilibrium_sea_level` is reached
    /// unit: K
    pub equilibrium_temperature: Vec<FloatValue>,
    /// Equilibrium sea level contribution at each `equilibrium_temperature`
    /// unit: mm
    pub equilibrium_sea_level: Vec<FloatValue>,
}

impl Default for GlacierParameters {
    fn default() -> Self {
        Self {
            sensitivity: 1.067337e-2,
            exponent: 7.881273e-2,
            start_year: 1850.0,
            equilibrium_temperature: (0..104).map(|i| i as FloatValue / 10.0).collect(),
            equilibrium_sea_level: GLACIER_EQUILIBRIUM_SEA_LEVEL.to_vec(),
        }
    }
}

/// `SLR_GL_EQUISLR` of `SLR_GL_PARAMS_CMIP5MEAN.CFG`.
const GLACIER_EQUILIBRIUM_SEA_LEVEL: [FloatValue; 104] = [
    81.2036066, 96.5700785, 111.295999, 125.319312, 138.674473, 151.415694, 163.471114, 174.854137,
    185.497754, 195.338875, 204.326503, 212.564171, 220.350657, 227.743656, 234.966249, 242.181759,
    249.221825, 256.105628, 262.812856, 269.309297, 275.554553, 281.514538, 287.169530, 292.474062,
    297.434384, 302.245560, 306.979415, 311.593412, 316.020836, 320.196512, 324.056678, 327.587346,
    330.830768, 333.810833, 336.559538, 339.208759, 341.846291, 344.494723, 347.115199, 349.664732,
    352.099691, 354.402783, 356.569643, 358.607040, 360.523355, 362.380629, 364.211614, 366.021628,
    367.799990, 369.534506, 371.212939, 372.827278, 374.378076, 375.876689, 377.318612, 378.703502,
    380.041323, 381.352418, 382.631047, 383.868881, 385.057598, 386.187686, 387.250128, 388.240377,
    389.158627, 390.020519, 390.839558, 391.624727, 392.380889, 393.112658, 393.824654, 394.521255,
    395.206382, 395.887916, 396.565631, 397.238271, 397.899189, 398.536732, 399.149249, 399.736031,
    400.296376, 400.830875, 401.342973, 401.841069, 402.328026, 402.809373, 403.284130, 403.746093,
    404.196160, 404.636036, 405.067433, 405.491952, 405.911073, 406.326717, 406.739665, 407.148398,
    407.549935, 407.944227, 408.331567, 408.712224, 409.086468, 409.454624, 409.817076, 410.174132,
];

/// Surface mass balance of an ice sheet.
///
/// Each year adds `sensitivity * (linear_fraction * T + (1 - linear_fraction) * max(T, 0)^exponent)`,
/// reduced by `max(1 - S / volume, 0)^volume_exponent` when `volume` is set.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceMassBalanceParameters {
    /// unit: mm/yr
    pub sensitivity: FloatValue,
    /// Weight of the linear temperature term
    pub linear_fraction: FloatValue,
    /// Exponent of the warming term
    pub exponent: FloatValue,
    /// First year of surface melt
    pub start_year: FloatValue,
    /// Sea level equivalent of the ice sheet, `None` for no depletion
    /// unit: mm
    pub volume: Option<FloatValue>,
    /// Exponent of the remaining volume fraction
    pub volume_exponent: FloatValue,
}

impl SurfaceMassBalanceParameters {
    /// MAGICC7 default Greenland parameters.
    pub fn greenland() -> Self {
        Self {
            sensitivity: 0.1148748,
            linear_fraction: 1.161776e-7,
            exponent: 2.016917,
            start_year: 1965.0,
            volume: Some(7360.0),
            volume_exponent: 0.5,
        }
    }

    /// MAGICC7 default Antarctic parameters.
    pub fn antarctica() -> Self {
        Self {
            sensitivity: -0.1208204,
            linear_fraction: -7.702955e-15,
            exponent: 1.523378,
            start_year: 1980.0,
            volume: None,
            volume_exponent: 0.5,
        }
    }
}

impl Default for SurfaceMassBalanceParameters {
    fn default() -> Self {
        Self::greenland()
    }
}

/// One discharge case of the Greenland ice sheet.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DischargeCaseParameters {
    /// Ice available for discharge
    /// unit: mm
    pub volume: FloatValue,
    /// Fraction of the remaining volume discharged per year at zero warming
    /// unit: 1/yr
    pub sensitivity: FloatValue,
    /// unit: 1/K
    pub exponent: FloatValue,
}

impl DischargeCaseParameters {
    /// MAGICC7 default low case.
    pub fn low() -> Self {
        Self {
            volume: 35.9803,
            sensitivity: 9.0617e-4,
            exponent: 0.38919,
        }
    }

    /// MAGICC7 default high case.
    pub fn high() -> Self {
        Self {
            volume: 53.6289,
            sensitivity: 7.933e-4,
            exponent: 0.47216,
        }
    }
}

impl Default for DischargeCaseParameters {
    fn default() -> Self {
        Self::low()
    }
}

fn low_over_defaults<'de, D: Deserializer<'de>>(d: D) -> Result<DischargeCaseParameters, D::Error> {
    over_defaults(d, DischargeCaseParameters::low())
}

fn high_over_defaults<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<DischargeCaseParameters, D::Error> {
    over_defaults(d, DischargeCaseParameters::high())
}

/// Solid ice discharge of the Greenland ice sheet, interpolated between a low and a high case.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GreenlandDischargeParameters {
    #[serde(deserialize_with = "low_over_defaults")]
    pub low: DischargeCaseParameters,
    #[serde(deserialize_with = "high_over_defaults")]
    pub high: DischargeCaseParameters,
    /// Position between the low (0) and high (1) case
    pub case: FloatValue,
    pub scaling: FloatValue,
    /// First year of discharge
    pub start_year: FloatValue,
}

impl Default for GreenlandDischargeParameters {
    fn default() -> Self {
        Self {
            low: DischargeCaseParameters::low(),
            high: DischargeCaseParameters::high(),
            case: 0.5,
            scaling: 5.0,
            start_year: 2000.0,
        }
    }
}

/// One Antarctic drainage region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntarcticRegionParameters {
    pub name: String,
    /// Years after `start_year` before the region responds
    /// unit: yr
    pub delay: usize,
    /// Ratio of subsurface ocean warming to global warming
    pub temperature_scaling: FloatValue,
    /// Quartic response to a unit basal melt pulse, highest power first
    /// unit: m per m/yr
    pub response: [FloatValue; 5],
}

/// Solid ice discharge of the Antarctic ice sheet (Levermann et al. 2014).
///
/// Basal melt from subsurface ocean warming is convolved with each region's
/// response function.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AntarcticDischargeParameters {
    /// Basal melt per K of subsurface ocean warming
    /// unit: m/yr/K
    pub basal_melt_sensitivity: FloatValue,
    /// Years after which the response functions are zero
    /// unit: yr
    pub response_span: usize,
    pub scaling: FloatValue,
    /// First year of discharge
    pub start_year: FloatValue,
    pub regions: Vec<AntarcticRegionParameters>,
}

impl Default for AntarcticDischargeParameters {
    fn default() -> Self {
        let region = |name: &str, delay, temperature_scaling, response| AntarcticRegionParameters {
            name: name.into(),
            delay,
            temperature_scaling,
            response,
        };
        Self {
            basal_melt_sensitivity: 11.5,
            response_span: 500,
            scaling: 1.0,
            start_year: 1850.0,
            regions: vec![
                region(
                    "Amundsen",
                    60,
                    0.35,
                    [
                        3.80221625e-15,
                        -1.20779839e-11,
                        5.34616000e-09,
                        -1.12359405e-07,
                        2.69854318e-05,
                    ],
                ),
                region(
                    "East Antarctica",
                    50,
                    0.24,
                    [
                        -4.80672912e-15,
                        4.68446533e-12,
                        -1.32757538e-09,
                        1.64696643e-07,
                        1.14405405e-05,
                    ],
                ),
                region(
                    "Ross",
                    0,
                    0.23,
                    [
                        -6.09433726e-14,
                        5.15605867e-11,
                        -1.36480364e-08,
                        1.75347964e-06,
                        -2.21886314e-05,
                    ],
                ),
                region(
                    "Weddell",
                    65,
                    0.18,
                    [
                        1.54097171e-14,
                        -1.53064596e-11,
                        5.22078742e-09,
                        -5.09486184e-07,
                        3.54817378e-05,
                    ],
                ),
            ],
        }
    }
}

fn greenland_over_defaults<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<SurfaceMassBalanceParameters, D::Error> {
    over_defaults(d, SurfaceMassBalanceParameters::greenland())
}

fn antarctica_over_defaults<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<SurfaceMassBalanceParameters, D::Error> {
    over_defaults(d, SurfaceMassBalanceParameters::antarctica())
}

/// Parameters for the sea level rise contributions other than thermal expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SeaLevelRiseParameters {
    pub glaciers: GlacierParameters,
    /// Greenland surface mass balance, MAGICC7: SLR_GIS_SMB_*
    #[serde(deserialize_with = "greenland_over_defaults")]
    pub greenland_surface_mass_balance: SurfaceMassBalanceParameters,
    pub greenland_discharge: GreenlandDischargeParameters,
    /// Antarctic surface mass balance, MAGICC7: SLR_AIS_SMB_*
    #[serde(deserialize_with = "antarctica_over_defaults")]
    pub antarctic_surface_mass_balance: SurfaceMassBalanceParameters,
    pub antarctic_discharge: AntarcticDischargeParameters,
}

impl Default for SeaLevelRiseParameters {
    fn default() -> Self {
        Self {
            glaciers: GlacierParameters::default(),
            greenland_surface_mass_balance: SurfaceMassBalanceParameters::greenland(),
            greenland_discharge: GreenlandDischargeParameters::default(),
            antarctic_surface_mass_balance: SurfaceMassBalanceParameters::antarctica(),
            antarctic_discharge: AntarcticDischargeParameters::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_overrides_keep_their_own_defaults() {
        let p: SeaLevelRiseParameters = serde_json::from_str(
            r#"{"antarctic_surface_mass_balance": {"start_year": 1990}, "greenland_discharge": {"high": {"volume": 50}}}"#,
        )
        .unwrap();
        assert_eq!(p.antarctic_surface_mass_balance.start_year, 1990.0);
        assert_eq!(p.antarctic_surface_mass_balance.sensitivity, -0.1208204);
        assert_eq!(p.antarctic_surface_mass_balance.volume, None);
        assert_eq!(p.greenland_discharge.high.volume, 50.0);
        assert_eq!(p.greenland_discharge.high.exponent, 0.47216);
        assert_eq!(p.greenland_discharge.low.volume, 35.9803);
    }
}
