//! Nitrate direct and cloud albedo forcing
//!
//! Both follow MAGICC7 v7.5.3 and are closed-form in prescribed emissions and optical thicknesses.
//! They read those inputs in reference years before and after the step,
//! so every input must be prescribed.
//!
//! # Nitrate
//!
//! The nitrate burden follows NOx, NH3 and SOx emissions (Hauglustaine et al. 2014):
//!
//! $$\alpha = \frac{N A}{S \lambda / A + 1}, \quad B = B_{pi} \alpha / \alpha_0$$
//!
//! where $N$, $A$ and $S$ are total NOx, NH3 and SOx emissions in moles,
//! and $\alpha_0$ is $\alpha$ in the first year.
//! The forcing is $\tau_{pi} (B / B_{pi})^e - \tau_{pi}$, times a factor.
//! Each hemisphere takes its share of the NOx x NH3 product, spread over its boxes by a pattern,
//! and splits between industrial and biomass sources by the same product.
//!
//! # Cloud albedo
//!
//! An aerosol number index weights the optical thickness of each cloud-active species,
//! normalised by its area mean in its switch year.
//! After that year each emitted species follows its emissions,
//! as each year's emissions plus the next year's.
//! Nitrate takes the place of an optical thickness the same way.
//! The forcing is the change in log10 of the index since the first year,
//! scaled so that the boxes equal `regional_pattern` in `pattern_year`, times a factor.
//! In the final year of the run the emitted species repeat the year before, as in MAGICC7.

use crate::parameters::{ForcingTarget, NitrateCloudForcingParameters};
use rscm_core::component::{
    Component, ComponentState, InputState, OutputWriter, VariableDefinition,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::names::{
    child, EMISSIONS_BC, EMISSIONS_NH3, EMISSIONS_NOX, EMISSIONS_OC, EMISSIONS_SOX,
    ERF_AEROSOLS_DIRECT_EFFECT, ERF_AEROSOLS_INDIRECT_EFFECT, OPTICAL_THICKNESS,
};
use rscm_core::spatial::FourBoxRegion;
use rscm_core::state::{FourBoxSlice, VariableSource};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};

use crate::reference::require_annual;

type Boxes = [FloatValue; 4];

/// Emissions at a time, by variable name.
pub type EmissionsLookup<'a> = dyn Fn(&str, Time) -> RSCMResult<FloatValue> + 'a;
/// Four-box optical thickness at a time, by variable name.
pub type OpticalThicknessLookup<'a> = dyn Fn(&str, Time) -> RSCMResult<Boxes> + 'a;

/// Prescribed inputs over a run with annual steps from `first` to `last`.
pub struct Drivers<'a> {
    /// Emissions in MAGICC7's units
    pub emissions: &'a EmissionsLookup<'a>,
    /// Four-box optical thickness of each cloud-active species
    pub optical_thickness: &'a OpticalThicknessLookup<'a>,
    /// First year of the run
    pub first: Time,
    /// Final year of the run
    pub last: Time,
}

/// Nitrate and cloud albedo forcing in a year.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NitrateCloud {
    /// Nitrate forcing from industrial sources
    pub nitrate_fossil: Boxes,
    /// Nitrate forcing from biomass burning
    pub nitrate_biosphere: Boxes,
    /// Cloud albedo forcing
    pub cloud_albedo: Boxes,
}

const SECTORS: [&str; 2] = ["Fossil", "Biosphere"];
const REGIONS: [FourBoxRegion; 4] = [
    FourBoxRegion::NorthernOcean,
    FourBoxRegion::NorthernLand,
    FourBoxRegion::SouthernOcean,
    FourBoxRegion::SouthernLand,
];

/// A cloud-active component: its optical thickness segment and the emissions that extend it.
struct CloudComponent {
    species: &'static str,
    emissions: Option<&'static str>,
}

const fn emitted(species: &'static str, emissions: &'static str) -> CloudComponent {
    CloudComponent {
        species,
        emissions: Some(emissions),
    }
}

const fn held(species: &'static str) -> CloudComponent {
    CloudComponent {
        species,
        emissions: None,
    }
}

const BC_GROUP: [CloudComponent; 2] = [
    emitted("BC|Fossil", "Emissions|BC|Fossil"),
    emitted("BC|Biosphere", "Emissions|BC|Biosphere"),
];
const OC_GROUP: [CloudComponent; 3] = [
    emitted("OC|Fossil", "Emissions|OC|Fossil"),
    emitted("OC|Biosphere", "Emissions|OC|Biosphere"),
    held("OC|Natural"),
];
const SOX_GROUP: [CloudComponent; 2] = [
    emitted("SOx|Fossil", "Emissions|SOx|Fossil"),
    emitted("SOx|Biosphere", "Emissions|SOx|Biosphere"),
];
const SEA_SALT_GROUP: [CloudComponent; 1] = [held("Sea Salt")];
const SOX_BIOSPHERE: &str = "SOx|Biosphere";
const BC_FOSSIL: &str = "BC|Fossil";

/// Name of the four-box optical thickness of a cloud-active species,
/// as in `Optical Thickness|SOx|Fossil`.
pub fn optical_thickness_name(species: &str) -> String {
    child(OPTICAL_THICKNESS, species)
}

fn optical_thickness_species() -> impl Iterator<Item = &'static str> {
    BC_GROUP
        .iter()
        .chain(&OC_GROUP)
        .chain(&SOX_GROUP)
        .chain(&SEA_SALT_GROUP)
        .map(|c| c.species)
}

fn emissions_names() -> Vec<String> {
    [
        EMISSIONS_NOX,
        EMISSIONS_NH3,
        EMISSIONS_SOX,
        EMISSIONS_BC,
        EMISSIONS_OC,
    ]
    .iter()
    .flat_map(|gas| SECTORS.map(|sector| child(gas, sector)))
    .collect()
}

fn emissions_unit(name: &str) -> &'static str {
    match name.split('|').nth(1) {
        Some("NOx" | "NH3") => "Mt N/yr",
        Some("SOx") => "Mt S/yr",
        Some("BC") => "Mt BC/yr",
        _ => "Mt OC/yr",
    }
}

fn nitrate_name(sector: &str) -> String {
    child(&child(ERF_AEROSOLS_DIRECT_EFFECT, "NO3"), sector)
}

/// Nitrate direct and cloud albedo forcing from emissions and optical thicknesses.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["forcing", "aerosol", "cloud", "magicc"], category = "Radiative Forcing")]
pub struct NitrateCloudForcing {
    parameters: NitrateCloudForcingParameters,
}

impl NitrateCloudForcing {
    /// Create the component, checking the parameters.
    pub fn from_parameters(parameters: NitrateCloudForcingParameters) -> RSCMResult<Self> {
        let check = |ok: bool, msg: &str| {
            if ok {
                Ok(())
            } else {
                Err(RSCMError::Error(format!("NitrateCloudForcing: {msg}")))
            }
        };
        let p = &parameters;
        check(
            p.nitrate.harmonisation.is_none() || p.nitrate.scale == 1.0,
            "nitrate takes harmonisation or scale, not both",
        )?;
        check(
            p.cloud_albedo.harmonisation.is_none() || p.cloud_albedo.scale == 1.0,
            "cloud_albedo takes harmonisation or scale, not both",
        )?;
        Ok(Self { parameters })
    }

    /// The parameters
    pub fn parameters(&self) -> &NitrateCloudForcingParameters {
        &self.parameters
    }

    /// Nitrate and cloud albedo forcing in `year`.
    pub fn forcing(&self, drivers: &Drivers, year: Time) -> RSCMResult<NitrateCloud> {
        let c = Calc::new(&self.parameters, drivers);
        let (nitrate_fossil, nitrate_biosphere) = c.nitrate_sources(year)?;
        Ok(NitrateCloud {
            nitrate_fossil,
            nitrate_biosphere,
            cloud_albedo: c.cloud_albedo(year)?,
        })
    }

    /// Box nitrate forcing per unit global nitrate forcing in `year` (MAGICC7 `RF_REGIONS_NO3`).
    ///
    /// MAGICC7 overwrites the configured pattern with this before its run,
    /// and uses it for nitrate's internal efficacy.
    pub fn nitrate_pattern(&self, drivers: &Drivers, year: Time) -> RSCMResult<Boxes> {
        let c = Calc::new(&self.parameters, drivers);
        let (rf, total) = c.nitrate_total(c.at(year), c.nitrate_factor(true)?)?;
        Ok(total.map(|b| b / rf))
    }
}

struct Calc<'a> {
    p: &'a NitrateCloudForcingParameters,
    d: &'a Drivers<'a>,
    area: Boxes,
}

fn area_fractions(nh_land: FloatValue, sh_land: FloatValue) -> Boxes {
    [
        (1.0 - nh_land) / 2.0,
        nh_land / 2.0,
        (1.0 - sh_land) / 2.0,
        sh_land / 2.0,
    ]
}

fn add(a: Boxes, b: Boxes) -> Boxes {
    std::array::from_fn(|i| a[i] + b[i])
}

impl<'a> Calc<'a> {
    fn new(p: &'a NitrateCloudForcingParameters, d: &'a Drivers<'a>) -> Self {
        Self {
            p,
            d,
            area: area_fractions(p.nh_land_fraction, p.sh_land_fraction),
        }
    }

    /// `year` moved onto the run's years, as MAGICC7 does for reference years.
    fn at(&self, year: Time) -> Time {
        let steps = (self.d.last - self.d.first).round();
        self.d.first + (year - self.d.first).round().clamp(0.0, steps)
    }

    fn area_mean(&self, b: Boxes) -> FloatValue {
        let a = self.area;
        b[0] * a[0] + b[1] * a[1] + b[2] * a[2] + b[3] * a[3]
    }

    fn e(&self, name: &str, year: Time) -> RSCMResult<FloatValue> {
        (self.d.emissions)(name, year)
    }

    fn sectors(&self, gas: &str, year: Time) -> RSCMResult<[FloatValue; 2]> {
        Ok([
            self.e(&child(gas, "Fossil"), year)?,
            self.e(&child(gas, "Biosphere"), year)?,
        ])
    }

    fn alpha(&self, year: Time) -> RSCMResult<FloatValue> {
        let p = &self.p.nitrate;
        let [nox_i, nox_b] = self.sectors(EMISSIONS_NOX, year)?;
        let [nh3_i, nh3_b] = self.sectors(EMISSIONS_NH3, year)?;
        let [sox_i, sox_b] = self.sectors(EMISSIONS_SOX, year)?;
        let n = ((nox_i + nox_b) + p.natural_nox_emissions) / 14.0;
        let a = ((nh3_i + nh3_b) + p.natural_nh3_emissions) / 14.0;
        let s = ((sox_i + sox_b) + self.p.natural_sox_emissions) / 32.07;
        Ok((n * a) / ((s * p.lambda_so2) / a + 1.0))
    }

    /// Global nitrate forcing before the factor.
    fn nitrate_unscaled(&self, year: Time) -> RSCMResult<FloatValue> {
        let p = &self.p.nitrate;
        let burden = (p.burden_pi * self.alpha(year)?) / self.alpha(self.d.first)?;
        let optical_thickness = (burden / p.burden_pi).powf(p.burden_to_optical_thickness_exponent);
        Ok(optical_thickness * p.optical_thickness_pi - p.optical_thickness_pi)
    }

    /// Nitrate factor, which is one where it is zero and only the shape matters.
    fn nitrate_factor(&self, shape_only: bool) -> RSCMResult<FloatValue> {
        let p = &self.p.nitrate;
        let factor = match &p.harmonisation {
            Some(h) => h.target / self.nitrate_unscaled(self.at(h.reference_year))?,
            None => p.scale,
        };
        Ok(if shape_only && factor == 0.0 {
            1.0
        } else {
            factor
        })
    }

    /// Global nitrate forcing and its boxes, before the source split.
    fn nitrate_total(&self, year: Time, factor: FloatValue) -> RSCMResult<(FloatValue, Boxes)> {
        let rf = self.nitrate_unscaled(year)? * factor;
        let [nox_i, nox_b] = self.sectors(EMISSIONS_NOX, year)?;
        let [nh3_i, nh3_b] = self.sectors(EMISSIONS_NH3, year)?;
        let product = (nh3_i * 0.5 + nh3_b * 0.5) * (nox_b * 0.5 + nox_i * 0.5);
        let denominator = product + product;
        // MAGICC7 sets the forcing to zero without anthropogenic NOx x NH3.
        let hemisphere = if denominator != 0.0 {
            ((rf + rf) * product) / denominator
        } else {
            0.0
        };
        let w = self.p.nitrate.regional_pattern;
        let mut total = [0.0; 4];
        for h in 0..2 {
            let land = self.area[2 * h + 1] * 2.0;
            let sea = 1.0 - land;
            let (wo, wl) = (w[2 * h], w[2 * h + 1]);
            total[2 * h] = ((hemisphere * (sea + land)) * wo) / (sea * wo + land * wl);
            total[2 * h + 1] = ((hemisphere * (land + sea)) * wl) / (land * wl + sea * wo);
        }
        Ok((rf, total))
    }

    fn nitrate_sources(&self, year: Time) -> RSCMResult<(Boxes, Boxes)> {
        let (_, total) = self.nitrate_total(year, self.nitrate_factor(false)?)?;
        let [nox_i, nox_b] = self.sectors(EMISSIONS_NOX, year)?;
        let [nh3_i, nh3_b] = self.sectors(EMISSIONS_NH3, year)?;
        let industrial = (nh3_i * 0.5) * (nox_i * 0.5);
        let biomass = (nh3_b * 0.5) * (nox_b * 0.5);
        let share = biomass / (industrial + biomass);
        let share = if share.is_nan() { 0.5 } else { share };
        Ok((total.map(|t| t * (1.0 - share)), total.map(|t| t * share)))
    }

    /// Emissions of `year` plus the next year's.
    fn drive(&self, name: &str, natural: Option<FloatValue>, year: Time) -> RSCMResult<FloatValue> {
        let (now, next) = (self.e(name, year)?, self.e(name, year + 1.0)?);
        Ok(match natural {
            None => next + now,
            Some(n) => ((now + n) + next) + n,
        })
    }

    /// MAGICC7 orders the natural terms differently for the drive in the switch year.
    fn base(&self, name: &str, natural: Option<FloatValue>, year: Time) -> RSCMResult<FloatValue> {
        let start = year.min(self.d.last - 1.0);
        match natural {
            None => self.drive(name, None, start),
            Some(n) => Ok(((self.e(name, start + 1.0)? + self.e(name, start)?) + n) + n),
        }
    }

    /// `value` until `switch`, then its value in `switch` scaled by the emission drive.
    fn extended(
        &self,
        year: Time,
        switch: Time,
        value: &dyn Fn(Time) -> RSCMResult<Boxes>,
        emissions: &str,
        natural: Option<FloatValue>,
    ) -> RSCMResult<Boxes> {
        if year >= self.d.last && self.d.last > self.d.first {
            return self.extended(self.d.last - 1.0, switch, value, emissions, natural);
        }
        if year < switch {
            return value(year);
        }
        let at_switch = value(switch)?;
        let drive = self.drive(emissions, natural, year)?;
        let base = self.base(emissions, natural, switch)?;
        Ok(at_switch.map(|v| (drive * v) / base))
    }

    fn optical_thickness(&self, species: &str, year: Time) -> RSCMResult<Boxes> {
        let ot = (self.d.optical_thickness)(&optical_thickness_name(species), year)?;
        Ok(if species == BC_FOSSIL {
            ot.map(|v| v * self.p.cloud_albedo.bci_soluble_ratio)
        } else {
            ot
        })
    }

    /// Each component of a group in `year`, summed in order.
    fn group(&self, components: &[CloudComponent], switch: Time, year: Time) -> RSCMResult<Boxes> {
        let switch = self.at(switch);
        let mut at_switch = [0.0; 4];
        for c in components {
            at_switch = add(at_switch, self.optical_thickness(c.species, switch)?);
        }
        let norm = self.area_mean(at_switch);
        let mut sum = [0.0; 4];
        for c in components {
            let scaled = |y: Time| Ok(self.optical_thickness(c.species, y)?.map(|v| v / norm));
            let value = match c.emissions {
                Some(name) => {
                    let natural =
                        (c.species == SOX_BIOSPHERE).then_some(self.p.natural_sox_emissions);
                    self.extended(year, switch, &scaled, name, natural)?
                }
                None => scaled(year)?,
            };
            sum = add(sum, value);
        }
        Ok(sum)
    }

    fn nitrate_index(&self, year: Time, factor: FloatValue) -> RSCMResult<Boxes> {
        let switch = self.at(self.p.cloud_albedo.switch_years.nitrate);
        let norm = self.area_mean(self.nitrate_total(switch, factor)?.1);
        let value = |y: Time| Ok(self.nitrate_total(y, factor)?.1.map(|v| v / norm));
        self.extended(year, switch, &value, &child(EMISSIONS_NOX, "Fossil"), None)
    }

    /// Aerosol number index, with nitrate when `nitrate` is given.
    fn number_index(&self, year: Time, nitrate: Option<Boxes>) -> RSCMResult<Boxes> {
        let c = &self.p.cloud_albedo;
        let (w, s) = (&c.weights, &c.switch_years);
        let total = (((w.nitrate + w.bc) + w.oc) + w.sox) + w.sea_salt;
        let mut out = nitrate.unwrap_or([0.0; 4]).map(|v| v * (w.nitrate / total));
        let groups: [(&[CloudComponent], FloatValue, FloatValue); 4] = [
            (&BC_GROUP, w.bc, s.bc),
            (&OC_GROUP, w.oc, s.oc),
            (&SOX_GROUP, w.sox, s.sox),
            (&SEA_SALT_GROUP, w.sea_salt, s.sea_salt),
        ];
        for (components, weight, switch) in groups {
            let sum = self.group(components, switch, year)?;
            out = std::array::from_fn(|i| out[i] + sum[i] * (weight / total));
        }
        Ok(out)
    }

    fn cdnc(&self, year: Time, reference: &Boxes, factor: FloatValue) -> RSCMResult<Boxes> {
        let nitrate = self.nitrate_index(year, factor)?;
        let index = self.number_index(year, Some(nitrate))?;
        Ok(std::array::from_fn(|i| {
            index[i].log10() - reference[i].log10()
        }))
    }

    fn cloud_albedo(&self, year: Time) -> RSCMResult<Boxes> {
        let c = &self.p.cloud_albedo;
        let reference = self.number_index(self.d.first, None)?;
        let factor = self.nitrate_factor(true)?;
        let at_pattern = self.cdnc(self.at(c.pattern_year), &reference, factor)?;
        let unscaled = |y: Time| -> RSCMResult<Boxes> {
            let cdnc = self.cdnc(y, &reference, factor)?;
            Ok(std::array::from_fn(|i| {
                (cdnc[i] / at_pattern[i]) * c.regional_pattern[i]
            }))
        };
        let scale = match &c.harmonisation {
            Some(ForcingTarget {
                target,
                reference_year,
            }) => target / self.area_mean(unscaled(self.at(*reference_year))?),
            None => c.scale,
        };
        Ok(unscaled(year)?.map(|v| v * scale))
    }
}

fn prescribed_only(name: &str, source: VariableSource) -> RSCMResult<()> {
    if source == VariableSource::Prescribed {
        return Ok(());
    }
    Err(RSCMError::Error(format!(
        "{name} must be prescribed, because NitrateCloudForcing reads it in other years"
    )))
}

#[typetag::serde(name = "magicc.NitrateCloudForcing")]
impl Component for NitrateCloudForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions: Vec<_> = emissions_names()
            .iter()
            .map(|name| VariableDefinition::scalar_input(name, emissions_unit(name)))
            .collect();
        definitions.extend(optical_thickness_species().map(|species| {
            VariableDefinition::four_box_input(&optical_thickness_name(species), "1")
        }));
        for sector in SECTORS {
            definitions.push(VariableDefinition::four_box_output(
                &nitrate_name(sector),
                "W/m^2",
            ));
        }
        definitions.push(VariableDefinition::four_box_output(
            ERF_AEROSOLS_INDIRECT_EFFECT,
            "W/m^2",
        ));
        definitions
    }

    fn solve_into(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        _internal_state: &mut dyn ComponentState,
        out: &mut OutputWriter,
    ) -> RSCMResult<()> {
        require_annual("NitrateCloudForcing", t_current, t_next)?;
        let emissions = |name: &str, t: Time| {
            let window = input_state.get_scalar_window(name);
            prescribed_only(name, window.variable_source())?;
            window.interpolate(t)
        };
        let optical_thickness = |name: &str, t: Time| -> RSCMResult<Boxes> {
            let window = input_state.get_four_box_window(name);
            prescribed_only(name, window.variable_source())?;
            let mut boxes = [0.0; 4];
            for (b, region) in boxes.iter_mut().zip(REGIONS) {
                *b = window.interpolate(t, region)?;
            }
            Ok(boxes)
        };
        let axis_window = input_state.get_scalar_window(EMISSIONS_NOX_FOSSIL);
        let years = axis_window.time_axis().values();
        let drivers = Drivers {
            emissions: &emissions,
            optical_thickness: &optical_thickness,
            first: years[0],
            last: years[years.len() - 1],
        };
        let forcing = self.forcing(&drivers, t_next)?;
        out.set(
            &nitrate_name("Fossil"),
            FourBoxSlice::from_array(forcing.nitrate_fossil),
        );
        out.set(
            &nitrate_name("Biosphere"),
            FourBoxSlice::from_array(forcing.nitrate_biosphere),
        );
        out.set(
            ERF_AEROSOLS_INDIRECT_EFFECT,
            FourBoxSlice::from_array(forcing.cloud_albedo),
        );
        Ok(())
    }
}

const EMISSIONS_NOX_FOSSIL: &str = "Emissions|NOx|Fossil";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::{CloudAlbedoParameters, NitrateParameters};

    const FIRST: Time = 2000.0;
    const LAST: Time = 2010.0;

    fn rising(name: &str, t: Time) -> RSCMResult<FloatValue> {
        let base = if name.contains("Biosphere") { 1.0 } else { 2.0 };
        Ok(base + 0.1 * (t - FIRST))
    }

    fn optical_thickness(name: &str, t: Time) -> RSCMResult<Boxes> {
        let base = if name.contains("Natural") || name.contains("Sea Salt") {
            0.5
        } else {
            1.0 + 0.05 * (t - FIRST).min(5.0)
        };
        Ok([base, base * 1.5, base * 0.5, base])
    }

    fn drivers<'a>(
        emissions: &'a EmissionsLookup<'a>,
        ot: &'a OpticalThicknessLookup<'a>,
    ) -> Drivers<'a> {
        Drivers {
            emissions,
            optical_thickness: ot,
            first: FIRST,
            last: LAST,
        }
    }

    fn parameters() -> NitrateCloudForcingParameters {
        NitrateCloudForcingParameters {
            nitrate: NitrateParameters {
                harmonisation: Some(ForcingTarget {
                    target: -0.05,
                    reference_year: 2008.0,
                }),
                ..Default::default()
            },
            cloud_albedo: CloudAlbedoParameters {
                switch_years: crate::parameters::CloudSpecies {
                    nitrate: 2004.0,
                    bc: 2005.0,
                    oc: 2005.0,
                    sox: 2005.0,
                    sea_salt: 2005.0,
                },
                pattern_year: 2003.0,
                harmonisation: Some(ForcingTarget {
                    target: -0.8,
                    reference_year: 2008.0,
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn component(p: NitrateCloudForcingParameters) -> NitrateCloudForcing {
        NitrateCloudForcing::from_parameters(p).unwrap()
    }

    #[test]
    fn nitrate_is_zero_in_the_first_year_and_meets_its_target() {
        let c = component(parameters());
        let d = drivers(&rising, &optical_thickness);
        let first = c.forcing(&d, FIRST).unwrap();
        assert_eq!(first.nitrate_fossil, [0.0; 4]);
        let reference = c.forcing(&d, 2008.0).unwrap();
        let total: Boxes = add(reference.nitrate_fossil, reference.nitrate_biosphere);
        let calc = Calc::new(c.parameters(), &d);
        assert!((calc.area_mean(total) - -0.05).abs() < 1e-12);
    }

    #[test]
    fn cloud_albedo_meets_its_target_and_pattern() {
        let c = component(parameters());
        let d = drivers(&rising, &optical_thickness);
        let calc = Calc::new(c.parameters(), &d);
        let at_target = c.forcing(&d, 2008.0).unwrap().cloud_albedo;
        assert!((calc.area_mean(at_target) - -0.8).abs() < 1e-12);
        let at_pattern = c.forcing(&d, 2003.0).unwrap().cloud_albedo;
        let ratio: Vec<FloatValue> = (0..4)
            .map(|i| at_pattern[i] / c.parameters().cloud_albedo.regional_pattern[i])
            .collect();
        assert!(ratio.iter().all(|r| (r - ratio[0]).abs() < 1e-12));
    }

    #[test]
    fn scale_replaces_harmonisation() {
        let mut p = parameters();
        p.cloud_albedo.harmonisation = None;
        p.cloud_albedo.scale = 2.0;
        let d = drivers(&rising, &optical_thickness);
        let doubled = component(p.clone())
            .forcing(&d, 2007.0)
            .unwrap()
            .cloud_albedo;
        p.cloud_albedo.scale = 1.0;
        let single = component(p).forcing(&d, 2007.0).unwrap().cloud_albedo;
        assert_eq!(doubled, single.map(|v| v * 2.0));
    }

    #[test]
    fn final_year_repeats_the_year_before() {
        let mut p = parameters();
        p.cloud_albedo.weights.sea_salt = 0.0;
        p.cloud_albedo.weights.oc = 0.0;
        let c = component(p);
        let d = drivers(&rising, &optical_thickness);
        let last = c.forcing(&d, LAST).unwrap().cloud_albedo;
        let before = c.forcing(&d, LAST - 1.0).unwrap().cloud_albedo;
        assert_eq!(last, before);
    }

    #[test]
    fn zero_anthropogenic_precursors_give_zero_nitrate() {
        let mut p = parameters();
        p.nitrate.harmonisation = None;
        let zero = |_: &str, _: Time| Ok(0.0);
        let d = drivers(&zero, &optical_thickness);
        let forcing = component(p).forcing(&d, 2005.0).unwrap();
        assert_eq!(forcing.nitrate_fossil, [0.0; 4]);
        assert_eq!(forcing.nitrate_biosphere, [0.0; 4]);
    }

    #[test]
    fn rejects_scale_with_harmonisation() {
        let mut p = parameters();
        p.nitrate.scale = 2.0;
        assert!(NitrateCloudForcing::from_parameters(p).is_err());
    }
}
