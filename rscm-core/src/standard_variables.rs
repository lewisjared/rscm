//! Standard variable definitions for RSCM.
//!
//! This module defines the standard variables used in reduced-complexity climate models.
//! These variables are registered at compile-time using the [`define_static_variable!`] macro
//! and are automatically available in the global [`VARIABLE_REGISTRY`].
//!
//! # Variable Naming Conventions
//!
//! Variable names follow the MAGICC convention using `|` as a hierarchical separator:
//! - `Emissions|CO2` - CO2 emissions
//! - `Atmospheric Concentration|CO2` - Atmospheric CO2 concentration
//! - `Effective Radiative Forcing|CO2` - CO2 effective radiative forcing
//!
//! # Usage
//!
//! Variables can be accessed via the registry:
//!
//! ```rust
//! use rscm_core::variable::VARIABLE_REGISTRY;
//! use rscm_core::standard_variables::VAR_CO2_EMISSIONS;
//!
//! // Use the static constant directly
//! assert_eq!(VAR_CO2_EMISSIONS.name, "Emissions|CO2");
//!
//! // Or look up in the registry
//! let var = VARIABLE_REGISTRY.get_with_static("Emissions|CO2").unwrap();
//! assert_eq!(var.unit, "GtC / yr");
//! ```
//!
//! # Available Variables
//!
//! ## Emissions
//! - `VAR_CO2_EMISSIONS` - CO2 emissions in GtC / yr
//! - `VAR_CH4_EMISSIONS` - CH4 emissions in MtCH4 / yr
//! - `VAR_N2O_EMISSIONS` - N2O emissions in MtN2O / yr
//!
//! ## Concentrations
//! - `VAR_CO2_CONCENTRATION` - Atmospheric CO2 concentration in ppm
//! - `VAR_CH4_CONCENTRATION` - Atmospheric CH4 concentration in ppb
//! - `VAR_N2O_CONCENTRATION` - Atmospheric N2O concentration in ppb
//!
//! ## Radiative Forcing
//! - `VAR_CO2_ERF` - CO2 effective radiative forcing in W / m^2
//! - `VAR_CH4_ERF` - CH4 effective radiative forcing in W / m^2
//! - `VAR_N2O_ERF` - N2O effective radiative forcing in W / m^2
//! - `VAR_TOTAL_ERF` - Total effective radiative forcing in W / m^2
//!
//! ## Temperature
//! - `VAR_GLOBAL_TEMPERATURE` - Global mean surface temperature in K
//! - `VAR_OCEAN_HEAT_UPTAKE` - Ocean heat uptake in W / m^2

use crate::define_static_variable;
use crate::variable::TimeConvention;

// ============================================================================
// Emissions Variables
// ============================================================================

define_static_variable!(
    VAR_CO2_EMISSIONS,
    name = "Emissions|CO2",
    unit = "GtC / yr",
    time_convention = TimeConvention::MidYear,
    description = "Carbon dioxide emissions from fossil fuels, industry, and land use change",
);

define_static_variable!(
    VAR_CH4_EMISSIONS,
    name = "Emissions|CH4",
    unit = "MtCH4 / yr",
    time_convention = TimeConvention::MidYear,
    description = "Methane emissions from all sources",
);

define_static_variable!(
    VAR_N2O_EMISSIONS,
    name = "Emissions|N2O",
    unit = "MtN2O / yr",
    time_convention = TimeConvention::MidYear,
    description = "Nitrous oxide emissions from all sources",
);

// ============================================================================
// Concentration Variables
// ============================================================================

define_static_variable!(
    VAR_CO2_CONCENTRATION,
    name = "Atmospheric Concentration|CO2",
    unit = "ppm",
    time_convention = TimeConvention::StartOfYear,
    description = "Atmospheric carbon dioxide concentration",
);

define_static_variable!(
    VAR_CH4_CONCENTRATION,
    name = "Atmospheric Concentration|CH4",
    unit = "ppb",
    time_convention = TimeConvention::StartOfYear,
    description = "Atmospheric methane concentration",
);

define_static_variable!(
    VAR_N2O_CONCENTRATION,
    name = "Atmospheric Concentration|N2O",
    unit = "ppb",
    time_convention = TimeConvention::StartOfYear,
    description = "Atmospheric nitrous oxide concentration",
);

// ============================================================================
// Effective Radiative Forcing Variables
// ============================================================================

define_static_variable!(
    VAR_CO2_ERF,
    name = "Effective Radiative Forcing|CO2",
    unit = "W / m^2",
    time_convention = TimeConvention::Instantaneous,
    description = "Effective radiative forcing from carbon dioxide",
);

define_static_variable!(
    VAR_CH4_ERF,
    name = "Effective Radiative Forcing|CH4",
    unit = "W / m^2",
    time_convention = TimeConvention::Instantaneous,
    description = "Effective radiative forcing from methane",
);

define_static_variable!(
    VAR_N2O_ERF,
    name = "Effective Radiative Forcing|N2O",
    unit = "W / m^2",
    time_convention = TimeConvention::Instantaneous,
    description = "Effective radiative forcing from nitrous oxide",
);

define_static_variable!(
    VAR_TOTAL_ERF,
    name = "Effective Radiative Forcing|Total",
    unit = "W / m^2",
    time_convention = TimeConvention::Instantaneous,
    description = "Total effective radiative forcing from all sources",
);

// ============================================================================
// Temperature Variables
// ============================================================================

define_static_variable!(
    VAR_GLOBAL_TEMPERATURE,
    name = "Surface Temperature|Global",
    unit = "K",
    time_convention = TimeConvention::StartOfYear,
    description = "Global mean surface temperature anomaly relative to preindustrial",
);

define_static_variable!(
    VAR_OCEAN_HEAT_UPTAKE,
    name = "Ocean Heat Uptake",
    unit = "W / m^2",
    time_convention = TimeConvention::Instantaneous,
    description = "Net heat flux into the ocean",
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::VARIABLE_REGISTRY;

    #[test]
    fn test_standard_variables_registered() {
        // Check that static variables are accessible
        assert_eq!(VAR_CO2_EMISSIONS.name, "Emissions|CO2");
        assert_eq!(VAR_CO2_EMISSIONS.unit, "GtC / yr");
        assert_eq!(VAR_CO2_EMISSIONS.time_convention, TimeConvention::MidYear);

        assert_eq!(VAR_CO2_CONCENTRATION.name, "Atmospheric Concentration|CO2");
        assert_eq!(
            VAR_CO2_CONCENTRATION.time_convention,
            TimeConvention::StartOfYear
        );

        assert_eq!(VAR_CO2_ERF.name, "Effective Radiative Forcing|CO2");
        assert_eq!(VAR_CO2_ERF.time_convention, TimeConvention::Instantaneous);
    }

    #[test]
    fn test_standard_variables_in_registry() {
        // Check that variables are registered in the global registry
        let var = VARIABLE_REGISTRY
            .get_with_static("Emissions|CO2")
            .expect("CO2 emissions should be registered");
        assert_eq!(var.unit, "GtC / yr");

        let var = VARIABLE_REGISTRY
            .get_with_static("Atmospheric Concentration|CO2")
            .expect("CO2 concentration should be registered");
        assert_eq!(var.unit, "ppm");

        let var = VARIABLE_REGISTRY
            .get_with_static("Effective Radiative Forcing|CO2")
            .expect("CO2 ERF should be registered");
        assert_eq!(var.unit, "W / m^2");
    }

    #[test]
    fn test_all_standard_variables_registered() {
        let expected_vars = [
            "Emissions|CO2",
            "Emissions|CH4",
            "Emissions|N2O",
            "Atmospheric Concentration|CO2",
            "Atmospheric Concentration|CH4",
            "Atmospheric Concentration|N2O",
            "Effective Radiative Forcing|CO2",
            "Effective Radiative Forcing|CH4",
            "Effective Radiative Forcing|N2O",
            "Effective Radiative Forcing|Total",
            "Surface Temperature|Global",
            "Ocean Heat Uptake",
        ];

        for name in expected_vars {
            assert!(
                VARIABLE_REGISTRY.get_with_static(name).is_some(),
                "Variable '{}' should be registered",
                name
            );
        }
    }

    #[test]
    fn test_emissions_are_mid_year() {
        // All emissions should use mid-year convention
        for name in ["Emissions|CO2", "Emissions|CH4", "Emissions|N2O"] {
            let var = VARIABLE_REGISTRY.get_with_static(name).unwrap();
            assert_eq!(
                var.time_convention,
                TimeConvention::MidYear,
                "{} should use MidYear convention",
                name
            );
        }
    }

    #[test]
    fn test_concentrations_are_start_of_year() {
        // All concentrations should use start-of-year convention
        for name in [
            "Atmospheric Concentration|CO2",
            "Atmospheric Concentration|CH4",
            "Atmospheric Concentration|N2O",
        ] {
            let var = VARIABLE_REGISTRY.get_with_static(name).unwrap();
            assert_eq!(
                var.time_convention,
                TimeConvention::StartOfYear,
                "{} should use StartOfYear convention",
                name
            );
        }
    }

    #[test]
    fn test_forcing_are_instantaneous() {
        // All radiative forcing should use instantaneous convention
        for name in [
            "Effective Radiative Forcing|CO2",
            "Effective Radiative Forcing|CH4",
            "Effective Radiative Forcing|N2O",
            "Effective Radiative Forcing|Total",
        ] {
            let var = VARIABLE_REGISTRY.get_with_static(name).unwrap();
            assert_eq!(
                var.time_convention,
                TimeConvention::Instantaneous,
                "{} should use Instantaneous convention",
                name
            );
        }
    }
}
