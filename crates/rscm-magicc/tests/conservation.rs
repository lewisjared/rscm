//! Conservation tests for MAGICC components.
//!
//! These tests verify that physical conservation laws are satisfied:
//! - Carbon mass conservation in the carbon cycle
//! - Energy conservation in the climate system

use approx::assert_relative_eq;
use rscm_magicc::carbon::{OceanCarbon, OceanCarbonState, TerrestrialCarbon};
use rscm_magicc::parameters::{OceanCarbonParameters, TerrestrialCarbonParameters};

mod carbon_cycle_conservation {
    use super::*;

    /// Test that terrestrial carbon pools conserve mass.
    ///
    /// The sum of all pool changes should equal:
    /// NPP - total respiration - land use emissions
    #[test]
    fn test_terrestrial_carbon_mass_balance() {
        let component = TerrestrialCarbon::from_parameters(TerrestrialCarbonParameters::default());
        let params = TerrestrialCarbonParameters::default();

        // Initial pool sizes
        let pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        let total_initial: f64 = pools.iter().sum();

        // Run for 10 years with elevated CO2 (net uptake scenario)
        let co2 = params.co2_pi * 1.5;
        let temperature = 0.0;
        let landuse = 0.0;
        let dt = 1.0;

        let mut current_pools = pools;
        let mut cumulative_flux = 0.0;

        for _ in 0..10 {
            let (new_pools, net_flux) =
                component.solve_pools(co2, temperature, landuse, current_pools, dt);

            cumulative_flux += net_flux * dt;
            current_pools = new_pools;
        }

        let total_final: f64 = current_pools.iter().sum();
        let pool_change = total_final - total_initial;

        // Pool change should approximately equal cumulative net flux
        // (positive flux = uptake from atmosphere = increase in pools)
        assert_relative_eq!(pool_change, cumulative_flux, epsilon = 1.0);
    }

    /// Test that terrestrial carbon pools remain non-negative.
    #[test]
    fn test_terrestrial_pools_non_negative() {
        let component = TerrestrialCarbon::from_parameters(TerrestrialCarbonParameters::default());
        let params = TerrestrialCarbonParameters::default();

        let pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        // Extreme scenario: high warming, high land use
        let co2 = params.co2_pi;
        let temperature = 10.0; // Extreme warming
        let landuse = 10.0; // High deforestation
        let dt = 1.0;

        let mut current_pools = pools;

        for _ in 0..50 {
            let (new_pools, _) =
                component.solve_pools(co2, temperature, landuse, current_pools, dt);

            for (i, &pool) in new_pools.iter().enumerate() {
                assert!(
                    pool >= 0.0,
                    "Pool {} went negative: {}",
                    ["Plant", "Detritus", "Soil", "Humus"][i],
                    pool
                );
            }

            current_pools = new_pools;
        }
    }

    /// Test that ocean carbon accumulates correctly.
    #[test]
    fn test_ocean_carbon_cumulative_uptake() {
        let component = OceanCarbon::from_parameters(OceanCarbonParameters::default());
        let mut state = OceanCarbonState::default();
        let params = OceanCarbonParameters::default();

        let co2_elevated = 400.0;
        let sst = 0.0;
        let mut pco2 = params.pco2_pi;
        let mut cumulative = 0.0;
        let dt = 1.0;

        let mut total_flux = 0.0;

        for _ in 0..50 {
            let (new_pco2, new_cumulative, flux) =
                component.solve_ocean(&mut state, co2_elevated, sst, pco2, cumulative, dt);

            total_flux += flux * dt;
            pco2 = new_pco2;
            cumulative = new_cumulative;
        }

        // Cumulative uptake should match integrated flux
        assert_relative_eq!(cumulative, total_flux, epsilon = 1.0);
    }

    /// Test that ocean pCO2 approaches atmospheric CO2 at equilibrium.
    #[test]
    fn test_ocean_pco2_approaches_equilibrium() {
        let component = OceanCarbon::from_parameters(OceanCarbonParameters::default());
        let mut state = OceanCarbonState::default();
        let params = OceanCarbonParameters::default();

        let co2_target = 400.0;
        let sst = 0.0;
        let mut pco2 = params.pco2_pi;
        let mut cumulative = 0.0;
        let dt = 1.0;

        // Run for a long time
        for _ in 0..500 {
            let (new_pco2, new_cumulative, _) =
                component.solve_ocean(&mut state, co2_target, sst, pco2, cumulative, dt);
            pco2 = new_pco2;
            cumulative = new_cumulative;
        }

        // Ocean pCO2 should approach (but not exceed) atmospheric
        assert!(
            pco2 < co2_target,
            "Ocean pCO2 {} should be less than atmospheric {}",
            pco2,
            co2_target
        );

        // Should have approached within ~100 ppm (the Revelle factor limits how close
        // the surface can get to the atmosphere without deep ocean equilibration)
        assert!(
            pco2 > co2_target - 100.0,
            "Ocean pCO2 {} should have approached atmospheric {} within 100 ppm",
            pco2,
            co2_target
        );
    }
}

mod energy_conservation {
    use rscm_magicc::parameters::ClimateUDEBParameters;

    /// Test that the ECS/lambda relationship is consistent.
    ///
    /// At equilibrium: lambda_global = Q_2x / ECS
    #[test]
    fn test_ecs_lambda_consistency() {
        let params = ClimateUDEBParameters::default();

        let lambda_global = params.lambda_global();
        let expected_ecs = params.rf_2xco2 / lambda_global;

        assert!(
            (params.ecs - expected_ecs).abs() < 0.01,
            "ECS ({}) should equal RF_2xCO2 / lambda_global ({})",
            params.ecs,
            expected_ecs
        );
    }

    /// Test that RLO (ratio of land to ocean warming) is reasonable.
    #[test]
    fn test_rlo_reasonable() {
        let params = ClimateUDEBParameters::default();

        // RLO should be > 1 (land warms more than ocean)
        assert!(
            params.rlo > 1.0,
            "RLO should be > 1 (land warms faster than ocean)"
        );

        // RLO should be reasonable (typically 1.2-1.6)
        assert!(params.rlo < 2.0, "RLO should be < 2.0 (reasonable range)");
    }

    /// Test that positive forcing leads to warming.
    #[test]
    fn test_positive_forcing_causes_warming() {
        let params = ClimateUDEBParameters::default();

        // ECS should be positive for warming from positive forcing
        assert!(
            params.ecs > 0.0,
            "ECS should be positive for warming from positive forcing"
        );

        // RF_2xCO2 should be positive (CO2 doubling forcing)
        assert!(
            params.rf_2xco2 > 0.0,
            "RF_2xCO2 should be positive (CO2 doubling forcing)"
        );

        // Lambda (feedback parameter) should be positive for stable climate
        let lambda = params.lambda_global();
        assert!(
            lambda > 0.0,
            "Lambda should be positive for stable climate response"
        );
    }
}

// TODO: Add forcing_consistency tests when GHG forcing components are implemented

mod physical_bounds {
    use super::*;

    /// Test that terrestrial NPP remains positive.
    #[test]
    fn test_npp_positive() {
        let params = TerrestrialCarbonParameters::default();

        // NPP should be positive at pre-industrial
        assert!(params.npp_pi > 0.0, "Pre-industrial NPP should be positive");

        // Respiration should be positive
        assert!(
            params.respiration_pi > 0.0,
            "Pre-industrial respiration should be positive"
        );
    }

    /// Test that ocean exchange rate is positive.
    #[test]
    fn test_ocean_exchange_positive() {
        let params = OceanCarbonParameters::default();

        // Gas exchange rate should be positive
        let k = params.gas_exchange_rate();
        assert!(k > 0.0, "Gas exchange rate should be positive");
    }
}
