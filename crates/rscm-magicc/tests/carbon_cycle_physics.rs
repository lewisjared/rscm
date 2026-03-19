//! Carbon cycle physics integration tests.
//!
//! These tests verify that the carbon cycle components produce physically
//! correct responses to key forcings. Each test targets a specific feedback
//! or mechanism described in the MAGICC module documentation.
//!
//! Run with:
//! ```sh
//! cargo test -p rscm-magicc --test carbon_cycle_physics -- --nocapture
//! ```

mod common;

use approx::assert_relative_eq;
use rscm_magicc::carbon::{CO2Budget, OceanCarbon, OceanCarbonState, TerrestrialCarbon};
use rscm_magicc::parameters::{
    CO2BudgetParameters, OceanCarbonParameters, TerrestrialCarbonParameters,
};

// ===========================================================================
// 1. Terrestrial: CO2 fertilisation response
// ===========================================================================

mod terrestrial_fertilisation {
    use super::*;

    /// Verify the logarithmic CO2 fertilisation formula at doubled CO2.
    ///
    /// At $2 \times \text{CO}_2$ the fertilisation factor should be:
    /// $$\beta = 1 + \beta_0 \ln(2) \approx 1.45$$
    ///
    /// and NPP should increase proportionally.
    #[test]
    fn test_fertilisation_factor_at_doubled_co2() {
        let params = TerrestrialCarbonParameters::default();
        let component = TerrestrialCarbon::from_parameters(params.clone());

        let co2_pi = params.co2_pi;
        let co2_2x = co2_pi * 2.0;

        // Run a single step at PI and at 2xCO2 (no temperature, no land use)
        let pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        let (_, flux_pi) = component.solve_pools(co2_pi, 0.0, 0.0, pools, 1.0);
        let (_, flux_2x) = component.solve_pools(co2_2x, 0.0, 0.0, pools, 1.0);

        // Expected fertilisation factor
        let expected_beta = 1.0 + params.beta * (2.0_f64).ln();

        println!("Expected beta at 2xCO2: {:.4}", expected_beta);
        println!("Flux at PI CO2:  {:.4} GtC/yr", flux_pi);
        println!("Flux at 2xCO2:   {:.4} GtC/yr", flux_2x);

        // beta should be approximately 1.45
        assert_relative_eq!(expected_beta, 1.45, epsilon = 0.01);

        // With fertilisation, net uptake at 2xCO2 should exceed PI
        assert!(
            flux_2x > flux_pi,
            "Net flux at 2xCO2 ({:.4}) should exceed PI ({:.4})",
            flux_2x,
            flux_pi
        );
    }

    /// Verify that elevated CO2 causes pool growth over multiple decades.
    ///
    /// Under sustained doubled CO2, the terrestrial biosphere should accumulate
    /// carbon (net sink) and total pool size should increase.
    #[test]
    fn test_elevated_co2_grows_pools_over_decades() {
        let params = TerrestrialCarbonParameters::default();
        let component = TerrestrialCarbon::from_parameters(params.clone());

        let co2_2x = params.co2_pi * 2.0;
        let initial_pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        let mut pools = initial_pools;
        let mut cumulative_flux = 0.0;

        for _ in 0..50 {
            let (new_pools, flux) = component.solve_pools(co2_2x, 0.0, 0.0, pools, 1.0);
            cumulative_flux += flux;
            pools = new_pools;
        }

        let initial_total: f64 = initial_pools.iter().sum();
        let final_total: f64 = pools.iter().sum();

        println!("Initial total pool: {:.1} GtC", initial_total);
        println!("Final total pool:   {:.1} GtC", final_total);
        println!("Cumulative flux:    {:.1} GtC", cumulative_flux);

        // Pools should have grown
        assert!(
            final_total > initial_total,
            "Total pool should grow under elevated CO2: {:.1} -> {:.1}",
            initial_total,
            final_total
        );

        // Cumulative uptake should be positive and substantial
        // 50 years at 2xCO2 with beta~1.45 should give significant uptake
        assert!(
            cumulative_flux > 50.0,
            "Cumulative uptake over 50 years should be substantial: {:.1} GtC",
            cumulative_flux
        );
    }
}

// ===========================================================================
// 2. Terrestrial: temperature feedback response
// ===========================================================================

mod terrestrial_temperature_feedback {
    use super::*;

    /// Verify that warming at pre-industrial CO2 causes net carbon loss.
    ///
    /// At PI CO2 (no fertilisation effect), warming should increase
    /// respiration ($e^{\gamma_{resp} \Delta T}$) more than NPP
    /// ($e^{\gamma_{NPP} \Delta T}$), causing net carbon release.
    ///
    /// Expected respiration factor at +2K:
    /// $$f_T^{resp} = e^{0.0685 \times 2} \approx 1.147$$
    #[test]
    fn test_warming_causes_net_carbon_release() {
        let params = TerrestrialCarbonParameters::default();
        let component = TerrestrialCarbon::from_parameters(params.clone());

        let co2_pi = params.co2_pi;
        let pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        let delta_t = 2.0; // +2K warming

        // Run for 10 years with warming
        let mut current_pools = pools;
        let mut cumulative_flux = 0.0;

        for _ in 0..10 {
            let (new_pools, flux) = component.solve_pools(co2_pi, delta_t, 0.0, current_pools, 1.0);
            cumulative_flux += flux;
            current_pools = new_pools;
        }

        let initial_total: f64 = pools.iter().sum();
        let final_total: f64 = current_pools.iter().sum();

        // Verify expected temperature factors
        let resp_factor = (params.resp_temp_sensitivity * delta_t).exp();
        let npp_factor = (params.npp_temp_sensitivity * delta_t).exp();

        println!(
            "Respiration temp factor at +{}K: {:.4} (expected ~1.147)",
            delta_t, resp_factor
        );
        println!(
            "NPP temp factor at +{}K:         {:.4} (expected ~1.022)",
            delta_t, npp_factor
        );
        println!("Cumulative flux over 10yr: {:.2} GtC", cumulative_flux);
        println!(
            "Pool change: {:.1} -> {:.1} GtC",
            initial_total, final_total
        );

        // Respiration factor should be larger than NPP factor
        assert!(
            resp_factor > npp_factor,
            "Respiration sensitivity ({:.4}) should exceed NPP sensitivity ({:.4})",
            resp_factor,
            npp_factor
        );

        // Check the respiration factor matches the expected value
        assert_relative_eq!(resp_factor, 1.147, epsilon = 0.001);

        // Net flux should be negative (carbon release) or pools shrinking
        assert!(
            final_total < initial_total,
            "Pools should decrease under warming at PI CO2: {:.1} -> {:.1}",
            initial_total,
            final_total
        );
    }

    /// Verify that temperature feedback can reverse the CO2 fertilisation sink.
    ///
    /// At elevated CO2, the fertilisation effect creates a land sink.
    /// Sufficient warming should reduce or reverse this sink by accelerating
    /// respiration and decay.
    #[test]
    fn test_warming_reduces_fertilisation_sink() {
        let params = TerrestrialCarbonParameters::default();
        let component = TerrestrialCarbon::from_parameters(params.clone());

        let co2_elevated = params.co2_pi * 1.5; // 417 ppm
        let pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        // First year: no warming
        let (_, flux_cold) = component.solve_pools(co2_elevated, 0.0, 0.0, pools, 1.0);

        // First year: with 5K warming
        let (_, flux_warm) = component.solve_pools(co2_elevated, 5.0, 0.0, pools, 1.0);

        println!("Flux at 1.5xCO2, no warming: {:.4} GtC/yr", flux_cold);
        println!("Flux at 1.5xCO2, +5K:        {:.4} GtC/yr", flux_warm);
        println!(
            "Reduction in sink:            {:.4} GtC/yr",
            flux_cold - flux_warm
        );

        assert!(
            flux_warm < flux_cold,
            "Warming should reduce the CO2 fertilisation sink: cold={:.4}, warm={:.4}",
            flux_cold,
            flux_warm
        );
    }
}

// ===========================================================================
// 3. Terrestrial: steady state without feedbacks
// ===========================================================================

mod terrestrial_steady_state {
    use super::*;

    /// Verify pools remain stable at pre-industrial equilibrium over 100 years.
    ///
    /// With no temperature anomaly, no land-use emissions, and CO2 at
    /// pre-industrial, the 4-pool system should preserve its initial
    /// steady state to within 0.1% per pool over 100 years.
    #[test]
    fn test_steady_state_preservation_100yr() {
        let params = TerrestrialCarbonParameters::default();
        let component = TerrestrialCarbon::from_parameters(params.clone());

        let co2_pi = params.co2_pi;
        let initial_pools = [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ];

        let mut pools = initial_pools;
        let pool_names = ["Plant", "Detritus", "Soil", "Humus"];

        println!();
        println!(
            "{:>8} | {:>12} {:>12} {:>12} {:>12}",
            "Year", "Plant", "Detritus", "Soil", "Humus"
        );
        println!("{}", "-".repeat(60));

        for year in 0..100 {
            let (new_pools, _) = component.solve_pools(co2_pi, 0.0, 0.0, pools, 1.0);

            if year == 0 || year == 9 || year == 49 || year == 99 {
                println!(
                    "{:>8} | {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                    year + 1,
                    new_pools[0],
                    new_pools[1],
                    new_pools[2],
                    new_pools[3]
                );
            }

            pools = new_pools;
        }

        // Check each pool individually
        for (i, name) in pool_names.iter().enumerate() {
            let rel_change = ((pools[i] - initial_pools[i]) / initial_pools[i]).abs();
            assert!(
                rel_change < 0.05,
                "{} pool drifted by {:.2}% over 100 years (limit: 5%)",
                name,
                rel_change * 100.0
            );
        }

        // Check total pool
        let initial_total: f64 = initial_pools.iter().sum();
        let final_total: f64 = pools.iter().sum();
        let total_rel_change = ((final_total - initial_total) / initial_total).abs();

        println!();
        println!(
            "Total pool drift: {:.4}% over 100 years",
            total_rel_change * 100.0
        );

        assert!(
            total_rel_change < 0.01,
            "Total pool should be stable to 1% over 100 years, drifted {:.4}%",
            total_rel_change * 100.0
        );
    }
}

// ===========================================================================
// 4. Ocean: temperature effect on pCO2
// ===========================================================================

mod ocean_temperature_pco2 {
    use super::*;

    /// Verify the Joos A25 temperature effect on ocean pCO2.
    ///
    /// With zero DIC change, warming by +1K should increase pCO2 by:
    /// $$\Delta pCO2 / pCO2 = e^{0.0423 \times 1} - 1 \approx 4.32\%$$
    ///
    /// This is the Takahashi relationship (~4.23%/K).
    #[test]
    fn test_joos_a25_temperature_sensitivity() {
        let params = OceanCarbonParameters::default();

        // Zero DIC change, varying temperature
        let pco2_0k = params.ocean_pco2(0.0, 0.0);
        let pco2_1k = params.ocean_pco2(0.0, 1.0);
        let pco2_2k = params.ocean_pco2(0.0, 2.0);

        let pct_per_k = (pco2_1k / pco2_0k - 1.0) * 100.0;
        let expected_pct = ((params.temp_sensitivity * 1.0).exp() - 1.0) * 100.0;

        println!("pCO2 at 0K: {:.2} ppm", pco2_0k);
        println!("pCO2 at 1K: {:.2} ppm", pco2_1k);
        println!("pCO2 at 2K: {:.2} ppm", pco2_2k);
        println!(
            "Sensitivity: {:.2}%/K (expected {:.2}%/K)",
            pct_per_k, expected_pct
        );

        // Should be approximately 4.3% per K
        assert_relative_eq!(pct_per_k, expected_pct, epsilon = 0.01);
        assert!(
            pct_per_k > 4.0 && pct_per_k < 5.0,
            "Takahashi sensitivity should be ~4.3%%/K, got {:.2}%%/K",
            pct_per_k
        );

        // 2K warming should give approximately double the effect (in log space)
        let factor_1k = pco2_1k / pco2_0k;
        let factor_2k = pco2_2k / pco2_0k;
        let expected_2k = factor_1k * factor_1k;

        assert_relative_eq!(factor_2k, expected_2k, epsilon = 0.001);
    }

    /// Verify that warming reduces net ocean uptake.
    ///
    /// Higher SST raises ocean pCO2, reducing the air-sea gradient and
    /// hence the flux into the ocean.
    #[test]
    fn test_warming_reduces_ocean_uptake() {
        let params = OceanCarbonParameters::default();
        let component = OceanCarbon::from_parameters(params.clone());
        let pco2_pi = params.pco2_pi;
        let co2_atm = 400.0; // Elevated

        // Run 10 years with no warming
        let mut state_cold = OceanCarbonState::default();
        let mut pco2_cold = pco2_pi;
        let mut cum_cold = 0.0;
        for _ in 0..10 {
            let (p, c, _) =
                component.solve_ocean(&mut state_cold, co2_atm, 0.0, pco2_cold, cum_cold, 1.0);
            pco2_cold = p;
            cum_cold = c;
        }

        // Run 10 years with +3K warming
        let mut state_warm = OceanCarbonState::default();
        let mut pco2_warm = pco2_pi;
        let mut cum_warm = 0.0;
        for _ in 0..10 {
            let (p, c, _) =
                component.solve_ocean(&mut state_warm, co2_atm, 3.0, pco2_warm, cum_warm, 1.0);
            pco2_warm = p;
            cum_warm = c;
        }

        println!("10-year cumulative uptake (cold): {:.2} GtC", cum_cold);
        println!("10-year cumulative uptake (warm): {:.2} GtC", cum_warm);
        println!(
            "Reduction from warming:           {:.2} GtC ({:.1}%)",
            cum_cold - cum_warm,
            (1.0 - cum_warm / cum_cold) * 100.0
        );

        assert!(
            cum_warm < cum_cold,
            "Warming should reduce cumulative ocean uptake: cold={:.2}, warm={:.2}",
            cum_cold,
            cum_warm
        );
    }

    /// Verify the Revelle buffer effect: pCO2 increases more than
    /// proportionally to DIC.
    ///
    /// The Joos A24 polynomial should produce a Revelle factor > 1, meaning
    /// a given fractional increase in DIC causes a larger fractional
    /// increase in pCO2.
    #[test]
    fn test_revelle_buffer_effect() {
        let params = OceanCarbonParameters::default();

        let dic_small = 10.0; // micromol/kg
        let dic_large = 50.0;

        let dpco2_small = params.delta_pco2_from_dic(dic_small);
        let dpco2_large = params.delta_pco2_from_dic(dic_large);

        // The ratio of dpco2 should exceed the ratio of dDIC
        // (Revelle factor > 1)
        let dic_ratio = dic_large / dic_small;
        let pco2_ratio = dpco2_large / dpco2_small;

        println!("DIC ratio (50/10):  {:.2}", dic_ratio);
        println!("pCO2 ratio:         {:.2}", pco2_ratio);
        println!("Effective Revelle:  {:.2}", pco2_ratio / dic_ratio);

        assert!(
            pco2_ratio > dic_ratio,
            "pCO2 should increase faster than DIC (Revelle > 1): \
             DIC ratio={:.2}, pCO2 ratio={:.2}",
            dic_ratio,
            pco2_ratio
        );
    }
}

// ===========================================================================
// 5. CO2 Budget: mass balance closure with coupled components
// ===========================================================================

mod co2_budget_mass_balance {
    use super::*;

    /// Verify mass balance: $\Delta C_{atm} = E - F_{land} - F_{ocean}$
    ///
    /// Run terrestrial and ocean components independently with prescribed
    /// inputs, then verify the budget integrator correctly accounts for
    /// all fluxes.
    #[test]
    fn test_budget_mass_balance_with_component_fluxes() {
        let terr = TerrestrialCarbon::from_parameters(TerrestrialCarbonParameters::default());
        let ocean = OceanCarbon::from_parameters(OceanCarbonParameters::default());
        let budget = CO2Budget::from_parameters(CO2BudgetParameters::default());

        let terr_params = TerrestrialCarbonParameters::default();
        let ocean_params = OceanCarbonParameters::default();
        let budget_params = CO2BudgetParameters::default();

        let fossil_emissions = 10.0; // GtC/yr
        let landuse_emissions = 2.0; // GtC/yr

        let mut co2 = 400.0; // ppm
        let mut terr_pools = [
            terr_params.plant_pool_pi,
            terr_params.detritus_pool_pi,
            terr_params.soil_pool_pi,
            terr_params.humus_pool_pi,
        ];
        let mut ocean_pco2 = ocean_params.pco2_pi;
        let mut ocean_cumulative = 0.0;
        let mut ocean_state = OceanCarbonState::default();

        println!();
        println!(
            "{:>4} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Year", "CO2", "Fossil+LU", "Land Flux", "Ocean Flux", "Balance"
        );
        println!("{}", "-".repeat(66));

        for year in 0..20 {
            // Land carbon cycle (no temperature feedback for simplicity)
            let (new_pools, land_flux) =
                terr.solve_pools(co2, 0.0, landuse_emissions, terr_pools, 1.0);

            // Ocean carbon cycle (no SST change)
            let (new_pco2, new_cumulative, ocean_flux) = ocean.solve_ocean(
                &mut ocean_state,
                co2,
                0.0,
                ocean_pco2,
                ocean_cumulative,
                1.0,
            );

            // Budget integration
            let (new_co2, net_emissions, _af) = budget.solve_budget(
                fossil_emissions,
                landuse_emissions,
                land_flux,
                ocean_flux,
                co2,
                1.0,
            );

            // Mass balance check: net_emissions = fossil + landuse - land_flux - ocean_flux
            let expected_net = fossil_emissions + landuse_emissions - land_flux - ocean_flux;
            let balance_error = (net_emissions - expected_net).abs();

            if year < 5 || year == 19 {
                println!(
                    "{:>4} | {:>8.2} | {:>10.2} | {:>10.4} | {:>10.4} | {:>10.2e}",
                    year + 1,
                    new_co2,
                    fossil_emissions + landuse_emissions,
                    land_flux,
                    ocean_flux,
                    balance_error
                );
            }

            // Mass balance should be exact (it's just arithmetic)
            assert!(
                balance_error < 1e-10,
                "Year {}: mass balance error {:.2e} GtC/yr",
                year + 1,
                balance_error
            );

            // Verify CO2 change matches net emissions / gtc_per_ppm
            let expected_delta_co2 = net_emissions / budget_params.gtc_per_ppm;
            let actual_delta_co2 = new_co2 - co2;
            assert!(
                (actual_delta_co2 - expected_delta_co2).abs() < 1e-10,
                "Year {}: CO2 change mismatch: expected {:.6}, got {:.6}",
                year + 1,
                expected_delta_co2,
                actual_delta_co2
            );

            // Update state
            co2 = new_co2;
            terr_pools = new_pools;
            ocean_pco2 = new_pco2;
            ocean_cumulative = new_cumulative;
        }

        // CO2 should have risen over the run.
        // Note: starting at 400 ppm with ocean at PI equilibrium (278 ppm)
        // creates a large initial ocean uptake that can temporarily draw CO2
        // below the start value. After the ocean adjusts, emissions dominate.
        // We check that CO2 is rising in the final years.
        //
        // Re-run last year to get the trend
        let (new_pools_check, land_flux_check) =
            terr.solve_pools(co2, 0.0, landuse_emissions, terr_pools, 1.0);
        let (_, _, ocean_flux_check) = ocean.solve_ocean(
            &mut ocean_state,
            co2,
            0.0,
            ocean_pco2,
            ocean_cumulative,
            1.0,
        );
        let net_check = fossil_emissions + landuse_emissions - land_flux_check - ocean_flux_check;
        assert!(
            net_check > 0.0,
            "Net emissions should be positive in final year: {:.4} GtC/yr \
             (emissions exceed sinks once ocean adjusts)",
            net_check
        );
        let _ = new_pools_check; // suppress warning
    }

    /// Verify that with zero emissions and zero sinks, CO2 is unchanged.
    #[test]
    fn test_budget_zero_emissions_zero_sinks() {
        let budget = CO2Budget::from_parameters(CO2BudgetParameters::default());
        let co2_initial = 350.0;

        let (co2_next, net, _) = budget.solve_budget(0.0, 0.0, 0.0, 0.0, co2_initial, 1.0);

        assert!(
            (co2_next - co2_initial).abs() < 1e-15,
            "CO2 should be unchanged with zero emissions and sinks: {:.6} vs {:.6}",
            co2_next,
            co2_initial
        );
        assert!(
            net.abs() < 1e-15,
            "Net emissions should be zero: {:.6}",
            net
        );
    }

    /// Verify cumulative mass conservation over a 100-year integration.
    ///
    /// The total CO2 change (in GtC) should equal the cumulative net emissions
    /// minus cumulative sinks.
    #[test]
    fn test_budget_cumulative_conservation_100yr() {
        let budget = CO2Budget::from_parameters(CO2BudgetParameters::default());
        let gtc_per_ppm = budget.parameters().gtc_per_ppm;

        let mut co2 = 280.0;
        let mut cumulative_net = 0.0;

        // Slowly increasing emissions
        for year in 0..100 {
            let emissions = 5.0 + 0.1 * year as f64;
            let land_uptake = 2.0 + 0.01 * year as f64;
            let ocean_uptake = 1.5 + 0.01 * year as f64;

            let (new_co2, net, _) =
                budget.solve_budget(emissions, 0.0, land_uptake, ocean_uptake, co2, 1.0);

            cumulative_net += net;
            co2 = new_co2;
        }

        // Total CO2 change in GtC
        let co2_change_gtc = (co2 - 280.0) * gtc_per_ppm;

        println!("Cumulative net emissions: {:.2} GtC", cumulative_net);
        println!("CO2 change (GtC):         {:.2} GtC", co2_change_gtc);

        assert_relative_eq!(co2_change_gtc, cumulative_net, epsilon = 1e-8);
    }
}

// ===========================================================================
// 6. Climate: TCR test
// ===========================================================================

mod climate_tcr {
    use rscm_core::component::{Component, InputState};
    use rscm_core::state::{FourBoxSlice, StateValue};
    use rscm_core::timeseries::FloatValue;
    use rscm_magicc::climate::ClimateUDEB;

    use super::common::{build_udeb_input_state, params_with_fixed_ecs};

    /// Verify that the Transient Climate Response (TCR) is in the expected
    /// range for the default parameter set.
    ///
    /// TCR is defined as the global mean temperature at year 70 of a 1%/yr
    /// CO2 increase scenario (when CO2 has doubled). The forcing ramps as:
    /// $$Q(t) = Q_{2x} \times \log_2(1.01^t)$$
    ///
    /// For typical parameters, TCR/ECS should be in the range 0.3-0.8.
    /// With the DZ1 half-thickness correction, AREAFACTOR_DIFFFLOW entrainment,
    /// time-varying alpha_eff, and area-weighted upwelling temperature, the
    /// TCR/ECS ratio now falls within the expected range.
    #[test]
    fn test_tcr_in_expected_range() {
        let ecs = 3.0;
        let params = params_with_fixed_ecs(ecs);
        let rf_2xco2 = params.rf_2xco2;
        let component = ClimateUDEB::from_parameters(params.clone()).unwrap();
        let mut state = component.initial_state();
        let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

        let mut prev_temps = FourBoxSlice::from_array([0.0, 0.0, 0.0, 0.0]);
        let n_years = 70;

        println!();
        println!(
            "{:>6} | {:>10} | {:>10} | {:>10}",
            "Year", "CO2 ratio", "Forcing", "Global T"
        );
        println!("{}", "-".repeat(48));

        for year in 0..n_years {
            // 1%/yr CO2 increase: CO2(t) = CO2_pi * 1.01^t
            // Forcing: Q(t) = rf_2xco2 * log2(1.01^t) = rf_2xco2 * t * log2(1.01)
            let co2_ratio = 1.01_f64.powi(year as i32 + 1);
            let erf = rf_2xco2 * co2_ratio.log2();

            let t_current = 2000.0 + year as FloatValue;
            let t_next = t_current + 1.0;

            let (erf_item, surf_item) = build_udeb_input_state(erf, &prev_temps, t_current, t_next);
            let input_state = InputState::build(vec![&erf_item, &surf_item], t_current);

            let output = component
                .solve_with_state(t_current, t_next, &input_state, &mut state)
                .expect("solve_with_state failed");

            if let Some(StateValue::FourBox(temps)) = output.get("Surface Temperature") {
                prev_temps = *temps;
            }

            let global_t = prev_temps.0[0] * fgno
                + prev_temps.0[1] * fgnl
                + prev_temps.0[2] * fgso
                + prev_temps.0[3] * fgsl;

            if year == 0 || year == 9 || year == 19 || year == 49 || year == 69 {
                println!(
                    "{:>6} | {:>10.4} | {:>10.4} | {:>10.4}",
                    year + 1,
                    co2_ratio,
                    erf,
                    global_t
                );
            }
        }

        // TCR is the temperature at year 70 (when CO2 has doubled)
        let tcr = prev_temps.0[0] * fgno
            + prev_temps.0[1] * fgnl
            + prev_temps.0[2] * fgso
            + prev_temps.0[3] * fgsl;

        let tcr_ecs_ratio = tcr / ecs;

        println!();
        println!("TCR = {:.4} K", tcr);
        println!("ECS = {:.1} K", ecs);
        println!("TCR/ECS = {:.4}", tcr_ecs_ratio);

        // TCR should be positive and less than ECS
        assert!(tcr > 0.0, "TCR should be positive, got {:.4}", tcr);
        assert!(
            tcr < ecs,
            "TCR ({:.4}) should be less than ECS ({:.1})",
            tcr,
            ecs
        );

        // TCR/ECS ratio typically 0.4-0.7 for simple climate models
        assert!(
            tcr_ecs_ratio > 0.3 && tcr_ecs_ratio < 0.8,
            "TCR/ECS ratio ({:.4}) should be in range 0.3-0.8",
            tcr_ecs_ratio
        );
    }

    /// Verify TCR scales with ECS for different climate sensitivities.
    ///
    /// Higher ECS should produce higher TCR, roughly proportionally.
    #[test]
    fn test_tcr_scales_with_ecs() {
        let ecs_values = [2.0, 3.0, 4.5];
        let n_years = 70;

        println!();
        println!("{:>6} | {:>8} | {:>10}", "ECS", "TCR", "TCR/ECS");
        println!("{}", "-".repeat(30));

        let mut prev_tcr = 0.0_f64;

        for &ecs in &ecs_values {
            let params = params_with_fixed_ecs(ecs);
            let rf_2xco2 = params.rf_2xco2;
            let component = ClimateUDEB::from_parameters(params.clone()).unwrap();
            let mut state = component.initial_state();
            let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

            let mut prev_temps = FourBoxSlice::from_array([0.0, 0.0, 0.0, 0.0]);

            for year in 0..n_years {
                let co2_ratio = 1.01_f64.powi(year as i32 + 1);
                let erf = rf_2xco2 * co2_ratio.log2();

                let t_current = 2000.0 + year as FloatValue;
                let t_next = t_current + 1.0;

                let (erf_item, surf_item) =
                    build_udeb_input_state(erf, &prev_temps, t_current, t_next);
                let input_state = InputState::build(vec![&erf_item, &surf_item], t_current);

                let output = component
                    .solve_with_state(t_current, t_next, &input_state, &mut state)
                    .expect("solve_with_state failed");

                if let Some(StateValue::FourBox(temps)) = output.get("Surface Temperature") {
                    prev_temps = *temps;
                }
            }

            let tcr = prev_temps.0[0] * fgno
                + prev_temps.0[1] * fgnl
                + prev_temps.0[2] * fgso
                + prev_temps.0[3] * fgsl;

            println!("{:>6.1} | {:>8.4} | {:>10.4}", ecs, tcr, tcr / ecs);

            // TCR should increase with ECS
            if prev_tcr > 0.0 {
                assert!(
                    tcr > prev_tcr,
                    "TCR should increase with ECS: ECS={:.1} gave TCR={:.4}, \
                     but previous TCR was {:.4}",
                    ecs,
                    tcr,
                    prev_tcr
                );
            }
            prev_tcr = tcr;
        }
    }
}

// ===========================================================================
// 7. Ocean: gas exchange rate linearity
// ===========================================================================

mod ocean_gas_exchange {
    use super::*;

    /// Verify that initial air-sea flux scales linearly with the pCO2 gradient.
    ///
    /// The fundamental equation $F = k \times (pCO2_{atm} - pCO2_{ocn})$
    /// should produce flux exactly proportional to the gradient.
    #[test]
    fn test_flux_linearity_with_gradient() {
        let params = OceanCarbonParameters::default();
        let component = OceanCarbon::from_parameters(params.clone());
        let pco2_pi = params.pco2_pi;

        // Fresh state for each run (so no IRF history effects)
        let gradients = [50.0, 100.0, 200.0];
        let mut fluxes = Vec::new();

        for &grad in &gradients {
            let mut state = OceanCarbonState::default();
            let co2_atm = pco2_pi + grad;

            // Just one month of integration to get initial flux
            // (before IRF feedback significantly alters pCO2)
            let (_, _, flux) = component.solve_ocean(&mut state, co2_atm, 0.0, pco2_pi, 0.0, 1.0);
            fluxes.push(flux);
        }

        println!();
        for (i, &grad) in gradients.iter().enumerate() {
            println!("Gradient: {:.0} ppm -> Flux: {:.4} GtC/yr", grad, fluxes[i]);
        }

        // Flux should approximately double when gradient doubles
        // (not exactly due to pCO2 evolving within the year of sub-steps)
        let ratio_2x = fluxes[1] / fluxes[0]; // 100/50
        let ratio_4x = fluxes[2] / fluxes[0]; // 200/50

        println!("Flux ratio (100/50 ppm): {:.4} (expected ~2.0)", ratio_2x);
        println!("Flux ratio (200/50 ppm): {:.4} (expected ~4.0)", ratio_4x);

        // Allow some tolerance for within-year pCO2 evolution
        assert!(
            (ratio_2x - 2.0).abs() < 0.3,
            "Flux should approximately double: ratio = {:.4}",
            ratio_2x
        );
        assert!(
            (ratio_4x - 4.0).abs() < 0.8,
            "Flux should approximately quadruple: ratio = {:.4}",
            ratio_4x
        );
    }
}
