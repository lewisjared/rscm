use numpy::array;
use numpy::ndarray::{Array, Axis};
use rscm_components::{
    CO2ERFParameters, CarbonCycle, CarbonCycleParameters, SolverOptions, CO2ERF,
};
use rscm_core::interpolate::strategies::{InterpolationStrategy, NextStrategy, PreviousStrategy};
use rscm_core::model::ModelBuilder;
use rscm_core::spatial::ScalarGrid;
use rscm_core::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_carbon_cycle() {
    let tau = 20.3;
    let conc_pi = 280.0;
    let conc_initial = 280.0;
    let t_initial = 1800.0;
    let emissions_level = 10.0;
    let step_year = 1850.0;
    //Can use any temperature as the temperature feedback is set to zero
    // so this is effectively a noise parameter.
    let temperature_value = 1.0;
    let step_size = 1.0 / 120.0;

    let gtc_per_ppm = 2.13;

    // Have to have no temperature feedback for this to work
    let alpha_temperature = 0.0;

    let time_axis = TimeAxis::from_values(Array::range(t_initial, 2100.0, 1.0));

    let mut builder = ModelBuilder::new();

    let get_exp_values_before_step = |time: Time| -> FloatValue {
        (conc_initial - conc_pi) * (-(time - t_initial) / tau).exp() + conc_pi
    };

    let get_exp_values_after_step = |time: Time| -> FloatValue {
        emissions_level / gtc_per_ppm * tau * (1.0 - (-(time - step_year) / tau).exp())
            + get_exp_values_before_step(time)
    };

    let emissions = Timeseries::new(
        array![0.0, 0.0, emissions_level, emissions_level].insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_bounds(array![
            t_initial,
            (t_initial + step_year) / 2.0,
            step_year,
            step_year + 50.0,
            2100.0
        ])),
        ScalarGrid,
        "GtC / yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );
    let temperature = Timeseries::new(
        array![temperature_value].insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_bounds(array![t_initial, 2100.0])),
        ScalarGrid,
        "K".to_string(),
        InterpolationStrategy::from(NextStrategy::new(true)),
    );

    // Build a model consisting of a single carbon cycle component
    let mut model = builder
        .with_component(Arc::new(
            CarbonCycle::from_parameters(CarbonCycleParameters {
                tau,
                conc_pi,
                alpha_temperature,
            })
            .with_solver_options(SolverOptions { step_size }),
        ))
        .with_initial_values(HashMap::from([
            ("Cumulative Land Uptake".to_string(), 0.0),
            ("Cumulative Emissions|CO2".to_string(), 0.0),
            ("Atmospheric Concentration|CO2".to_string(), conc_initial),
        ]))
        .with_time_axis(time_axis.clone())
        .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
        .with_exogenous_variable("Surface Temperature", temperature)
        .build()
        .unwrap();

    model.run();

    let co2_conc = model
        .timeseries()
        .get_data("Atmospheric Concentration|CO2")
        .and_then(|data| data.as_scalar())
        .unwrap();

    let co2_emissions = model
        .timeseries()
        .get_data("Emissions|CO2|Anthropogenic")
        .and_then(|data| data.as_scalar())
        .unwrap();
    let expected_concentrations: Vec<FloatValue> = time_axis
        .values()
        .iter()
        .map(|t| match *t < step_year {
            true => get_exp_values_before_step(*t),
            false => get_exp_values_after_step(*t),
        })
        .collect();
    let expected_emissions: Vec<FloatValue> = time_axis
        .values()
        .iter()
        .map(|t| match *t < step_year {
            true => 0.0,
            false => emissions_level,
        })
        .collect();
    assert_eq!(
        co2_emissions.values().column(0).to_vec(),
        expected_emissions
    );

    // Verify CO2 concentrations match the analytical solution
    let actual_concentrations: Vec<FloatValue> = co2_conc.values().column(0).to_vec();
    for (i, (actual, expected)) in actual_concentrations
        .iter()
        .zip(expected_concentrations.iter())
        .enumerate()
    {
        let rel_error = if expected.abs() > 1e-10 {
            (actual - expected).abs() / expected.abs()
        } else {
            (actual - expected).abs()
        };
        assert!(
            rel_error < 0.01,
            "Concentration mismatch at index {}: actual={}, expected={}, rel_error={}",
            i,
            actual,
            expected,
            rel_error
        );
    }
}

#[test]
fn test_coupled_model() {
    let tau = 20.3;
    let conc_pi = 280.0;
    let t_initial = 1750.0;
    let erf_2xco2 = 4.0;
    let step_year = 1850.0;
    let step_size = 1.0 / 120.0;

    // Have to have no temperature feedback for this to work
    let alpha_temperature = 0.0;

    let time_axis = TimeAxis::from_values(Array::range(t_initial, 2100.0, 1.0));
    let emissions = Timeseries::new(
        array![0.0, 0.0, step_size, step_size].insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_bounds(array![
            t_initial,
            (t_initial + step_year) / 2.0,
            step_year,
            step_year + 50.0,
            2100.0
        ])),
        ScalarGrid,
        "GtC / yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );
    let surface_temp = Timeseries::new(
        array![0.42].insert_axis(Axis(1)),
        Arc::new(TimeAxis::from_bounds(array![t_initial, 2100.0])),
        ScalarGrid,
        "K".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    let mut builder = ModelBuilder::new();

    // Build a model consisting of a carbon cycle and a CO2-only ERF component
    let mut model = builder
        .with_component(Arc::new(CarbonCycle::from_parameters(
            CarbonCycleParameters {
                tau,
                conc_pi,
                alpha_temperature,
            },
        )))
        .with_component(Arc::new(CO2ERF::from_parameters(CO2ERFParameters {
            erf_2xco2,
            conc_pi,
        })))
        .with_time_axis(time_axis)
        .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
        .with_exogenous_variable("Surface Temperature", surface_temp)
        .with_initial_values(HashMap::from([
            ("Cumulative Land Uptake".to_string(), 0.0),
            ("Cumulative Emissions|CO2".to_string(), 0.0),
            ("Atmospheric Concentration|CO2".to_string(), 300.0),
        ]))
        .build()
        .unwrap();

    let mut variable_names: Vec<&str> =
        model.timeseries().iter().map(|x| x.name.as_str()).collect();
    variable_names.sort();

    println!("{:?}", variable_names);
    assert_eq!(
        variable_names,
        vec![
            "Atmospheric Concentration|CO2",
            "Cumulative Emissions|CO2",
            "Cumulative Land Uptake",
            "Effective Radiative Forcing|CO2",
            "Emissions|CO2|Anthropogenic",
            "Surface Temperature"
        ]
    );

    println!("{:?}", model.as_dot());

    // Run the model
    model.run()
}
