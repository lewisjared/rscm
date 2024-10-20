"""
Common model builders
"""

import numpy as np

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
)


def example_model_builder(t_initial, t_final) -> ModelBuilder:
    """
    Build a simple coupled model with a carbon cycle and CO2 ERF component

    Parameters
    ----------
    t_initial
        Start year
    t_final
        End year

    Returns
    -------
        Model builder with the components and exogenous variables set up
    """
    time_axis = TimeAxis.from_values(
        np.concat(
            [np.arange(t_initial, 2015.0, 5), np.arange(2015.0, t_final + 1, 1.0)]
        )
    )
    tau = 20.3
    conc_pi = 280.0
    erf_2xco2 = 4.0
    alpha_temperature = 0.0  # Have no temperature feedback

    # %%
    co2_erf_component = CO2ERFBuilder.from_parameters(
        dict(erf_2xco2=erf_2xco2, conc_pi=conc_pi)
    ).build()
    carbon_cycle_component = CarbonCycleBuilder.from_parameters(
        dict(tau=tau, conc_pi=conc_pi, alpha_temperature=alpha_temperature)
    ).build()

    initial_state = {
        "Cumulative Land Uptake": 0.0,
        "Cumulative Emissions|CO2": 0.0,
        "Atmospheric Concentration|CO2": 300.0,
    }

    step_year = 1850.0
    step_size = 10.0

    emissions = Timeseries(
        np.asarray([0.0, 0.0, step_size, step_size]),
        TimeAxis.from_bounds(
            np.asarray(
                [
                    t_initial,
                    (t_initial + step_year) / 2.0,
                    step_year,
                    step_year + 50.0,
                    t_final,
                ]
            )
        ),
        "GtC / yr",
        InterpolationStrategy.Previous,
    )

    surface_temp = Timeseries(
        np.asarray([0.42]),
        TimeAxis.from_bounds(np.asarray([float(t_initial), float(t_final)])),
        "K",
        InterpolationStrategy.Previous,
    )

    model = (
        ModelBuilder()
        .with_rust_component(carbon_cycle_component)
        .with_rust_component(co2_erf_component)
        .with_time_axis(time_axis)
        .with_exogenous_variable("Surface Temperature", surface_temp)
        .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
        .with_initial_values(initial_state)
    )

    return model
