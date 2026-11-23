# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Building a Complete Scenario Pipeline
#
# This tutorial runs several future scenarios from one shared historical run.
# The pattern has three steps:
#
# 1. Run the model over the historical period, up to the branch point (2015).
# 2. Record the model there with `Model.checkpoint()`.
# 3. Branch a model for each scenario from that record with `Model.branch()`, giving it the scenario's emissions.
#
# Every scenario then starts from the same historical state, including any internal component state,
# so differences between them come only from their emissions pathways.
#
# ## Related Resources
#
# - [Coupled Models](../coupled_model/): Building multi-component models
# - [State Serialisation](../state_serialisation/): Saving and loading model state
# - [Key Concepts](../../key_concepts/): Core RSCM architecture

# %%
import numpy as np

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    Model,
    ModelBuilder,
    TimeAxis,
    Timeseries,
)

# %% [markdown]
# ## Define Scenario Data
#
# A real workflow would load SSP emissions from a database.
# Here each pathway is a CO2 emissions `Timeseries` that interpolates linearly between a few breakpoints:
#
# - **Historical**: Approximate emissions from 1750 to 2015.
# - **ssp119**: Rapid decarbonisation, net-zero by 2050.
# - **ssp245**: Middle-of-the-road pathway.
# - **ssp585**: Fossil-fuelled development.

# %%
T_START = 1750.0
T_BRANCH = 2015.0
T_END = 2100.0

# %% tags=["remove_input"]
EMISSIONS_BREAKPOINTS = {
    "historical": (
        [T_START, 1850.0, 1950.0, 2000.0, T_BRANCH],
        [0.0, 0.1, 1.5, 6.5, 10.0],
    ),
    "ssp119": ([T_BRANCH, 2030.0, 2050.0, 2075.0, T_END], [10.0, 7.0, 0.0, -2.0, -3.0]),
    "ssp245": ([T_BRANCH, 2030.0, 2050.0, 2075.0, T_END], [10.0, 11.0, 10.0, 8.0, 5.0]),
    "ssp585": (
        [T_BRANCH, 2030.0, 2050.0, 2075.0, T_END],
        [10.0, 14.0, 18.0, 20.0, 22.0],
    ),
}


def emissions_timeseries(times: list[float], values: list[float]) -> Timeseries:
    """CO2 emissions interpolating linearly between breakpoints."""
    return Timeseries(
        np.array(values),
        TimeAxis.from_values(np.array(times)),
        "GtC/yr",
        InterpolationStrategy.Linear,
    )


historical_times, historical_values = EMISSIONS_BREAKPOINTS["historical"]
historical_emissions = emissions_timeseries(historical_times, historical_values)
# Each scenario continues the historical pathway, so its series covers the whole run.
scenario_emissions = {
    name: emissions_timeseries(
        historical_times + times[1:], historical_values + values[1:]
    )
    for name, (times, values) in EMISSIONS_BREAKPOINTS.items()
    if name != "historical"
}

# %% tags=["remove_input"]
import matplotlib.pyplot as plt

SCENARIO_COLOURS = {
    "historical": "black",
    "ssp119": "green",
    "ssp245": "orange",
    "ssp585": "red",
}

fig, ax = plt.subplots(figsize=(10, 4))
hist_times = np.arange(T_START, T_BRANCH + 1, 5)
ax.plot(
    hist_times,
    [historical_emissions.at_time(t) for t in hist_times],
    color=SCENARIO_COLOURS["historical"],
    label="historical",
)
future_times = np.arange(T_BRANCH, T_END + 1, 5)
for scenario, ts in scenario_emissions.items():
    ax.plot(
        future_times,
        [ts.at_time(t) for t in future_times],
        color=SCENARIO_COLOURS[scenario],
        label=scenario,
    )
ax.axvline(T_BRANCH, color="gray", linestyle="--", alpha=0.5, label="Branch point")
ax.set_xlabel("Year")
ax.set_ylabel("Emissions (GtC/yr)")
ax.set_title("CO2 Emissions Scenarios")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Configure the Model
#
# The model couples two components:
#
# ```
# Emissions -> [Carbon Cycle] -> CO2 Concentration -> [CO2 ERF] -> Forcing
# ```
#
# `build_model` wraps the builder chain.
# The carbon cycle reads the surface air temperature change,
# but `alpha_temperature=0` turns that feedback off, so a constant zero series stands in for it.
#
# A temperature response needs a climate component such as TwoLayer,
# with a `VariableSchema` aggregate that sums the forcing from each species.
# See [Coupled Models](../coupled_model/) and [Variable Schemas](../variable_schema/).

# %%
PRE_INDUSTRIAL = {
    "Cumulative Carbon Uptake|Land": 0.0,
    "Cumulative Emissions|CO2": 0.0,
    "Atmospheric Concentrations|CO2": 280.0,
}

surface_temperature = Timeseries(
    np.array([0.0]),
    TimeAxis.from_bounds(np.array([T_START, T_END])),
    "K",
    InterpolationStrategy.Previous,
)


def build_model(time_axis: TimeAxis, co2_emissions: Timeseries, initial_values: dict):
    """Build the carbon cycle and CO2 ERF model on `time_axis`."""
    carbon_cycle = CarbonCycleBuilder.from_parameters(
        dict(tau=30.0, co2_pi=280.0, alpha_temperature=0.0)
    ).build()
    co2_erf = CO2ERFBuilder.from_parameters(dict(erf_2xco2=3.7, co2_pi=280.0)).build()

    return (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_component(carbon_cycle)
        .with_component(co2_erf)
        .with_prescribed_variable("Emissions|CO2", co2_emissions)
        .with_prescribed_variable("Surface Air Temperature Change", surface_temperature)
        .with_initial_values(initial_values)
    ).build()


# %% [markdown]
# ## Step 1: Run the Historical Period
#
# The time axis uses decadal steps before 1900, 5-yearly steps to 2000 and annual steps to 2100.
# The model is built with one scenario's emissions, which match every other scenario up to 2015,
# and stepped until it reaches the branch point.

# %%
time_axis = TimeAxis.from_values(
    np.concatenate(
        [
            np.arange(T_START, 1900.0, 10.0),
            np.arange(1900.0, 2000.0, 5.0),
            np.arange(2000.0, T_END + 1, 1.0),
        ]
    )
)

historical_model = build_model(time_axis, scenario_emissions["ssp245"], PRE_INDUSTRIAL)
while historical_model.current_time() < T_BRANCH:
    historical_model.step()
historical_model.current_time()

# %% [markdown]
# ## Step 2: Record the Branch Point
#
# `checkpoint()` returns a `Record` holding the model's builder, its values up to 2015
# and each component's internal state.
# It serialises as JSON, so the historical state can be archived or shared.
# See [State Serialisation](../state_serialisation/) for what a record holds.

# %%
branch_point = historical_model.checkpoint()
historical_results = branch_point.timeseries()
print(
    f"Recorded at step {branch_point.time_index}, {len(branch_point.to_json())} bytes of JSON"
)

# %% [markdown]
# ## Step 3: Run the Scenarios
#
# `Model.branch()` rebuilds the model from the record with each scenario's emissions.
# Values up to the branch point stay as recorded, so the new emissions take effect after it.
# The time axis cannot change, and `parameters` can also swap a component's parameters by its label.

# %%
scenario_results = {}
for scenario, co2_emissions in scenario_emissions.items():
    model = Model.branch(branch_point, prescribed={"Emissions|CO2": co2_emissions})
    model.run()
    scenario_results[scenario] = model.timeseries()

# %% [markdown]
# ## Step 4: Compare the Scenarios
#
# `to_dataframe()` converts each `TimeseriesCollection` into a pandas DataFrame,
# and `pd.concat` stacks the four runs into `all_runs` with a (variable, unit, scenario) index.
# Every variable here is scalar, so the region level is dropped.

# %%
import pandas as pd

runs = {"historical": historical_results, **scenario_results}
all_runs = (
    pd.concat({s: r.to_dataframe() for s, r in runs.items()}, names=["scenario"])
    .droplevel("region")
    .reorder_levels(["variable", "unit", "scenario"])
)

# %% tags=["remove_input"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
panels = [
    ("Atmospheric Concentrations|CO2", "CO2 (ppm)", "Atmospheric CO2 Concentration"),
    (
        "Effective Radiative Forcing|CO2",
        "ERF (W/m2)",
        "CO2 Effective Radiative Forcing",
    ),
]
for ax, (variable, ylabel, title) in zip(axes, panels):
    for (_, scenario), row in all_runs.xs(variable, level="variable").iterrows():
        values = row.dropna()
        ax.plot(
            values.index,
            values.values,
            label=scenario,
            color=SCENARIO_COLOURS[scenario],
        )
    ax.axvline(T_BRANCH, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# The carbon cycle relaxes concentrations towards its 280 ppm pre-industrial value.
# Net-negative emissions in ssp119 push concentrations below that value by 2100,
# which this simple carbon cycle allows.
#
# Selecting two variables and two years from `all_runs` gives a compact comparison:

# %%
(
    all_runs.droplevel("unit")
    .loc[
        ["Atmospheric Concentrations|CO2", "Effective Radiative Forcing|CO2"],
        [2050.0, 2100.0],
    ]
    .dropna()
    .round(2)
)

# %% [markdown]
# `all_runs.to_csv(...)` exports the combined results, with the index preserving the metadata.
#
# ## Key Patterns
#
# - **Branching**: `Model.branch()` starts every scenario from one recorded historical state,
#   including component internal state.
# - **Records**: `checkpoint()` saves the branch point, so later scenarios can branch from a file.
# - **Time axis management**: One time axis can use coarser steps in the historical period.
#
# ## Next Steps
#
# - Add more greenhouse gases (CH4, N2O) using additional components.
# - Include aerosol forcing for more realistic scenarios.
# - Load real SSP data from scenario databases.
# - Add uncertainty analysis by running parameter ensembles.
