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
# 1. Run the model over the historical period.
# 2. Read the state variables at the branch point (2015) from the historical results.
# 3. Build a new model for each scenario, starting from those values.
#
# Every scenario then starts from the same historical state,
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

emissions = {
    name: Timeseries(
        np.array(values),
        TimeAxis.from_values(np.array(times)),
        "GtC/yr",
        InterpolationStrategy.Linear,
    )
    for name, (times, values) in EMISSIONS_BREAKPOINTS.items()
}
historical_emissions = emissions.pop("historical")
scenario_emissions = emissions

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
# `build_model` wraps the builder chain, because the historical run and every scenario use it.
# The carbon cycle reads the surface air temperature change,
# but `alpha_temperature=0` turns that feedback off, so a constant zero series stands in for it.
#
# To add a temperature response, you would add a climate component such as TwoLayer,
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
# The historical time axis uses decadal steps before 1900, 5-yearly steps to 2000 and annual steps to 2015.
# The future runs use their own annual axis, so the two periods can have different resolutions.

# %%
historical_time_axis = TimeAxis.from_values(
    np.concatenate(
        [
            np.arange(T_START, 1900.0, 10.0),
            np.arange(1900.0, 2000.0, 5.0),
            np.arange(2000.0, T_BRANCH + 1, 1.0),
        ]
    )
)

historical_model = build_model(
    historical_time_axis, historical_emissions, PRE_INDUSTRIAL
)
historical_model.run()

# %% [markdown]
# `to_toml()` serialises the whole model, so the historical end state can be archived or inspected later.
# Its top-level tables show what it holds.
# See [State Serialisation](../state_serialisation/) for restoring a model from this string.

# %%
historical_state = historical_model.to_toml()
tables = [
    line.strip("[]") for line in historical_state.splitlines() if line.startswith("[")
]
list(dict.fromkeys(table.split(".")[0] for table in tables))

# %% [markdown]
# ## Step 2: Read the Branch-Point State
#
# The state variables at 2015 come from the historical results.
# They are the same variables that `PRE_INDUSTRIAL` set for 1750.

# %%
historical_results = historical_model.timeseries()
branch_state = {
    name: historical_results.get_timeseries_by_name(name).at_time(T_BRANCH)
    for name in PRE_INDUSTRIAL
}
branch_state

# %% [markdown]
# ## Step 3: Run the Scenarios
#
# Each scenario gets a fresh model on the future time axis, starting from `branch_state`.
#
# A model restored with `Model.from_toml()` keeps its original time axis and prescribed data,
# so it cannot be extended with new scenario data.
# Rebuilding from the state variables works here because these components keep no
# internal state between steps.
# Components with internal state store it in the `component_states` section of the TOML,
# which this approach does not carry over.

# %%
future_time_axis = TimeAxis.from_values(np.arange(T_BRANCH, T_END + 1, 1.0))

scenario_results = {}
for scenario, co2_emissions in scenario_emissions.items():
    model = build_model(future_time_axis, co2_emissions, branch_state)
    model.run()
    scenario_results[scenario] = model.timeseries()

# %% [markdown]
# ## Step 4: Compare the Scenarios
#
# A hidden helper converts each `TimeseriesCollection` into a pandas DataFrame
# with a (variable, unit, scenario) index, and stacks the four runs into `all_runs`.


# %% tags=["remove_input"]
import pandas as pd

from rscm.core import TimeseriesCollection


def timeseries_to_dataframe(
    collection: TimeseriesCollection, scenario: str
) -> pd.DataFrame:
    """Convert a TimeseriesCollection to a DataFrame indexed by (variable, unit, scenario)."""
    rows = {}
    for name in collection.names():
        ts = collection.get_timeseries_by_name(name)
        rows[(name, ts.unit, scenario)] = pd.Series(
            ts.values(), index=ts.time_axis.values()
        )
    df = pd.DataFrame(rows).T
    df.index.names = ["variable", "unit", "scenario"]
    return df


all_runs = pd.concat(
    [
        timeseries_to_dataframe(historical_results, "historical"),
        *(timeseries_to_dataframe(r, s) for s, r in scenario_results.items()),
    ]
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
# - **State extraction**: Read the branch-point values from `model.timeseries()`
#   to initialise the scenario runs.
# - **Component reuse**: These components hold only parameters, so they can be rebuilt for each scenario.
# - **Time axis management**: Historical and future periods can have different temporal resolutions.
#
# ## Next Steps
#
# - Add more greenhouse gases (CH4, N2O) using additional components.
# - Include aerosol forcing for more realistic scenarios.
# - Load real SSP data from scenario databases.
# - Add uncertainty analysis by running parameter ensembles.
