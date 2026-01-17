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
# This tutorial demonstrates how to build a complete climate modelling pipeline that:
#
# 1. Runs a historical spin-up period
# 2. Saves the model state at a branch point (e.g., 2015)
# 3. Runs multiple future scenarios from that common starting point
# 4. Compares and visualises results across scenarios
#
# This pattern is essential for climate scenario analysis where you want to compare
# different emissions pathways (e.g., SSP scenarios) while ensuring they all start
# from the same historical state.
#
# ## Related Resources
#
# - [Coupled Models](coupled_model.py): Building multi-component models
# - [State Serialisation](state_serialisation.py): Saving and loading model state
# - [Key Concepts](../key_concepts.md): Core RSCM architecture

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)

# %% [markdown]
# ## Define Scenario Data
#
# In a real workflow, you would load scenario data from files (e.g., SSP or RCP
# emissions from a database). Here we define simplified scenarios programmatically
# to illustrate the pattern.
#
# Our scenarios represent different emissions pathways:
#
# - **Historical**: Observed emissions from 1750-2015
# - **SSP1-1.9**: Rapid decarbonisation (net-zero by 2050)
# - **SSP2-4.5**: Middle-of-the-road pathway
# - **SSP5-8.5**: Fossil-fuelled development (high emissions)


# %%
def create_historical_emissions(t_start: float, t_branch: float) -> Timeseries:
    """
    Create historical CO2 emissions timeseries.

    This is a simplified representation of historical emissions growth.
    Real applications would load from observational datasets.

    Parameters
    ----------
    t_start
        Start year of the timeseries
    t_branch
        End year (typically 2015 for SSP scenarios)

    Returns
    -------
        Timeseries of emissions in GtC/yr
    """
    # Simplified historical: exponential growth from pre-industrial
    times = np.array([t_start, 1850.0, 1950.0, 2000.0, t_branch])
    # Emissions in GtC/yr (approximate historical values)
    values = np.array([0.0, 0.1, 1.5, 6.5, 10.0])

    return Timeseries(
        values,
        TimeAxis.from_values(times),
        "GtC / yr",
        InterpolationStrategy.Linear,
    )


# %%
def create_scenario_emissions(
    t_branch: float, t_end: float, scenario: str
) -> Timeseries:
    """
    Create future emissions timeseries for a given scenario.

    Parameters
    ----------
    t_branch
        Branch year from historical (scenario start)
    t_end
        End year of projections
    scenario
        Scenario identifier (ssp119, ssp245, ssp585)

    Returns
    -------
        Timeseries of emissions in GtC/yr
    """
    times = np.array([t_branch, 2030.0, 2050.0, 2075.0, t_end])

    scenarios = {
        # SSP1-1.9: Rapid decarbonisation
        "ssp119": np.array([10.0, 7.0, 0.0, -2.0, -3.0]),
        # SSP2-4.5: Middle of the road
        "ssp245": np.array([10.0, 11.0, 10.0, 8.0, 5.0]),
        # SSP5-8.5: High emissions
        "ssp585": np.array([10.0, 14.0, 18.0, 20.0, 22.0]),
    }

    if scenario not in scenarios:
        msg = f"Unknown scenario: {scenario}. Choose from {list(scenarios)}"
        raise ValueError(msg)

    return Timeseries(
        scenarios[scenario],
        TimeAxis.from_values(times),
        "GtC / yr",
        InterpolationStrategy.Linear,
    )


# %% [markdown]
# ## Configure Model Components
#
# We build a coupled model with:
#
# 1. **Carbon Cycle**: Converts emissions to atmospheric CO2 concentration
# 2. **CO2 ERF**: Calculates effective radiative forcing from CO2
#
# The components are connected through their inputs and outputs:
#
# ```
# Emissions -> [Carbon Cycle] -> CO2 Concentration -> [CO2 ERF] -> Forcing
# ```
#
# Note: To add temperature response, you would include a climate component
# (e.g., TwoLayer) that takes total radiative forcing as input. This requires
# an aggregation component to sum forcings from multiple species (CO2, CH4, etc.)
# into a single "Effective Radiative Forcing" input.


# %%
def create_component_builders():
    """
    Create builders for the model components.

    Returns
    -------
        Tuple of (carbon_cycle, co2_erf) component builders
    """
    # Carbon cycle parameters
    carbon_cycle = CarbonCycleBuilder.from_parameters(
        dict(
            tau=30.0,  # Atmospheric residence time (years)
            conc_pi=280.0,  # Pre-industrial CO2 (ppm)
            alpha_temperature=0.0,  # Temperature feedback (disabled for simplicity)
        )
    ).build()

    # CO2 forcing parameters
    co2_erf = CO2ERFBuilder.from_parameters(
        dict(
            erf_2xco2=3.7,  # Forcing for CO2 doubling (W/m2)
            conc_pi=280.0,  # Pre-industrial CO2 (ppm)
        )
    ).build()

    return carbon_cycle, co2_erf


# %%
def get_initial_values() -> dict:
    """
    Get initial values for state variables.

    These represent pre-industrial equilibrium conditions.
    """
    return {
        "Cumulative Land Uptake": 0.0,
        "Cumulative Emissions|CO2": 0.0,
        "Atmospheric Concentration|CO2": 280.0,  # Pre-industrial
    }


# %% [markdown]
# ## Step 1: Run Historical Period
#
# First, we run the model over the historical period (1750-2015).
# This establishes the climate state from which all future scenarios branch.

# %%
# Time configuration
T_START = 1750.0
T_BRANCH = 2015.0  # Historical/future branch point
T_END = 2100.0

# %%
# Create historical time axis with coarser steps early, finer steps later
historical_time_axis = TimeAxis.from_values(
    np.concatenate(
        [
            np.arange(T_START, 1900.0, 10.0),  # Decadal 1750-1900
            np.arange(1900.0, 2000.0, 5.0),  # 5-yearly 1900-2000
            np.arange(2000.0, T_BRANCH + 1, 1.0),  # Annual 2000-2015
        ]
    )
)

print(f"Historical period: {T_START} to {T_BRANCH}")
print(f"Number of timesteps: {len(historical_time_axis.values())}")

# %%
# Create historical emissions
historical_emissions = create_historical_emissions(T_START, T_BRANCH)

# Plot the historical emissions
plt.figure(figsize=(8, 4))
# Interpolate to model time axis for plotting
times = historical_time_axis.values()
emissions_interp = [historical_emissions.at_time(t) for t in times]
plt.plot(times, emissions_interp)
plt.xlabel("Year")
plt.ylabel("Emissions (GtC/yr)")
plt.title("Historical CO2 Emissions")
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Build and run historical model
carbon_cycle, co2_erf = create_component_builders()

# Surface temperature is needed as an exogenous input for carbon cycle feedback
# Since we have feedback disabled (alpha_temperature=0), we just need a placeholder
surface_temp_exog = Timeseries(
    np.array([0.0]),
    TimeAxis.from_bounds(np.array([T_START, T_END])),
    "K",
    InterpolationStrategy.Previous,
)

historical_model = (
    ModelBuilder()
    .with_time_axis(historical_time_axis)
    .with_rust_component(carbon_cycle)
    .with_rust_component(co2_erf)
    .with_exogenous_variable("Emissions|CO2|Anthropogenic", historical_emissions)
    .with_exogenous_variable("Surface Temperature", surface_temp_exog)
    .with_initial_values(get_initial_values())
).build()

# %%
# Run the historical period
historical_model.run()

print(f"Historical run complete. Final time: {historical_model.current_time()}")

# %%
# Save the historical end-state for branching
historical_state = historical_model.to_toml()

# Preview the state (truncated)
print("Model state preview:")
print(historical_state[:2000] + "\n...")

# %% [markdown]
# ## Step 2: Define Future Scenarios
#
# We create emissions pathways for each scenario that branch from the historical
# end-state. The key insight is that we can restore the model from the saved
# state and run with different future emissions.

# %%
# Define scenarios to run
SCENARIOS = ["ssp119", "ssp245", "ssp585"]

# Create scenario emissions
scenario_emissions = {}
for scenario in SCENARIOS:
    scenario_emissions[scenario] = create_scenario_emissions(T_BRANCH, T_END, scenario)

# %%
# Plot all scenario emissions
plt.figure(figsize=(10, 5))

# Historical
hist_times = np.arange(T_START, T_BRANCH + 1, 5)
hist_values = [historical_emissions.at_time(t) for t in hist_times]
plt.plot(hist_times, hist_values, "k-", linewidth=2, label="Historical")

# Scenarios
colors = {"ssp119": "green", "ssp245": "orange", "ssp585": "red"}
future_times = np.arange(T_BRANCH, T_END + 1, 5)

for scenario, emissions in scenario_emissions.items():
    values = [emissions.at_time(t) for t in future_times]
    plt.plot(future_times, values, color=colors[scenario], linewidth=2, label=scenario)

plt.axvline(T_BRANCH, color="gray", linestyle="--", alpha=0.5, label="Branch point")
plt.xlabel("Year")
plt.ylabel("Emissions (GtC/yr)")
plt.title("CO2 Emissions Scenarios")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Step 3: Run Future Scenarios
#
# For each scenario, we:
#
# 1. Restore the model from the historical end-state
# 2. Replace the emissions timeseries with the scenario data
# 3. Run to 2100
# 4. Collect results
#
# Note: The current RSCM API restores the full model including its time axis.
# For scenario runs, we rebuild the model with a new time axis that continues
# from the branch point.


# %%
def run_scenario(scenario_name: str, emissions: Timeseries) -> TimeseriesCollection:
    """
    Run a single scenario from the historical branch point.

    Parameters
    ----------
    scenario_name
        Name of the scenario (for logging)
    emissions
        Emissions timeseries for this scenario

    Returns
    -------
        TimeseriesCollection with all model outputs
    """
    # Create time axis for future period
    # Include overlap with historical end for continuity
    future_time_axis = TimeAxis.from_values(np.arange(T_BRANCH, T_END + 1, 1.0))

    # Rebuild components (they are stateless, just parameters)
    carbon_cycle, co2_erf = create_component_builders()

    # Get the final state values from historical run
    historical_results = historical_model.timeseries()

    # Extract final values for state variables
    initial_values = {
        "Cumulative Land Uptake": historical_results.get_timeseries_by_name(
            "Cumulative Land Uptake"
        ).at_time(T_BRANCH),
        "Cumulative Emissions|CO2": historical_results.get_timeseries_by_name(
            "Cumulative Emissions|CO2"
        ).at_time(T_BRANCH),
        "Atmospheric Concentration|CO2": historical_results.get_timeseries_by_name(
            "Atmospheric Concentration|CO2"
        ).at_time(T_BRANCH),
    }

    # Build model for scenario run
    model = (
        ModelBuilder()
        .with_time_axis(future_time_axis)
        .with_rust_component(carbon_cycle)
        .with_rust_component(co2_erf)
        .with_exogenous_variable("Emissions|CO2|Anthropogenic", emissions)
        .with_exogenous_variable("Surface Temperature", surface_temp_exog)
        .with_initial_values(initial_values)
    ).build()

    # Run
    model.run()
    print(f"Scenario {scenario_name} complete")

    return model.timeseries()


# %%
# Run all scenarios
scenario_results = {}

for scenario in SCENARIOS:
    scenario_results[scenario] = run_scenario(scenario, scenario_emissions[scenario])

# %% [markdown]
# ## Step 4: Post-Processing and Comparison
#
# Now we combine results from all scenarios for comparison. We use pandas DataFrames
# with multi-indexes for convenient handling of multi-scenario timeseries data.


# %%
def timeseries_to_dataframe(
    ts_collection: TimeseriesCollection,
    time_axis: TimeAxis,
    scenario: str,
) -> pd.DataFrame:
    """
    Convert RSCM TimeseriesCollection to pandas DataFrame.

    Parameters
    ----------
    ts_collection
        RSCM timeseries collection
    time_axis
        Time axis for the data
    scenario
        Scenario name for metadata

    Returns
    -------
        DataFrame with (variable, unit, scenario) multi-index
    """
    data = []
    index_tuples = []

    for name in ts_collection.names():
        ts = ts_collection.get_timeseries_by_name(name)
        index_tuples.append((name, ts.units, scenario))
        data.append(ts.values())

    return pd.DataFrame(
        data,
        columns=time_axis.values(),
        index=pd.MultiIndex.from_tuples(
            index_tuples, names=["variable", "unit", "scenario"]
        ),
    )


# %%
# Convert historical results
historical_df = timeseries_to_dataframe(
    historical_model.timeseries(), historical_time_axis, "historical"
)

# Convert scenario results
scenario_dfs = []
future_time_axis = TimeAxis.from_values(np.arange(T_BRANCH, T_END + 1, 1.0))

for scenario, results in scenario_results.items():
    df = timeseries_to_dataframe(results, future_time_axis, scenario)
    scenario_dfs.append(df)

# Combine all runs - use concat with outer join to handle different time columns
all_runs = pd.concat([historical_df, *scenario_dfs], axis=0)
print(f"Combined dataset shape: {all_runs.shape}")
all_runs

# %% [markdown]
# ### Plot Key Variables
#
# Compare scenarios across key climate variables:
#
# - CO2 concentration
# - Effective radiative forcing

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

scenario_colors = {
    "historical": "black",
    "ssp119": "green",
    "ssp245": "orange",
    "ssp585": "red",
}


def plot_variable(
    ax: plt.Axes, df: pd.DataFrame, variable: str, ylabel: str, title: str
):
    """Plot a variable across all scenarios."""
    # Filter for the specific variable
    var_data = df.loc[df.index.get_level_values("variable") == variable]

    for idx in var_data.index:
        scenario = idx[2]  # (variable, unit, scenario)
        values = var_data.loc[idx].dropna()
        ax.plot(
            values.index,
            values.values,
            label=scenario,
            color=scenario_colors.get(scenario, "gray"),
            linewidth=2 if scenario != "historical" else 1.5,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# CO2 Concentration
plot_variable(
    axes[0],
    all_runs,
    "Atmospheric Concentration|CO2",
    "CO2 (ppm)",
    "Atmospheric CO2 Concentration",
)
axes[0].axvline(T_BRANCH, color="gray", linestyle="--", alpha=0.5)

# Effective Radiative Forcing
plot_variable(
    axes[1],
    all_runs,
    "Effective Radiative Forcing|CO2",
    "ERF (W/m2)",
    "CO2 Effective Radiative Forcing",
)
axes[1].axvline(T_BRANCH, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### End-of-Century Summary
#
# Extract key metrics at 2100 for comparison.

# %%
# Get 2100 values for each scenario
summary_data = []

for scenario in SCENARIOS:
    # Filter by scenario using multi-index
    scenario_mask = all_runs.index.get_level_values("scenario") == scenario
    scenario_data = all_runs.loc[scenario_mask]

    # Get CO2 concentration at 2100
    co2_var = "Atmospheric Concentration|CO2"
    co2_row = scenario_data.loc[
        scenario_data.index.get_level_values("variable") == co2_var
    ]
    co2 = co2_row[2100.0].values[0]

    # Get ERF at 2100
    erf_var = "Effective Radiative Forcing|CO2"
    erf_row = scenario_data.loc[
        scenario_data.index.get_level_values("variable") == erf_var
    ]
    erf = erf_row[2100.0].values[0]

    summary_data.append(
        {
            "Scenario": scenario,
            "CO2 (ppm)": f"{co2:.0f}",
            "ERF (W/m2)": f"{erf:.2f}",
        }
    )

summary_df = pd.DataFrame(summary_data)
print("\nEnd-of-Century (2100) Summary:")
print(summary_df.to_string(index=False))

# %% [markdown]
# ## Exporting Results
#
# For further analysis or archival, results can be exported in various formats.
# The pandas DataFrame with multi-index preserves all metadata.

# %%
# Export to CSV (uncomment to use)
# all_runs.to_csv("scenario_results.csv")

# Export scenario comparison table
# summary_df.to_csv("scenario_summary.csv", index=False)

# %% [markdown]
# ## Summary
#
# This tutorial demonstrated a complete scenario pipeline workflow:
#
# 1. **Historical spin-up**: Run the model from pre-industrial to a branch point
# 2. **State capture**: Save model state using `to_toml()` for reproducibility
# 3. **Scenario branching**: Create multiple futures from the same starting point
# 4. **Results collection**: Gather outputs using `TimeseriesCollection`
# 5. **Post-processing**: Convert to pandas DataFrames for analysis and visualisation
# 6. **Comparison**: Plot and tabulate scenario differences
#
# ### Key Patterns
#
# - **State extraction**: Get final values from `model.timeseries()` for initialising
#   branch runs
# - **Component reuse**: Components are stateless; rebuild for each scenario
# - **Time axis management**: Historical and future periods can have different
#   temporal resolutions
# - **Metadata handling**: Use pandas DataFrames with multi-indexes for scenario data
#
# ### Next Steps
#
# - Add more greenhouse gases (CH4, N2O) using additional components
# - Include aerosol forcing for more realistic scenarios
# - Load real SSP/RCP data from scenario databases
# - Add uncertainty analysis by running parameter ensembles
