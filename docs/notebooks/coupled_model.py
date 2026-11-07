# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: rscm (3.13.6)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Coupled Models
#
# This notebook builds coupled climate models by combining components that exchange state.
#
# ## Overview
#
# Modelling a complex system usually needs several components coupled together.
# Each component models one aspect of the Earth system with a set of equations and known inputs and outputs.
# Those inputs come from other components or are prescribed as prescribed variables.
#
# ## Related Resources
#
# - [Components in Python](../component_python/): Creating Python components
# - [Components in Rust](../component_rust/): Creating Rust components
# - [Key Concepts](../../key_concepts/): Core RSCM architecture
# - [State Serialisation](../state_serialisation/): Saving and loading model state

# %%
import numpy as np

from rscm.components import CarbonCycleBuilder, CO2ERFBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    VariableSchema,
)
from rscm.two_layer import TwoLayerBuilder

# %% [markdown]
# ## Building a Model
#
# The `ModelBuilder` class constructs coupled models from components.
# It captures the model's requirements (components, time axis, prescribed variables)
# before creating a concrete model.
# This makes model configurations easy to share and extend.

# %%
model_builder = ModelBuilder()

# %% tags=["remove_input"]
import matplotlib.pyplot as plt
import pandas as pd
import pydot
from IPython.display import SVG, display


def show_graph(model):
    """Render the component graph, labelling each node with its component name."""
    graph = pydot.graph_from_dot_data(model.as_dot())[0]
    for node in graph.get_nodes():
        label = node.get_label()
        if label:
            node.set_label(label.strip('"').split(" {")[0])
    display(SVG(graph.create_svg()))


def as_dataframe(collection: TimeseriesCollection, time_axis: TimeAxis) -> pd.DataFrame:
    """Convert a timeseries collection to a DataFrame indexed by (variable, unit)."""
    series = [collection.get_timeseries_by_name(name) for name in collection.names()]
    index = pd.MultiIndex.from_tuples(
        [(name, ts.unit) for name, ts in zip(collection.names(), series)],
        names=["variable", "unit"],
    )
    return pd.DataFrame(
        [ts.values() for ts in series], columns=time_axis.values(), index=index
    )


def plot_variables(
    results: pd.DataFrame, variables: list[str], steps: tuple[str, ...] = ()
):
    """Plot each variable on its own axis, drawing the variables in `steps` as step functions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, var in zip(axes.flat, variables):
        data = results.xs(var, level="variable")
        drawstyle = "steps-post" if var in steps else "default"
        ax.plot(data.columns, data.iloc[0], drawstyle=drawstyle)
        ax.set_title(var)
        ax.set_ylabel(data.index[0])
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    fig.tight_layout()
    plt.show()


# %% [markdown]
# ### Time axis
#
# The time axis defines the steps on which the model is solved.
# Components only exchange information at the end of a step.
#
# The steps do not need to be the same size across the axis.
# Longer steps suit a preindustrial spin-up, and shorter steps suit the period of rising emissions.
# Here we use 5-year steps until 2015 and annual steps after.

# %%
t_initial = 1750.0
t_final = 2100.0

time_axis = TimeAxis.from_values(
    np.concat([np.arange(t_initial, 2015.0, 5), np.arange(2015.0, t_final + 1, 1.0)])
)
print(
    f"{len(time_axis)} time points from {time_axis.values()[0]} to {time_axis.values()[-1]}"
)

# %% [markdown]
# Each `with_*` method updates the builder and returns it, so calls can be chained.

# %%
model_builder = model_builder.with_time_axis(time_axis)

# %% [markdown]
# ### Components
#
# A model consists of one or more components.
#
# Components can be implemented in either Rust (for performance) or Python (for rapid prototyping).
# See [Components in Rust](../component_rust/) and [Components in Python](../component_python/)
# for implementation details.
#
# Below we create a model with a basic carbon cycle and a CO2 ERF calculation.
# The CO2 ERF component calculates the radiative forcing from the CO2 concentration
# that the carbon cycle component produces.

# %%
tau = 20.3
conc_pi = 280.0
erf_2xco2 = 4.0
alpha_temperature = 0.0  # No temperature feedback

co2_erf_component = CO2ERFBuilder.from_parameters(
    dict(erf_2xco2=erf_2xco2, conc_pi=conc_pi)
).build()
carbon_cycle_component = CarbonCycleBuilder.from_parameters(
    dict(tau=tau, conc_pi=conc_pi, alpha_temperature=alpha_temperature)
).build()

# %%
model_builder = model_builder.with_component(carbon_cycle_component).with_component(
    co2_erf_component
)

# %% [markdown]
# ## Prescribed Variables
#
# Variables that no component produces must be provided as prescribed (external) data.
# Each prescribed variable is a `Timeseries` with values, a time axis, units and an interpolation strategy.
#
# Here CO2 emissions step from 0 to 10 GtC/yr in 1850.

# %%
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

# %% [markdown]
# With the `Previous` strategy each value holds until the next point, so the emissions form a step.

# %% tags=["remove_input"]
fig, ax = plt.subplots(figsize=(8, 3))
ax.stairs(emissions.values(), emissions.time_axis.bounds())
ax.set_xlabel("Year")
ax.set_ylabel("Emissions (GtC / yr)")
plt.show()

# %% [markdown]
# Components read prescribed data at the model's time points.
# The interpolation strategy decides the value between the timeseries' own points:
#
# - `Linear` interpolates between neighbouring points.
# - `Previous` holds the value of the preceding point, giving a step function.
# - `Next` takes the value of the following point.
#
# `at_time()` returns the interpolated value at a given time.

# %%
for strategy in [
    InterpolationStrategy.Linear,
    InterpolationStrategy.Previous,
    InterpolationStrategy.Next,
]:
    ts = Timeseries(emissions.values(), emissions.time_axis, emissions.unit, strategy)
    print(f"{strategy}: {ts.at_time(1825.0):.2f}")

# %% [markdown]
# The surface temperature is held at a constant 0.42 K.

# %%
surface_temp = Timeseries(
    np.asarray([0.42]),
    TimeAxis.from_bounds(np.asarray([t_initial, t_final])),
    "K",
    InterpolationStrategy.Previous,
)

model_builder = model_builder.with_prescribed_variable(
    "Surface Temperature", surface_temp
).with_prescribed_variable("Emissions|CO2|Anthropogenic", emissions)

# %% [markdown]
# ## Initial Values
#
# Components that track state (using `#[states(...)]` in Rust or `State()` in Python) need initial values.
# These values are used at the first step.
# Later steps use the state from the previous step.

# %%
initial_state = {
    "Cumulative Land Uptake": 0.0,
    "Cumulative Emissions|CO2": 0.0,
    "Atmospheric Concentration|CO2": 300.0,
}

model_builder = model_builder.with_initial_values(initial_state)

# %% [markdown]
# ## Build the Model
#
# Building the model generates a directed graph of component relationships.
# Each step, the model solves the components in topological order of this graph,
# so every producer runs before the components that consume its outputs.
# If required information such as an initial value or prescribed data is missing,
# the build step raises an exception explaining what is needed.

# %%
model = model_builder.build()

# %% [markdown]
# ### Inspecting the Execution Graph
#
# After building a model, `coupling()` shows the order the runtime steps components in,
# variable flow, where each input comes from and whether each variable is a point or an average.
# Output formats are `"rich"` (coloured terminal, default), `"plain"` (no colours) and `"json"`.

# %%
print(model.coupling())

# %% [markdown]
# The same model can be built in one chain:

# %%
model = (
    ModelBuilder()
    .with_component(carbon_cycle_component)
    .with_component(co2_erf_component)
    .with_time_axis(time_axis)
    .with_prescribed_variable("Surface Temperature", surface_temp)
    .with_prescribed_variable("Emissions|CO2|Anthropogenic", emissions)
    .with_initial_values(initial_state)
).build()

# %% [markdown]
# `model.as_dot()` returns the component graph in Graphviz DOT format.
# Each node is a component and each edge is a variable passed between them.
# `NullComponent` is an empty root that links the components without upstream dependencies.

# %% tags=["remove_input"]
show_graph(model)

# %% [markdown]
# ## Run the model
#
# Once we have a concrete model we can solve it.
# You can either step through the model one step at a time or run all steps at once.
# Both `step()` and `run()` raise a `RuntimeError` if a component fails.
# See [Model Debugging and Inspection](../debugging_inspection/) for handling these errors.

# %%
print(model.current_time_bounds())
model.step()
print(model.current_time_bounds())

# %% [markdown]
# `model.timeseries()` returns the results as a `TimeseriesCollection`.
# A short `as_dataframe` helper (in the downloadable notebook) converts it to a pandas DataFrame.
# After one step only the first time point is filled.

# %%
results = as_dataframe(model.timeseries(), time_axis)
results.iloc[:, :3]

# %% [markdown]
# `run()` solves the remaining steps.

# %%
model.run()
results = as_dataframe(model.timeseries(), time_axis)

# %% tags=["remove_input"]
plot_variables(
    results,
    [
        "Emissions|CO2|Anthropogenic",
        "Atmospheric Concentration|CO2",
        "Effective Radiative Forcing|CO2",
        "Surface Temperature",
    ],
    steps=("Emissions|CO2|Anthropogenic",),
)

# %% [markdown]
# ## Complex Coupled Model with Feedback
#
# The example above uses a prescribed surface temperature.
# In reality, temperature responds to radiative forcing,
# and temperature changes can affect carbon cycle processes
# (for example soil respiration and ocean solubility).
#
# Below we build a coupled model with this feedback:
#
# ```
# Emissions → Carbon Cycle → CO2 Concentration → CO2 ERF → Two-Layer → Temperature
#                  ↑                                                          │
#                  └──────────────────── feedback ────────────────────────────┘
# ```
#
# 1. **Carbon Cycle**: converts emissions to atmospheric CO2 concentration.
# 2. **CO2 ERF**: calculates radiative forcing from the CO2 concentration.
# 3. **Two-Layer Model**: computes surface temperature from the radiative forcing.
# 4. **Temperature Feedback**: warming lengthens the carbon cycle's CO2 lifetime,
#    so more CO2 stays in the atmosphere.

# %%
co2_erf = CO2ERFBuilder.from_parameters(
    dict(
        erf_2xco2=3.7,  # Forcing from CO2 doubling (W/m²)
        conc_pi=278.0,  # Pre-industrial CO2 (ppm)
    )
).build()

two_layer = TwoLayerBuilder.from_parameters(
    dict(
        lambda0=1.1,  # Climate feedback parameter (W/m²/K)
        a=0.0,  # Nonlinear feedback coefficient
        efficacy=1.3,  # Ocean heat uptake efficacy
        eta=0.7,  # Heat exchange coefficient (W/m²/K)
        heat_capacity_surface=8.0,  # Surface heat capacity (W yr/m²/K)
        heat_capacity_deep=100.0,  # Deep ocean heat capacity (W yr/m²/K)
    )
).build()

# %% [markdown]
# ### Emissions Scenario
#
# The emissions scenario has three phases, interpolated linearly between points:
#
# - Pre-industrial period (1750 to 1850): near-zero emissions.
# - Historical growth (1850 to 2020): increasing to 10 GtC/yr.
# - Future scenario (2020 to 2100): peak and decline.

# %%
years = np.array([1750.0, 1850.0, 1950.0, 2000.0, 2020.0, 2050.0, 2100.0])
emission_values = np.array([0.0, 0.5, 3.0, 7.0, 10.0, 5.0, 1.0])  # GtC/yr

emissions_scenario = Timeseries(
    emission_values,
    TimeAxis.from_bounds(np.concatenate([years, [2101.0]])),  # bounds need n+1 values
    "GtC / yr",
    InterpolationStrategy.Linear,
)

# %% [markdown]
# ### Variable Schema for ERF Aggregation
#
# The CO2 ERF component outputs `Effective Radiative Forcing|CO2`,
# but the Two-Layer model expects `Effective Radiative Forcing` as input.
# A `VariableSchema` aggregates the CO2 forcing into the total ERF.
#
# Other forcing components (CH4, N2O, aerosols) can later contribute to the total through the same schema.
# See [Variable Schemas](../variable_schema/) for more.

# %%
erf_schema = VariableSchema()

erf_schema.add_variable("Emissions|CO2|Anthropogenic", "GtC / yr")
erf_schema.add_variable("Surface Temperature", "K")
erf_schema.add_variable("Atmospheric Concentration|CO2", "ppm")
erf_schema.add_variable("Cumulative Land Uptake", "Gt C")
erf_schema.add_variable("Cumulative Emissions|CO2", "Gt C")
erf_schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
erf_schema.add_variable("Deep Ocean Temperature", "K")

erf_schema.add_aggregate(
    "Effective Radiative Forcing",
    "W/m^2",
    "sum",
    ["Effective Radiative Forcing|CO2"],
)

# %% [markdown]
# ### Build the Feedback-Coupled Model
#
# `Surface Temperature` is not a prescribed variable here, because the Two-Layer model produces it.
# The model builder resolves the dependency graph, whatever order the components are added in.
#
# Temperature feeds back on the carbon cycle, so the components form a feedback loop.
# `with_lag_for("Surface Temperature", "CarbonCycle")` breaks it:
# the carbon cycle reads the temperature the Two-Layer model wrote in the previous step.
# The feedback loop therefore closes across steps rather than within one.
# Without the lag, `build()` fails and suggests this call.
#
# The build is wrapped in a function of the carbon cycle's temperature sensitivity,
# so we can compare against a run without feedback later.


# %%
def build_feedback_model(alpha_temperature: float):
    """Build the feedback-coupled model with the given temperature sensitivity."""
    carbon_cycle = CarbonCycleBuilder.from_parameters(
        dict(
            tau=25.0,  # Baseline atmospheric lifetime (years)
            conc_pi=278.0,  # Pre-industrial CO2 (ppm)
            alpha_temperature=alpha_temperature,  # Temperature sensitivity (1/K)
        )
    ).build()

    return (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_schema(erf_schema)
        .with_component(carbon_cycle)
        .with_component(co2_erf)
        .with_component(two_layer)
        .with_lag_for("Surface Temperature", "CarbonCycle")
        .with_prescribed_variable("Emissions|CO2|Anthropogenic", emissions_scenario)
        .with_initial_values(
            {
                "Cumulative Land Uptake": 0.0,
                "Cumulative Emissions|CO2": 0.0,
                "Atmospheric Concentration|CO2": 278.0,
                "Surface Temperature": 0.0,  # Anomaly from pre-industrial
                "Deep Ocean Temperature": 0.0,
            }
        )
    ).build()


feedback_model = build_feedback_model(alpha_temperature=0.1)

# %% [markdown]
# ### Visualise the Dependency Graph
#
# `coupling()` tags `Surface Temperature` as `(lagged from TwoLayer)` for the carbon cycle,
# because it reads the Two-Layer model's value from the previous step.
# Its "Lagged Edges" section lists the same edge.

# %%
print(feedback_model.coupling())

# %% [markdown]
# The component graph shows:
#
# - **NullComponent**: the root.
# - **CarbonCycle**: takes emissions and temperature, outputs CO2 concentration.
# - **CO2ERF**: takes CO2 concentration, outputs radiative forcing.
# - **AggregatorComponent**: sums the forcing contributors declared in the schema.
# - **TwoLayer**: takes radiative forcing, outputs surface temperature.

# %% tags=["remove_input"]
show_graph(feedback_model)

# %%
feedback_model.run()
feedback_results = as_dataframe(feedback_model.timeseries(), time_axis)

# %% tags=["remove_input"]
plot_variables(
    feedback_results,
    [
        "Emissions|CO2|Anthropogenic",
        "Atmospheric Concentration|CO2",
        "Effective Radiative Forcing",
        "Surface Temperature",
    ],
)

# %% [markdown]
# ### Comparing With and Without Temperature Feedback
#
# To see the effect of the temperature feedback,
# we build the same model with `alpha_temperature=0` and compare.

# %%
no_feedback_model = build_feedback_model(alpha_temperature=0.0)
no_feedback_model.run()
no_feedback_results = as_dataframe(no_feedback_model.timeseries(), time_axis)

# %% tags=["remove_input"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, var in zip(axes, ["Atmospheric Concentration|CO2", "Surface Temperature"]):
    for results_df, label, style in [
        (feedback_results, "With feedback", "-"),
        (no_feedback_results, "Without feedback", "--"),
    ]:
        data = results_df.xs(var, level="variable")
        ax.plot(data.columns, data.iloc[0], style, label=label)
    ax.set_title(var)
    ax.set_xlabel("Year")
    ax.set_ylabel(data.index[0])
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# The temperature feedback amplifies warming by reducing the efficiency of natural carbon sinks
# at higher temperatures.
# With the feedback, the model gives higher CO2 concentrations, more radiative forcing
# and higher surface temperatures.
#
# ## Summary
#
# - `ModelBuilder` collects the time axis, components, prescribed data and initial values,
#   and `build()` turns them into a model.
# - The interpolation strategy sets how prescribed data is read between its points.
# - `coupling()` and `as_dot()` show what the builder resolved.
# - A component that reads another's output from the previous step closes a feedback loop across steps.
#
# For more on writing components, see [Components in Python](../component_python/)
# and [Components in Rust](../component_rust/).
