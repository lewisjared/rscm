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
# # Variable Schemas and Aggregation
#
# This tutorial shows how to aggregate several component outputs into a derived value,
# such as the total `Effective Radiative Forcing`.
# You use a `VariableSchema` to declare variables and define how they combine.
#
# ## Why Aggregation?
#
# Climate models compute forcing from several sources: CO2, CH4, N2O, aerosols and more.
# The climate response needs the total forcing, the sum of these contributions.
# Without aggregation you would write a component that reads every forcing variable and sums them,
# and update it whenever you add or remove a forcing source.
#
# A `VariableSchema` declares the aggregation rules at the model level instead.
# The framework adds virtual components that compute the aggregates.
#
# ## Prerequisites
#
# This tutorial assumes you can:
#
# - Build models with `ModelBuilder` (see [Coupled Models](../coupled_model/)).
# - Write Python components (see [Components in Python](../component_python/)).
#
# The [Key Concepts](../../key_concepts/) page describes the core RSCM architecture.

# %%
import numpy as np
import pandas as pd

from rscm.component import Component, Input, Output
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableSchema,
)

# %% [markdown]
# ## Creating Simple Forcing Components
#
# Two simple components calculate radiative forcing from greenhouse gas concentrations.
# A real model would use proper radiative transfer relationships.
# These use simplified ones.


# %%
class CO2Forcing(Component):
    """CO2 forcing from concentration: ERF = alpha * ln(C / C_pi)."""

    concentration = Input("Atmospheric Concentration|CO2", unit="ppm")
    forcing = Output("Effective Radiative Forcing|CO2", unit="W/m^2")

    def __init__(self, alpha: float = 5.35, c_pi: float = 280.0):
        self.alpha = alpha
        self.c_pi = c_pi

    def solve(self, t_current: float, t_next: float, inputs: "CO2Forcing.Inputs"):
        """Compute CO2 forcing for a single step."""
        concentration = inputs.concentration.get()
        return self.Outputs(forcing=self.alpha * np.log(concentration / self.c_pi))


# %% [markdown]
# `CH4Forcing` has the same shape.
# It reads `Atmospheric Concentration|CH4` in ppb and writes `Effective Radiative Forcing|CH4`
# using a square-root relationship, `ERF = alpha * (sqrt(C) - sqrt(C_pi))`.


# %% tags=["remove_input"]
class CH4Forcing(Component):
    """CH4 forcing from concentration: ERF = alpha * (sqrt(C) - sqrt(C_pi))."""

    concentration = Input("Atmospheric Concentration|CH4", unit="ppb")
    forcing = Output("Effective Radiative Forcing|CH4", unit="W/m^2")

    def __init__(self, alpha: float = 0.036, c_pi: float = 700.0):
        self.alpha = alpha
        self.c_pi = c_pi

    def solve(self, t_current: float, t_next: float, inputs: "CH4Forcing.Inputs"):
        """Compute CH4 forcing for a single step."""
        concentration = inputs.concentration.get()
        forcing = self.alpha * (np.sqrt(concentration) - np.sqrt(self.c_pi))
        return self.Outputs(forcing=forcing)


# %% [markdown]
# ## Building a Model Without a Schema
#
# First, see what happens without a schema.
# The model runs from 1750 to 2100 in 10-year steps.
# `co2_conc` and `ch4_conc` are prescribed concentrations,
# rising from preindustrial levels (280 ppm and 700 ppb) to 500 ppm and 2200 ppb in 2050.

# %% tags=["remove_input"]
time_axis = TimeAxis.from_values(np.arange(1750.0, 2101.0, 10.0))
scenario_bounds = np.array([1750.0, 1850.0, 1960.0, 2000.0, 2020.0, 2050.0, 2100.0])

co2_conc = Timeseries(
    np.array([280.0, 280.0, 315.0, 370.0, 420.0, 500.0]),
    TimeAxis.from_bounds(scenario_bounds),
    "ppm",
    InterpolationStrategy.Linear,
)
ch4_conc = Timeseries(
    np.array([700.0, 700.0, 1200.0, 1800.0, 1900.0, 2200.0]),
    TimeAxis.from_bounds(scenario_bounds),
    "ppb",
    InterpolationStrategy.Linear,
)

# %%
model_no_schema = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_component(CO2Forcing())
    .with_component(CH4Forcing())
    .with_prescribed_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_prescribed_variable("Atmospheric Concentration|CH4", ch4_conc)
).build()
model_no_schema.run()

print("\n".join(sorted(model_no_schema.timeseries().names())))

# %% [markdown]
# Each component produces its own forcing variable, but there is no total.
# To get one, you would have to sum the outputs after the run.

# %% [markdown]
# ## Introducing VariableSchema
#
# A `VariableSchema` declares:
#
# - **Variables**: named quantities with units.
# - **Aggregates**: derived values computed from other variables.
#
# The function below declares the concentrations and forcings of the two gases.
# Later sections reuse it to try other aggregates.


# %%
def ghg_variables() -> VariableSchema:
    """Declare the CO2 and CH4 concentrations and forcings."""
    schema = VariableSchema()
    schema.add_variable("Atmospheric Concentration|CO2", "ppm")
    schema.add_variable("Atmospheric Concentration|CH4", "ppb")
    schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
    schema.add_variable("Effective Radiative Forcing|CH4", "W/m^2")
    return schema


schema = ghg_variables()
schema.add_aggregate(
    name="Effective Radiative Forcing",
    unit="W/m^2",
    operation="sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
)

# %% [markdown]
# `validate()` checks the schema and raises a `ValueError` if it is inconsistent.

# %%
schema.validate()
print(f"{len(schema.variables)} variables, {len(schema.aggregates)} aggregate")

# %% [markdown]
# ## Building a Model With a Schema
#
# Attach the schema with `with_schema()`.
# The rest of the build is unchanged.

# %%
model_with_schema = (
    ModelBuilder()
    .with_schema(schema)
    .with_time_axis(time_axis)
    .with_component(CO2Forcing())
    .with_component(CH4Forcing())
    .with_prescribed_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_prescribed_variable("Atmospheric Concentration|CH4", ch4_conc)
).build()
model_with_schema.run()

results = model_with_schema.timeseries()
print("\n".join(sorted(results.names())))

# %% [markdown]
# The `Effective Radiative Forcing` aggregate now appears in the results.
# The model sums the CO2 and CH4 contributions at each step.

# %% tags=["remove_input"]
import matplotlib.pyplot as plt

times = time_axis.values()
co2_erf = results.get_timeseries_by_name("Effective Radiative Forcing|CO2").values()
ch4_erf = results.get_timeseries_by_name("Effective Radiative Forcing|CH4").values()
total_erf = results.get_timeseries_by_name("Effective Radiative Forcing").values()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, co2_erf, "--", label="CO2")
ax.plot(times, ch4_erf, "--", label="CH4")
ax.plot(times, total_erf, linewidth=2, label="Total (aggregate)")
ax.set_xlabel("Year")
ax.set_ylabel("Effective Radiative Forcing (W/m²)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Inspecting the Aggregator
#
# The schema adds a virtual `AggregatorComponent` to the model.
# `coupling()` lists it after the two forcing components, with both forcings as inputs.

# %%
print(model_with_schema.coupling(format="plain"))

# %% [markdown]
# ## Aggregate Operations
#
# RSCM supports three aggregation operations:
#
# | Operation      | Description | Use Case |
# |----------------|-------------|----------|
# | `sum`          | Add all contributor values | Total forcing, total emissions |
# | `mean`         | Arithmetic average | Average temperature across regions |
# | `weighted_sum` | Weighted sum with specified weights | Area-weighted global average |
#
# The rest of this page builds the same kind of model with different schemas,
# so the build goes in a helper.


# %%
def forcing_model(schema, components, prescribed):
    """Build and run a model of Python components with the given schema."""
    builder = ModelBuilder().with_schema(schema).with_time_axis(time_axis)
    for component in components:
        builder.with_component(component)
    for name, timeseries in prescribed.items():
        builder.with_prescribed_variable(name, timeseries)
    model = builder.build()
    model.run()
    return model.timeseries()


ghg_components = [CO2Forcing(), CH4Forcing()]
ghg_concentrations = {
    "Atmospheric Concentration|CO2": co2_conc,
    "Atmospheric Concentration|CH4": ch4_conc,
}

# %% [markdown]
# ### Mean Aggregation
#
# A `Mean` aggregate averages its contributors.

# %%
mean_schema = ghg_variables()
mean_schema.add_aggregate(
    name="Average Forcing",
    unit="W/m^2",
    operation="mean",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
)

results_mean = forcing_model(mean_schema, ghg_components, ghg_concentrations)
avg_forcing = results_mean.get_timeseries_by_name("Average Forcing").values()

print(f"2100: CO2 {co2_erf[-1]:.2f}, CH4 {ch4_erf[-1]:.2f} W/m²")
print(f"Sum {total_erf[-1]:.2f}, Mean {avg_forcing[-1]:.2f} W/m²")

# %% [markdown]
# ### Weighted Aggregation
#
# A `weighted_sum` aggregate gives each contributor its own weight,
# for example the area fraction of a region.
# The weights follow the order of the contributors.

# %%
weighted_schema = ghg_variables()
weighted_schema.add_aggregate(
    name="Weighted Forcing",
    unit="W/m^2",
    operation="weighted_sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
    weights=[0.7, 0.3],
)

results_weighted = forcing_model(weighted_schema, ghg_components, ghg_concentrations)
weighted_forcing = results_weighted.get_timeseries_by_name("Weighted Forcing").values()

print(f"2100: 0.7 * CO2 + 0.3 * CH4 = {weighted_forcing[-1]:.2f} W/m²")
np.testing.assert_allclose(weighted_forcing, 0.7 * co2_erf + 0.3 * ch4_erf)

# %% [markdown]
# ## Hierarchical Aggregation
#
# Aggregates can reference other aggregates, which allows nested forcing categories:
#
# ```
# Effective Radiative Forcing
# ├── Effective Radiative Forcing|GHG (Sum)
# │   ├── Effective Radiative Forcing|CO2
# │   └── Effective Radiative Forcing|CH4
# └── Effective Radiative Forcing|Aerosol
# ```
#
# An aerosol component adds a negative forcing, proportional to emissions.


# %%
class AerosolForcing(Component):
    """Aerosol forcing proportional to emissions (negative forcing)."""

    emissions = Input("Emissions|Aerosol", unit="Tg/yr")
    forcing = Output("Effective Radiative Forcing|Aerosol", unit="W/m^2")

    def __init__(self, efficiency: float = -0.01):
        self.efficiency = efficiency

    def solve(self, t_current: float, t_next: float, inputs: "AerosolForcing.Inputs"):
        """Compute aerosol forcing for a single step."""
        return self.Outputs(forcing=inputs.emissions.get() * self.efficiency)


# %% [markdown]
# `aerosol_emissions` rises from 10 Tg/yr before 1850 to a peak of 80 Tg/yr in 2000, then falls.

# %% tags=["remove_input"]
aerosol_emissions = Timeseries(
    np.array([10.0, 10.0, 50.0, 80.0, 70.0, 40.0]),
    TimeAxis.from_bounds(scenario_bounds),
    "Tg/yr",
    InterpolationStrategy.Linear,
)

# %% [markdown]
# The schema declares the aerosol variables,
# then a GHG aggregate and a total that includes it.

# %%
hierarchical_schema = ghg_variables()
hierarchical_schema.add_variable("Emissions|Aerosol", "Tg/yr")
hierarchical_schema.add_variable("Effective Radiative Forcing|Aerosol", "W/m^2")

hierarchical_schema.add_aggregate(
    name="Effective Radiative Forcing|GHG",
    unit="W/m^2",
    operation="sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
)
hierarchical_schema.add_aggregate(
    name="Effective Radiative Forcing",
    unit="W/m^2",
    operation="sum",
    contributors=[
        "Effective Radiative Forcing|GHG",
        "Effective Radiative Forcing|Aerosol",
    ],
)
hierarchical_schema.validate()

results_hier = forcing_model(
    hierarchical_schema,
    [*ghg_components, AerosolForcing()],
    {**ghg_concentrations, "Emissions|Aerosol": aerosol_emissions},
)

# %% [markdown]
# The total sits below the GHG aggregate because the aerosol forcing is negative.

# %% tags=["remove_input"]
fig, ax = plt.subplots(figsize=(10, 5))
for name, label, style in [
    ("Effective Radiative Forcing|CO2", "CO2", ":"),
    ("Effective Radiative Forcing|CH4", "CH4", ":"),
    ("Effective Radiative Forcing|Aerosol", "Aerosol", ":"),
    ("Effective Radiative Forcing|GHG", "GHG (CO2 + CH4)", "--"),
    ("Effective Radiative Forcing", "Total (GHG + Aerosol)", "-"),
]:
    ax.plot(
        times, results_hier.get_timeseries_by_name(name).values(), style, label=label
    )
ax.axhline(0, color="grey", alpha=0.3)
ax.set_xlabel("Year")
ax.set_ylabel("Effective Radiative Forcing (W/m²)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Handling Missing Data (NaN)
#
# If a contributor has no value (NaN) at a time point,
# the aggregate excludes it and combines the remaining contributors.
# The component below returns NaN before a given year.


# %%
class SometimesNaNForcing(Component):
    """Return NaN before `start_year` and 1 W/m² after."""

    concentration = Input("Atmospheric Concentration|Test", unit="ppm")
    forcing = Output("Effective Radiative Forcing|Test", unit="W/m^2")

    def __init__(self, start_year: float = 2000.0):
        self.start_year = start_year

    def solve(
        self, t_current: float, t_next: float, inputs: "SometimesNaNForcing.Inputs"
    ):
        """Return NaN before start_year, 1.0 after."""
        if t_current < self.start_year:
            return self.Outputs(forcing=float("nan"))
        return self.Outputs(forcing=1.0)


# %%
nan_schema = VariableSchema()
nan_schema.add_variable("Atmospheric Concentration|CO2", "ppm")
nan_schema.add_variable("Atmospheric Concentration|Test", "ppm")
nan_schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
nan_schema.add_variable("Effective Radiative Forcing|Test", "W/m^2")
nan_schema.add_aggregate(
    name="Effective Radiative Forcing",
    unit="W/m^2",
    operation="sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|Test",
    ],
)

# %% [markdown]
# The test component reads a constant dummy concentration.
# The table shows each variable around the year the test forcing starts.

# %%
test_conc = Timeseries(
    np.array([100.0]),
    TimeAxis.from_bounds(np.array([1750.0, 2100.0])),
    "ppm",
    InterpolationStrategy.Previous,
)
results_nan = forcing_model(
    nan_schema,
    [CO2Forcing(), SometimesNaNForcing(start_year=2000.0)],
    {
        "Atmospheric Concentration|CO2": co2_conc,
        "Atmospheric Concentration|Test": test_conc,
    },
)

columns = {
    "CO2 ERF": "Effective Radiative Forcing|CO2",
    "Test ERF": "Effective Radiative Forcing|Test",
    "Total (Sum)": "Effective Radiative Forcing",
}
table = pd.DataFrame(
    {
        label: results_nan.get_timeseries_by_name(name).values()
        for label, name in columns.items()
    },
    index=pd.Index(times.astype(int), name="Year"),
)
table.loc[1980:2030].round(3)

# %% [markdown]
# The Test ERF is NaN up to 2000 and is excluded from the sum.
# The step starting in 2000 writes 1.0 at 2010, from which point it is included.
# A value at a time point is written by the step that ends there.

# %% [markdown]
# ## Schema Validation Errors
#
# `validate()` checks several constraints.
# Each example below breaks one of them.
#
# An aggregate's contributors must all be declared in the schema:

# %%
try:
    bad_schema = VariableSchema()
    bad_schema.add_variable("A", "units")
    bad_schema.add_aggregate(
        name="Total", unit="units", operation="sum", contributors=["A", "B"]
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# Contributors must share the aggregate's unit:

# %%
try:
    bad_schema = VariableSchema()
    bad_schema.add_variable("Temp", "K")
    bad_schema.add_variable("Forcing", "W/m^2")
    bad_schema.add_aggregate(
        name="Total", unit="K", operation="sum", contributors=["Temp", "Forcing"]
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# A `weighted_sum` aggregate needs one weight per contributor:

# %%
try:
    bad_schema = VariableSchema()
    for name in ["A", "B", "C"]:
        bad_schema.add_variable(name, "units")
    bad_schema.add_aggregate(
        name="Total",
        unit="units",
        operation="weighted_sum",
        contributors=["A", "B", "C"],
        weights=[0.5, 0.5],
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# Aggregates must not depend on each other in a cycle:

# %%
try:
    bad_schema = VariableSchema()
    bad_schema.add_aggregate(
        name="A", unit="units", operation="sum", contributors=["B"]
    )
    bad_schema.add_aggregate(
        name="B", unit="units", operation="sum", contributors=["A"]
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# ### Build-Time Validation
#
# With a schema attached, the model builder also checks every component against it.
# A component that reads or writes a variable missing from the schema fails to build,
# as does one that declares a unit with a different dimension.
# Here the aerosol component meets the GHG-only schema from earlier.
# The error suggests `schema.variable(...)`, the Rust builder method.
# In Python, add the variable with `add_variable()`.

# %%
try:
    (
        ModelBuilder()
        .with_schema(schema)
        .with_time_axis(time_axis)
        .with_component(AerosolForcing())
        .with_prescribed_variable("Emissions|Aerosol", aerosol_emissions)
    ).build()
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# Units only need the same dimension and species family, not the same spelling.
# When a Rust component declares a compatible unit that differs from the schema,
# the model converts values as the component reads them.
# See the [Units tutorial](../units_tutorial/) for details.

# %% [markdown]
# ## Inspecting a Schema
#
# The `variables` and `aggregates` attributes expose the schema's definitions.
# `VariableSchema` has no serialisation method yet,
# but you can read these definitions to export them yourself.

# %%
for name, aggregate in hierarchical_schema.aggregates.items():
    print(f"{name}: {aggregate.operation_type} of {aggregate.contributors}")

# %% [markdown]
# ## Key Points
#
# - Use a `VariableSchema` to combine several component outputs into one variable.
# - Contributors must share the aggregate's unit and grid type.
# - Aggregates appear as regular variables in the model output,
#   and their virtual aggregator components appear in `coupling()`.
# - Aggregates exclude NaN contributors.
# - Errors surface early, when you call `validate()` or build the model.
#
# ## Next Steps
#
# - [Spatial Grids](../grid_variables/) covers variables on regional grids and aggregating them.
# - [State Serialisation](../state_serialisation/) shows how to save and restore a model.
