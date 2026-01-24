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
# This tutorial teaches you how to aggregate multiple component outputs into
# derived values like "Total Radiative Forcing". You'll learn to use
# `VariableSchema` to declare variables and define how they combine.
#
# ## Why Aggregation?
#
# Climate models often compute forcing from multiple sources:
#
# - CO2 forcing
# - CH4 forcing
# - N2O forcing
# - Aerosol forcing
# - etc.
#
# To calculate climate response, you need the **total forcing** - the sum of
# all individual contributions. Without aggregation, you'd need to:
#
# 1. Create a separate component that reads all forcing variables
# 2. Manually sum them in that component
# 3. Update the component whenever you add/remove forcing sources
#
# `VariableSchema` solves this by letting you declare aggregation rules
# at the model level. The framework automatically creates virtual components
# that compute the aggregates.
#
# ## Prerequisites
#
# This tutorial assumes you understand:
#
# - Building models with `ModelBuilder` (see [Coupled Models](coupled_model.py))
# - Creating Python components (see [Components in Python](component_python.py))
#
# ## Related Resources
#
# - [Key Concepts](../key_concepts.md): Core RSCM architecture
# - [Coupled Models](coupled_model.py): Building models from components

# %%
import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import Image, display

from rscm.component import Component, Input, Output
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    Timeseries,
    VariableSchema,
)

# %% [markdown]
# ## Creating Simple Forcing Components
#
# Let's create two simple components that calculate radiative forcing from
# different greenhouse gases. In a real model, these would use proper
# radiative transfer equations - here we use simplified linear relationships.


# %%
class CO2Forcing(Component):
    """
    Calculate CO2 radiative forcing from concentration.

    Uses a simplified logarithmic relationship:
    ERF = alpha * ln(C / C_pi)
    """

    concentration = Input("Atmospheric Concentration|CO2", unit="ppm")
    forcing = Output("Effective Radiative Forcing|CO2", unit="W/m^2")

    def __init__(self, alpha: float = 5.35, c_pi: float = 280.0):
        """
        Initialise CO2 forcing component.

        Parameters
        ----------
        alpha
            Radiative efficiency (W/m^2 per doubling)
        c_pi
            Pre-industrial CO2 concentration (ppm)
        """
        self.alpha = alpha
        self.c_pi = c_pi

    def solve(self, t_current: float, t_next: float, inputs: "CO2Forcing.Inputs"):
        """Compute CO2 forcing for a single timestep."""
        concentration = inputs.concentration.at_start()
        # Simplified logarithmic forcing
        forcing = self.alpha * np.log(concentration / self.c_pi)
        return self.Outputs(forcing=forcing)


class CH4Forcing(Component):
    """
    Calculate CH4 radiative forcing from concentration.

    Uses a simplified square-root relationship:
    ERF = alpha * (sqrt(C) - sqrt(C_pi))
    """

    concentration = Input("Atmospheric Concentration|CH4", unit="ppb")
    forcing = Output("Effective Radiative Forcing|CH4", unit="W/m^2")

    def __init__(self, alpha: float = 0.036, c_pi: float = 700.0):
        """
        Initialise CH4 forcing component.

        Parameters
        ----------
        alpha
            Radiative efficiency parameter
        c_pi
            Pre-industrial CH4 concentration (ppb)
        """
        self.alpha = alpha
        self.c_pi = c_pi

    def solve(self, t_current: float, t_next: float, inputs: "CH4Forcing.Inputs"):
        """Compute CH4 forcing for a single timestep."""
        concentration = inputs.concentration.at_start()
        # Simplified square-root forcing
        forcing = self.alpha * (np.sqrt(concentration) - np.sqrt(self.c_pi))
        return self.Outputs(forcing=forcing)


# %% [markdown]
# ## Building a Model WITHOUT Schema
#
# First, let's see what happens without a schema. Each component produces
# its own forcing variable, but there's no total.

# %%
# Create time axis
time_axis = TimeAxis.from_values(np.arange(1750.0, 2101.0, 10.0))

# Create exogenous concentration data
co2_conc = Timeseries(
    np.array([280.0, 280.0, 315.0, 370.0, 420.0, 500.0]),
    TimeAxis.from_bounds(
        np.array([1750.0, 1850.0, 1960.0, 2000.0, 2020.0, 2050.0, 2100.0])
    ),
    "ppm",
    InterpolationStrategy.Linear,
)

ch4_conc = Timeseries(
    np.array([700.0, 700.0, 1200.0, 1800.0, 1900.0, 2200.0]),
    TimeAxis.from_bounds(
        np.array([1750.0, 1850.0, 1960.0, 2000.0, 2020.0, 2050.0, 2100.0])
    ),
    "ppb",
    InterpolationStrategy.Linear,
)

# %%
# Build model without schema
model_no_schema = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(CO2Forcing()))
    .with_py_component(PythonComponent.build(CH4Forcing()))
    .with_exogenous_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_exogenous_variable("Atmospheric Concentration|CH4", ch4_conc)
).build()

# Run the model
model_no_schema.run()

# %%
# Get results
results = model_no_schema.timeseries()
print("Available variables:")
for name in sorted(results.names()):
    print(f"  - {name}")

# %% [markdown]
# Notice there's no "Total Forcing" variable - just the individual components.
# To get a total, you'd have to calculate it manually after extraction.

# %% [markdown]
# ## Introducing VariableSchema
#
# A `VariableSchema` declares:
#
# 1. **Variables**: Named quantities with units (e.g., "ERF|CO2")
# 2. **Aggregates**: Derived values computed from other variables
#
# Let's create a schema that defines a total forcing aggregate.

# %%
# Create a schema with aggregation
schema = VariableSchema()

# Declare the input variables (concentrations)
schema.add_variable("Atmospheric Concentration|CO2", "ppm")
schema.add_variable("Atmospheric Concentration|CH4", "ppb")

# Declare the individual forcing outputs
schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
schema.add_variable("Effective Radiative Forcing|CH4", "W/m^2")

# Declare an aggregate that sums the individual forcings
schema.add_aggregate(
    name="Effective Radiative Forcing",
    unit="W/m^2",
    operation="Sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
)

# %%
# Verify the schema is valid
schema.validate()  # Raises ValueError if invalid
print("Schema is valid!")
print(f"Variables: {len(schema.variables)}")
print(f"Aggregates: {len(schema.aggregates)}")

# %% [markdown]
# ## Building a Model WITH Schema
#
# Now let's build the same model but with our schema attached.

# %%
# Build model with schema
model_with_schema = (
    ModelBuilder()
    .with_schema(schema)  # Attach the schema
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(CO2Forcing()))
    .with_py_component(PythonComponent.build(CH4Forcing()))
    .with_exogenous_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_exogenous_variable("Atmospheric Concentration|CH4", ch4_conc)
).build()

# Run the model
model_with_schema.run()

# %%
# Get results - now includes the aggregate!
results = model_with_schema.timeseries()
print("Available variables:")
for name in sorted(results.names()):
    print(f"  - {name}")

# %% [markdown]
# The "Effective Radiative Forcing" aggregate now appears automatically!
# It's computed by summing the CO2 and CH4 contributions at each timestep.

# %%
# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))

times = time_axis.values()

# Get timeseries data
co2_erf = results.get_timeseries_by_name("Effective Radiative Forcing|CO2")
ch4_erf = results.get_timeseries_by_name("Effective Radiative Forcing|CH4")
total_erf = results.get_timeseries_by_name("Effective Radiative Forcing")

ax.plot(times, co2_erf.values(), label="CO2 Forcing", linestyle="--")
ax.plot(times, ch4_erf.values(), label="CH4 Forcing", linestyle="--")
ax.plot(times, total_erf.values(), label="Total Forcing", linewidth=2)

ax.set_xlabel("Year")
ax.set_ylabel("Effective Radiative Forcing (W/m²)")
ax.set_title("Radiative Forcing with Automatic Aggregation")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualising the Component Graph
#
# When you add a schema with aggregates, RSCM inserts virtual "aggregator"
# components into the dependency graph. Let's visualise this.


# %%
def view_pydot(pdot):
    """Display a pydot graph in the notebook."""
    plt = Image(pdot.create_png())
    display(plt)


graph = pydot.graph_from_dot_data(model_with_schema.as_dot())[0]
view_pydot(graph)

# %% [markdown]
# Notice the aggregator node that takes inputs from both forcing components
# and produces the total forcing output.

# %% [markdown]
# ## Aggregate Operations
#
# RSCM supports three aggregation operations:
#
# | Operation  | Description | Use Case |
# |------------|-------------|----------|
# | `Sum`      | Add all contributor values | Total forcing, total emissions |
# | `Mean`     | Arithmetic average | Average temperature across regions |
# | `Weighted` | Weighted sum with specified weights | Area-weighted global average |
#
# ### Mean Aggregation
#
# Let's demonstrate mean aggregation:

# %%
# Create a schema with mean aggregation
mean_schema = VariableSchema()
mean_schema.add_variable("Atmospheric Concentration|CO2", "ppm")
mean_schema.add_variable("Atmospheric Concentration|CH4", "ppb")
mean_schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
mean_schema.add_variable("Effective Radiative Forcing|CH4", "W/m^2")

# Use Mean instead of Sum
mean_schema.add_aggregate(
    name="Average Forcing",
    unit="W/m^2",
    operation="Mean",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
)

# Build and run
model_mean = (
    ModelBuilder()
    .with_schema(mean_schema)
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(CO2Forcing()))
    .with_py_component(PythonComponent.build(CH4Forcing()))
    .with_exogenous_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_exogenous_variable("Atmospheric Concentration|CH4", ch4_conc)
).build()

model_mean.run()

# Compare Sum vs Mean
results_mean = model_mean.timeseries()
avg_forcing = results_mean.get_timeseries_by_name("Average Forcing")

print("At year 2100:")
print(f"  CO2 Forcing: {co2_erf.values()[-1]:.2f} W/m²")
print(f"  CH4 Forcing: {ch4_erf.values()[-1]:.2f} W/m²")
print(f"  Sum (Total): {total_erf.values()[-1]:.2f} W/m²")
print(f"  Mean (Avg):  {avg_forcing.values()[-1]:.2f} W/m²")

# %% [markdown]
# ### Weighted Aggregation
#
# Weighted aggregation is useful when contributors have different importance.
# For example, weighting regional values by area fraction:

# %%
# Create a schema with weighted aggregation
weighted_schema = VariableSchema()
weighted_schema.add_variable("Atmospheric Concentration|CO2", "ppm")
weighted_schema.add_variable("Atmospheric Concentration|CH4", "ppb")
weighted_schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
weighted_schema.add_variable("Effective Radiative Forcing|CH4", "W/m^2")

# Weighted sum: CO2 counts 70%, CH4 counts 30%
weighted_schema.add_aggregate(
    name="Weighted Forcing",
    unit="W/m^2",
    operation="Weighted",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
    weights=[0.7, 0.3],  # Weights must match contributor order
)

# Build and run
model_weighted = (
    ModelBuilder()
    .with_schema(weighted_schema)
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(CO2Forcing()))
    .with_py_component(PythonComponent.build(CH4Forcing()))
    .with_exogenous_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_exogenous_variable("Atmospheric Concentration|CH4", ch4_conc)
).build()

model_weighted.run()

results_weighted = model_weighted.timeseries()
weighted_forcing = results_weighted.get_timeseries_by_name("Weighted Forcing")

print("At year 2100:")
print(f"  CO2 Forcing: {co2_erf.values()[-1]:.2f} W/m²")
print(f"  CH4 Forcing: {ch4_erf.values()[-1]:.2f} W/m²")
print(f"  Weighted (0.7*CO2 + 0.3*CH4): {weighted_forcing.values()[-1]:.2f} W/m²")
print(
    f"  Manual calc: {0.7 * co2_erf.values()[-1] + 0.3 * ch4_erf.values()[-1]:.2f} W/m²"
)

# %% [markdown]
# ## Hierarchical Aggregation
#
# Aggregates can reference other aggregates, enabling hierarchical structures.
# This is useful for climate models with nested forcing categories:
#
# ```
# Total ERF
# ├── GHG Forcing (Sum)
# │   ├── CO2 Forcing
# │   └── CH4 Forcing
# └── Aerosol Forcing
# ```


# %%
class AerosolForcing(Component):
    """Simple aerosol forcing component (negative forcing)."""

    emissions = Input("Emissions|Aerosol", unit="Tg/yr")
    forcing = Output("Effective Radiative Forcing|Aerosol", unit="W/m^2")

    def __init__(self, efficiency: float = -0.01):
        """Initialise aerosol forcing component."""
        self.efficiency = efficiency

    def solve(self, t_current: float, t_next: float, inputs: "AerosolForcing.Inputs"):
        """Compute aerosol forcing for a single timestep."""
        emissions = inputs.emissions.at_start()
        forcing = emissions * self.efficiency
        return self.Outputs(forcing=forcing)


# %%
# Aerosol emissions data
aerosol_emissions = Timeseries(
    np.array([10.0, 10.0, 50.0, 80.0, 70.0, 40.0]),
    TimeAxis.from_bounds(
        np.array([1750.0, 1850.0, 1960.0, 2000.0, 2020.0, 2050.0, 2100.0])
    ),
    "Tg/yr",
    InterpolationStrategy.Linear,
)

# %%
# Create hierarchical schema
hierarchical_schema = VariableSchema()

# Input variables
hierarchical_schema.add_variable("Atmospheric Concentration|CO2", "ppm")
hierarchical_schema.add_variable("Atmospheric Concentration|CH4", "ppb")
hierarchical_schema.add_variable("Emissions|Aerosol", "Tg/yr")

# Individual forcing outputs
hierarchical_schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
hierarchical_schema.add_variable("Effective Radiative Forcing|CH4", "W/m^2")
hierarchical_schema.add_variable("Effective Radiative Forcing|Aerosol", "W/m^2")

# First level: GHG forcing aggregate
hierarchical_schema.add_aggregate(
    name="Effective Radiative Forcing|GHG",
    unit="W/m^2",
    operation="Sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|CH4",
    ],
)

# Second level: Total forcing (includes the GHG aggregate!)
hierarchical_schema.add_aggregate(
    name="Effective Radiative Forcing",
    unit="W/m^2",
    operation="Sum",
    contributors=[
        "Effective Radiative Forcing|GHG",  # Reference to another aggregate
        "Effective Radiative Forcing|Aerosol",
    ],
)

hierarchical_schema.validate()
print("Hierarchical schema is valid!")

# %%
# Build and run hierarchical model
model_hierarchical = (
    ModelBuilder()
    .with_schema(hierarchical_schema)
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(CO2Forcing()))
    .with_py_component(PythonComponent.build(CH4Forcing()))
    .with_py_component(PythonComponent.build(AerosolForcing()))
    .with_exogenous_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_exogenous_variable("Atmospheric Concentration|CH4", ch4_conc)
    .with_exogenous_variable("Emissions|Aerosol", aerosol_emissions)
).build()

model_hierarchical.run()

# %%
# Plot hierarchical results
results_hier = model_hierarchical.timeseries()

fig, ax = plt.subplots(figsize=(10, 6))

times = time_axis.values()

co2_f = results_hier.get_timeseries_by_name("Effective Radiative Forcing|CO2")
ch4_f = results_hier.get_timeseries_by_name("Effective Radiative Forcing|CH4")
aer_f = results_hier.get_timeseries_by_name("Effective Radiative Forcing|Aerosol")
ghg_f = results_hier.get_timeseries_by_name("Effective Radiative Forcing|GHG")
total_f = results_hier.get_timeseries_by_name("Effective Radiative Forcing")

ax.plot(times, co2_f.values(), label="CO2", linestyle=":", alpha=0.7)
ax.plot(times, ch4_f.values(), label="CH4", linestyle=":", alpha=0.7)
ax.plot(times, aer_f.values(), label="Aerosol", linestyle=":", alpha=0.7)
ax.plot(times, ghg_f.values(), label="GHG (CO2+CH4)", linestyle="--", linewidth=1.5)
ax.plot(times, total_f.values(), label="Total (GHG+Aerosol)", linewidth=2.5)

ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
ax.set_xlabel("Year")
ax.set_ylabel("Effective Radiative Forcing (W/m²)")
ax.set_title("Hierarchical Forcing Aggregation")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Handling Missing Data (NaN)
#
# What happens if a contributor doesn't produce a value? RSCM handles this
# gracefully by treating missing values as NaN and excluding them from
# aggregation.


# %%
class SometimesNaNForcing(Component):
    """A component that returns NaN before a certain year."""

    concentration = Input("Atmospheric Concentration|Test", unit="ppm")
    forcing = Output("Effective Radiative Forcing|Test", unit="W/m^2")

    def __init__(self, start_year: float = 2000.0):
        """Initialise component with start year for non-NaN output."""
        self.start_year = start_year

    def solve(
        self, t_current: float, t_next: float, inputs: "SometimesNaNForcing.Inputs"
    ):
        """Return NaN before start_year, 1.0 after."""
        if t_current < self.start_year:
            return self.Outputs(forcing=float("nan"))
        return self.Outputs(forcing=1.0)  # Constant 1 W/m² after start_year


# %%
# Create schema with NaN-producing component
nan_schema = VariableSchema()
nan_schema.add_variable("Atmospheric Concentration|CO2", "ppm")
nan_schema.add_variable("Atmospheric Concentration|Test", "ppm")
nan_schema.add_variable("Effective Radiative Forcing|CO2", "W/m^2")
nan_schema.add_variable("Effective Radiative Forcing|Test", "W/m^2")

nan_schema.add_aggregate(
    name="Total Forcing",
    unit="W/m^2",
    operation="Sum",
    contributors=[
        "Effective Radiative Forcing|CO2",
        "Effective Radiative Forcing|Test",
    ],
)

# Dummy concentration data for the test component
test_conc = Timeseries(
    np.array([100.0]),
    TimeAxis.from_bounds(np.array([1750.0, 2100.0])),
    "ppm",
    InterpolationStrategy.Previous,
)

# Build and run
model_nan = (
    ModelBuilder()
    .with_schema(nan_schema)
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(CO2Forcing()))
    .with_py_component(PythonComponent.build(SometimesNaNForcing(start_year=2000.0)))
    .with_exogenous_variable("Atmospheric Concentration|CO2", co2_conc)
    .with_exogenous_variable("Atmospheric Concentration|Test", test_conc)
).build()

model_nan.run()

# %%
results_nan = model_nan.timeseries()

co2_vals = results_nan.get_timeseries_by_name(
    "Effective Radiative Forcing|CO2"
).values()
test_vals = results_nan.get_timeseries_by_name(
    "Effective Radiative Forcing|Test"
).values()
total_vals = results_nan.get_timeseries_by_name("Total Forcing").values()

print("Year  | CO2 ERF | Test ERF | Total (Sum)")
print("-" * 45)
for i, year in enumerate(time_axis.values()[:8]):
    print(
        f"{year:.0f}  | {co2_vals[i]:7.3f} | {test_vals[i]:8.3f} | {total_vals[i]:7.3f}"
    )

print("\nNote: Before 2000, Test ERF is NaN and excluded from the sum.")
print("After 2000, Test ERF = 1.0 and is included in the sum.")

# %% [markdown]
# ## Schema Validation Errors
#
# The schema validates several constraints. Let's see what errors look like:

# %%
# Example 1: Undefined contributor
try:
    bad_schema = VariableSchema()
    bad_schema.add_variable("A", "units")
    bad_schema.add_aggregate(
        name="Total",
        unit="units",
        operation="Sum",
        contributors=["A", "B"],  # "B" is not defined!
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %%
# Example 2: Unit mismatch
try:
    bad_schema = VariableSchema()
    bad_schema.add_variable("Temp", "K")  # Kelvin
    bad_schema.add_variable("Forcing", "W/m^2")  # Different unit!
    bad_schema.add_aggregate(
        name="Total",
        unit="K",
        operation="Sum",
        contributors=["Temp", "Forcing"],  # Can't sum K and W/m^2
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %%
# Example 3: Weight count mismatch
try:
    bad_schema = VariableSchema()
    bad_schema.add_variable("A", "units")
    bad_schema.add_variable("B", "units")
    bad_schema.add_variable("C", "units")
    bad_schema.add_aggregate(
        name="Total",
        unit="units",
        operation="Weighted",
        contributors=["A", "B", "C"],  # 3 contributors
        weights=[0.5, 0.5],  # Only 2 weights!
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %%
# Example 4: Circular dependency
try:
    bad_schema = VariableSchema()
    bad_schema.add_aggregate(
        name="A",
        unit="units",
        operation="Sum",
        contributors=["B"],
    )
    bad_schema.add_aggregate(
        name="B",
        unit="units",
        operation="Sum",
        contributors=["A"],  # A depends on B, B depends on A!
    )
    bad_schema.validate()
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# ## Loading Schemas from Configuration
#
# Schemas can be serialised to JSON or TOML for configuration files.
# This enables model configurations without code changes.

# %%
import json

# Serialise schema to JSON
schema_json = json.dumps(
    {
        "variables": {
            name: {"name": v.name, "unit": v.unit, "grid_type": str(v.grid_type)}
            for name, v in schema.variables.items()
        },
        "aggregates": {
            name: {
                "name": a.name,
                "unit": a.unit,
                "grid_type": str(a.grid_type),
                "operation": a.operation_type,
                "contributors": a.contributors,
            }
            for name, a in schema.aggregates.items()
        },
    },
    indent=2,
)

print("Schema as JSON:")
print(schema_json)

# %% [markdown]
# ## Summary
#
# In this tutorial, you learned:
#
# 1. **Why aggregation matters**: Combining multiple outputs into totals
# 2. **Creating a VariableSchema**: Declaring variables and aggregates
# 3. **Aggregate operations**: Sum, Mean, and Weighted
# 4. **Hierarchical aggregation**: Aggregates can reference other aggregates
# 5. **NaN handling**: Missing values are excluded from computations
# 6. **Validation errors**: Schema constraints and error messages
#
# ### Key Points
#
# - Use `VariableSchema` when you need to combine multiple component outputs
# - All contributors must have matching units and grid types
# - Aggregates appear as regular variables in the model output
# - Virtual aggregator components are visible in the dependency graph
# - Validation catches errors early (at schema creation or model build time)
#
# ### Next Steps
#
# - Try adding more forcing components to the hierarchical example
# - Experiment with different weight values for area-weighted averages
# - See [State Serialisation](state_serialisation.py) for saving model configurations
