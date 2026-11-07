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
# # A Coupled MAGICC Model
#
# This notebook assembles MAGICC7's full emissions-driven chain by hand.
# Thirteen RSCM components carry emissions through chemistry, the carbon cycle and permafrost
# to effective radiative forcing (ERF), the four-box climate model and sea level rise.
# Temperature then feeds back on the chemistry, the carbon cycle and permafrost.
#
# The result reproduces MAGICC7 v7.5.3 to floating-point rounding,
# as [the comparison](#agreement-with-magicc7) checks,
# and [the component graph](#the-component-graph) shows how the thirteen components connect.
#
# You will:
#
# - See how MAGICC modules map onto RSCM components.
# - Configure the components from MAGICC7's own parameter names.
# - Break the feedback loops with lags, as MAGICC7 does.
# - Feed prescribed forcing in on the four-box grid.
# - Render the component graph and run the model from 1750 to 2105.
# - Check the result against MAGICC7 v7.5.3 and against `run_magicc7`.
#
# Code that only reads files, tidies tables or draws figures is hidden on this page.
# The downloadable notebook has all of it.
#
# ## Related Resources
#
# - [Coupled Models](../coupled_model/): the `ModelBuilder` basics this notebook builds on
# - [Variable Schemas](../variable_schema/): declaring variables and grids
# - [MAGICC runs](../../magicc_runs/): the `run_magicc7`, `run_magicc` and `run_magicc_emissions` runners
# - [MAGICC modules](../../modules/): module specifications, status and parity evidence

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from IPython.display import SVG, display

import rscm
from rscm.core import (
    FourBoxTimeseries,
    GridType,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableRole,
    VariableSchema,
)
from rscm.magicc import (
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    ForcingAggregationBuilder,
    GhgForcingBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    PermafrostBuilder,
    SeaLevelRiseBuilder,
    TerrestrialCarbonBuilder,
    full,
    run_magicc,
    run_magicc7,
)
from rscm.magicc import prescribed as pr


# %% tags=["remove_input"]
def reference_run(name):
    """Read a MAGICC7 reference run indexed by variable and region, one column per year."""
    relative = Path("tests/regression/data/full_run") / f"{name}.csv.gz"
    candidates = [*Path.cwd().parents, Path.cwd(), *Path(rscm.__file__).parents]
    directory = next(d for d in candidates if (d / relative).exists())
    data = pd.read_csv(directory / relative).set_index(["variable", "region"])
    data = data[[c for c in data.columns if c[:4].isdigit()]]
    data.columns = [int(c[:4]) for c in data.columns]
    return data


# %% [markdown]
# ## MAGICC modules as RSCM components
#
# MAGICC7 is a chain of modules that step forward one year at a time.
# RSCM ports each module to a Rust component with declared inputs, outputs and states.
# The `ModelBuilder` then connects components by matching variable names.
#
# | Domain    | MAGICC module                                | RSCM                                         |
# |-----------|----------------------------------------------|----------------------------------------------|
# | Chemistry | CH4 chemistry                                | `CH4ChemistryBuilder`                        |
# | Chemistry | CH4 oxidation to CO2                         | `CH4OxidationBuilder`                        |
# | Chemistry | N2O chemistry                                | `N2OChemistryBuilder`                        |
# | Chemistry | Halocarbon chemistry                         | `HalocarbonChemistryBuilder`                 |
# | Carbon    | Terrestrial carbon                           | `TerrestrialCarbonBuilder`                   |
# | Carbon    | Ocean carbon                                 | `OceanCarbonBuilder`                         |
# | Carbon    | CO2 budget                                   | `CO2BudgetBuilder`                           |
# | Carbon    | Permafrost                                   | `PermafrostBuilder`                          |
# | Carbon    | Nitrogen limitation                          | `nitrogen` parameters of `TerrestrialCarbon` |
# | Forcing   | Ozone                                        | `OzoneForcingBuilder`                        |
# | Forcing   | Aerosols and other prescribed forcing        | `rscm.magicc.prescribed`                     |
# | Forcing   | GHG forcing                                  | `GhgForcingBuilder`                          |
# | Forcing   | Forcing aggregation                          | `ForcingAggregationBuilder`                  |
# | Climate   | Climate (UDEB)                               | `ClimateUDEBBuilder`                         |
# | Climate   | Sea level rise                               | `SeaLevelRiseBuilder`                        |
#
# UDEB is MAGICC7's upwelling-diffusion ocean coupled to an energy balance.
# It resolves four boxes: northern ocean, northern land, southern ocean and southern land.
#
# MAGICC7 calculates aerosol, land-use, solar and volcanic forcing from input files and emissions
# before the run starts.
# `rscm.magicc.prescribed` does the same, and the results enter the model as prescribed inputs.
# `AerosolDirectBuilder` and `AerosolIndirectBuilder` are simpler stand-ins and are not part of this chain.
#
# The [module status table](../../modules/#rscm-implementation-status)
# records the parity evidence for each module.

# %% [markdown]
# ## Emissions
#
# The model needs emissions for every input that no component produces.
# This notebook takes them from the MAGICC7 v7.5.3 SSP2-4.5 run that RSCM's regression tests use,
# so it needs a clone of the RSCM repository with RSCM installed from it.
# The same file holds MAGICC7's results, which the run is checked against later.
# A hidden helper, `reference_run`, reads such a file as a table indexed by variable and region,
# with one column per year.
#
# MAGICC7 splits most species into a fossil and industrial sector and an AFOLU
# (agriculture, forestry and other land use) sector.
# `full.SECTORAL_EMISSIONS` lists them with their units.
# Halocarbons are a single series per species, in kt/yr.

# %%
reference = reference_run("01_ssp245_emissions")
years = reference.columns.to_numpy(dtype=float)
sectors = {"Fossil": "MAGICC Fossil and Industrial", "AFOLU": "MAGICC AFOLU"}
emissions = {
    f"{species}|{sector}": reference.loc[
        (f"Emissions|{species}|{theirs}", "World")
    ].to_numpy()
    for species in full.SECTORAL_EMISSIONS
    for sector, theirs in sectors.items()
}
halocarbon_species = full.halocarbon_species()
emissions |= {
    species: reference.loc[(f"Emissions|{species}", "World")].to_numpy()
    for species in halocarbon_species
}
print(f"{len(years)} years from {years[0]:.0f} to {years[-1]:.0f}")
print(
    f"{len(full.SECTORAL_EMISSIONS)} sectoral species and {len(halocarbon_species)} halocarbons"
)

# %% tags=["remove_input"]
shown = [
    ("CO2|Fossil", "Fossil CO2 (GtC/yr)"),
    ("CO2|AFOLU", "AFOLU CO2 (GtC/yr)"),
    ("CH4|Fossil", "Fossil CH4 (Mt CH4/yr)"),
    ("N2O|AFOLU", "AFOLU N2O (Mt N/yr)"),
    ("SOx|Fossil", "Fossil SOx (Mt S/yr)"),
    ("NOx|Fossil", "Fossil NOx (Mt N/yr)"),
    ("CFC12", "CFC-12 (kt/yr)"),
    ("HFC134a", "HFC-134a (kt/yr)"),
]
fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharex=True)
for ax, (name, title) in zip(axes.flat, shown):
    ax.plot(years, emissions[name])
    ax.set_title(title, fontsize=10)
fig.suptitle("SSP2-4.5 emissions")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## MAGICC7 parameters
#
# `full.default_parameters()` holds MAGICC7 v7.5.3's default configuration,
# keyed by MAGICC7's own lower-case names such as `core_climatesensitivity`.
# The helpers in `rscm.magicc.full` translate these names into the parameters of each component.
#
# MAGICC7's default run prescribes concentrations until 2015 and only then calculates them from emissions.
# Setting each `<gas>_switchfromconc2emis_year` to the first year makes the whole run emissions-driven,
# so the feedbacks act from 1750.
# Permafrost is on by default.
# `out_sealevel` adds thermal expansion and the glacier and ice sheet contributions.

# %%
parameters = full.default_parameters() | {
    **{
        f"{group}_switchfromconc2emis_year": 1750
        for group in ("co2", "ch4", "n2o", "fgas", "mhalo")
    },
    "out_sealevel": 1,
}
print(f"{len(parameters)} parameters, for example:")
for key in (
    "core_climatesensitivity",
    "core_co2ch4n2o_rfmethod",
    "pf_apply",
    "ncycle_apply",
):
    print(f"  {key} = {parameters[key]!r}")

# %% [markdown]
# ## Prescribed forcing
#
# `full.load_inputs` reads the MAGICC7 input files packaged with RSCM:
# historical aerosol forcing patterns and optical thickness, land-use albedo forcing,
# and solar and monthly volcanic forcing.
# `full.prescribed_forcing` scales the historical patterns with the emissions,
# as MAGICC7 does, and returns four-box ERF for thirteen agents.
# The global mean is the area-weighted mean of the four boxes.

# %%
ERF = "Effective Radiative Forcing"
area = np.array(pr.area_fractions())
inputs = full.load_inputs(parameters)
prescribed = full.prescribed_forcing(years, emissions, parameters, inputs)

# %% tags=["remove_input"]
groups = {
    "Aerosol direct": [
        name for name in prescribed if name.split("|")[0] in ("SOx", "BC", "OC", "NO3")
    ]
    + ["Mineral Dust"],
    "Cloud albedo": ["Cloud Albedo"],
    "Land use and BC on snow": ["Land Use", "BC on Snow"],
    "Solar": ["Solar"],
}
fig, ax = plt.subplots(figsize=(10, 4))
for label, names in groups.items():
    ax.plot(years, sum(prescribed[name] for name in names) @ area, label=label)
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_ylabel("W/m^2")
ax.set_title("Prescribed forcing, global mean")
ax.legend()
plt.show()

# %% [markdown]
# Volcanic forcing is different.
# MAGICC7 applies it monthly inside the climate model, so it is a parameter of `ClimateUDEB`
# rather than an input variable.
#
# ## Configuring the components
#
# Each builder takes a dictionary of parameters keyed by their Rust names.
# Omitted parameters take the MAGICC7 defaults documented on each builder.
# The cells below set the rest from the MAGICC7 configuration, as `run_magicc7` does.
# The helpers in `rscm.magicc.full` handle the components with long parameter lists.
#
# Natural CH4 and N2O emissions are not parameters MAGICC7 reads.
# It derives them from the concentration budget over a window of years
# (1981 to 1991 for N2O, 1994 to 2004 for CH4), and `full.natural_emissions` repeats that calculation.

# %%
p = parameters
natural = full.natural_emissions(years, emissions, p)
print(
    f"Natural emissions: CH4 {natural['CH4']:.2f} Mt CH4/yr, N2O {natural['N2O']:.3f} Mt N/yr"
)

# %% [markdown]
# `full.climate_parameters` carries the climate configuration over,
# including the monthly volcanic forcing from the input files.
# Setting `thermal_expansion` makes `ClimateUDEB` write the thermal expansion that `SeaLevelRise` reads.

# %%
climate_parameters = full.climate_parameters(p, inputs, years[0])
climate_parameters["thermal_expansion"] = {}
climate = ClimateUDEBBuilder.from_parameters(climate_parameters)

# %% [markdown]
# `ForcingAggregation` takes one entry per forcing agent.
# Each applies MAGICC7's efficacy and regional pattern to its agent.
# The halocarbon agents sum their species first.
# `summation` sets MAGICC7's order of addition, which matters for bit-for-bit agreement.

# %%
agents = full.forcing_agents(p, years, prescribed, emissions)


def with_prefix(terms):
    """Prefix nested agent names with the ERF variable name."""
    return [with_prefix(t) if isinstance(t, list) else f"{ERF}|{t}" for t in terms]


aggregation = ForcingAggregationBuilder.from_parameters(
    {
        "agents": agents,
        "efficacy_apply": p["rf_efficacy_apply"],
        "climate": full.efficacy_climate(climate_parameters),
        "summation": with_prefix(full.SUMMATION),
    }
)

# %% [markdown]
# Permafrost and the chemistry come next.
# Setting `permafrost` makes the CH4 components read the methane that thawing permafrost releases.
# Both CH4 components repeat the CH4 budget, so they share its mass conversion and natural emissions.

# %%
ch4_mass = {"ppb_to_tg": p["ch4_ppb2tgch4"], "mixbox_size": p["ch4_mixboxsize"]}
builders = {
    "Permafrost": PermafrostBuilder.from_parameters(full.permafrost_parameters(p)),
    "CH4Chemistry": CH4ChemistryBuilder.from_parameters(
        {
            "permafrost": True,
            "tau_total_init": p["ch4_tautot_init"],
            "natural_emissions": natural["CH4"],
            **ch4_mass,
        }
    ),
    "CH4Oxidation": CH4OxidationBuilder.from_parameters(
        {
            "permafrost": True,
            "permafrost_fraction": p["pf_co2fromch4oxinatm_fraction"],
            "natural_emissions": natural["CH4"],
            **ch4_mass,
        }
    ),
    "N2OChemistry": N2OChemistryBuilder.from_parameters(
        {
            "tau_n2o": p["n2o_tauinit"],
            "ppb_to_tg": p["n2o_ppb2tgn"],
            "natural_emissions": natural["N2O"],
        }
    ),
    "HalocarbonChemistry": HalocarbonChemistryBuilder.from_parameters(
        {
            "fgas_regions": {
                "rf_regions": p["rf_regions_fgas"],
                "lower_tau": p["fgas_rf_regionscaling_lowertau"],
                "upper_tau": p["fgas_rf_regionscaling_uppertau"],
            }
        }
    ),
}

# %% [markdown]
# The carbon cycle, forcing and climate complete the chain.
# `TerrestrialCarbon` takes its preindustrial pools and temperature feedbacks from MAGICC7's `co2_*` names.
# `GhgForcing` uses the OLBL formulas (after Etminan et al., 2016), MAGICC7's default for CO2, CH4 and N2O.

# %%
initial_concentrations = full.default_initial_concentrations()
terrestrial = {
    **{
        f"{pool}_pool_pi": p[f"co2_{pool}pool_initial"]
        for pool in ("plant", "detritus", "soil")
    },
    **{
        f"{ours}_temp_sensitivity": p[f"co2_feedbackfactor_{theirs}"]
        for ours, theirs in full.TERRESTRIAL_FEEDBACKS.items()
    },
    "npp_pi": p["co2_npp_initial"],
    "fertilization_method": p["co2_fertilization_method"],
    "plantbox_resp_fertscale": p["co2_plantboxresp_fertscale"],
}
builders |= {
    "TerrestrialCarbon": TerrestrialCarbonBuilder.from_parameters(terrestrial),
    "OceanCarbon": OceanCarbonBuilder.from_parameters({"permafrost": True}),
    "CO2Budget": CO2BudgetBuilder.from_parameters(
        {"permafrost": True, "ocean_end_of_year": True}
    ),
    "GhgForcing": GhgForcingBuilder.from_parameters(
        {
            **{
                f"{gas.lower()}_pi": initial_concentrations[gas]
                for gas in ("CO2", "CH4", "N2O")
            },
            "method": "Olbl",
            "ch4_strat_h2o_fraction": p["ch4_addedstrath2o_percent"],
        }
    ),
    "OzoneForcing": OzoneForcingBuilder.from_parameters(full.ozone_parameters(p)),
    "ForcingAggregation": aggregation,
    "ClimateUDEB": climate,
    "SeaLevelRise": SeaLevelRiseBuilder.from_parameters({}),
}
components = {name: builder.build() for name, builder in builders.items()}

# %% [markdown]
# ## Inspecting component interfaces
#
# `definitions()` on a built component lists the variables it reads and writes:
#
# - **Input**: read from another component or from prescribed data.
# - **Output**: written each step.
# - **State**: read at the start of the step and written at its end, so it needs an initial value.
#
# The names are what couple components.
# For example `CO2Budget` writes `Atmospheric Concentration|CO2`, which `GhgForcing` reads:

# %%
pd.DataFrame(
    [(d.name, d.role.name, d.unit) for d in components["CO2Budget"].definitions()],
    columns=["Variable", "Type", "Unit"],
)

# %% [markdown]
# Counting each component's variables shows where the bulk of them sits.
# `HalocarbonChemistry` tracks 41 species, each with its own emissions input, concentration state
# and forcing output.
# `ForcingAggregation` reads every forcing agent and writes the total ERF
# and the four-box `Effective Radiative Forcing|Efficacy Adjusted` that drives `ClimateUDEB`.

# %% tags=["remove_input"]
kinds = (VariableRole.Input, VariableRole.Output, VariableRole.State)
pd.DataFrame(
    {
        kind.name: [
            sum(d.role == kind for d in c.definitions()) for c in components.values()
        ]
        for kind in kinds
    },
    index=list(components),
)

# %% [markdown]
# ## Feedback loops and lags
#
# MAGICC has feedback loops:
#
# - Temperature from `ClimateUDEB` changes permafrost thaw, the CH4, N2O and halocarbon lifetimes,
#   land and ocean carbon uptake and ozone forcing, which all change the forcing that drives temperature.
# - CO2 from `CO2Budget` drives fertilisation on land and air-sea exchange,
#   which set the fluxes that `CO2Budget` integrates.
# - The CO2-doubling forcing and climate sensitivity set the internal efficacies in `ForcingAggregation`.
#
# The builder links every component to the producers of its inputs, whatever order they are added in.
# A loop must lag one of its edges, so the reader sees the value from the previous step.
# `with_lag(variable)` lags every reader of a variable,
# and `with_lag_for(variable, consumer)` lags one.
# Building a model with a loop that no lag breaks fails, and the error suggests the lags that would break it.
#
# MAGICC7 uses a staggered scheme.
# Permafrost, chemistry, the carbon cycle and ozone use the temperature at the start of each year,
# and the land and ocean the CO2 at the start of the year,
# while forcing uses the concentrations at its end.
# `SeaLevelRise` reads the annual mean warming of the same year.

# %%
lag = ["Surface Temperature"]
lag_for = [
    ("Atmospheric Concentration|CO2", "TerrestrialCarbon"),
    ("Atmospheric Concentration|CO2", "OceanCarbon"),
    ("Equilibrium Climate Sensitivity", "ForcingAggregation"),
]

# %% [markdown]
# Listing every input these lags apply to gives the lagged couplings:


# %% tags=["remove_input"]
def lagged_couplings(components):
    """List the inputs that read the previous step's value from another component."""
    producers = {
        d.name: name
        for name, component in components.items()
        for d in component.definitions()
        if d.role in (VariableRole.Output, VariableRole.State)
    }
    return [
        (producers[d.name], d.name, reader)
        for reader, component in components.items()
        for d in component.definitions()
        if d.role == VariableRole.Input
        and d.name in producers
        and (d.name in lag or (d.name, reader) in lag_for)
    ]


lagged = lagged_couplings(components)
(
    pd.DataFrame(lagged, columns=["Producer", "Variable", "Read by (start of step)"])
    .groupby(["Producer", "Variable"])["Read by (start of step)"]
    .agg(", ".join)
    .reset_index()
)

# %% [markdown]
# ## Prescribed inputs and the schema
#
# Inputs that no component produces come from prescribed timeseries.
# The emissions are scalar, with a few totals and renamings the components expect.
# The prescribed forcing is four-box.

# %%
SECTORAL_NAMES = {"CO2|AFOLU": "CO2|Land Use"}
prescribed_emissions = {
    **{
        f"Emissions|{SECTORAL_NAMES.get(name, name)}": (
            values,
            full.SECTORAL_EMISSIONS[name.split("|")[0]],
        )
        for name, values in emissions.items()
        if name.split("|")[0] in ("CO2", "CH4", "NOx", "CO", "NMVOC")
    },
    **{
        f"Emissions|{gas}": (
            emissions[f"{gas}|Fossil"] + emissions[f"{gas}|AFOLU"],
            full.SECTORAL_EMISSIONS[gas],
        )
        for gas in ("CH4", "N2O", "NOx", "CO", "NMVOC")
    },
    **{f"Emissions|{s}": (emissions[s], "kt/yr") for s in halocarbon_species},
}
prescribed_erf = {f"{ERF}|{name}": values for name, values in prescribed.items()}

# %% [markdown]
# A `VariableSchema` declares every variable with its unit and grid.
# A variable is four-box if any component writes or reads it on the four-box grid.
# Components that read a four-box variable as a scalar, such as `Surface Temperature` in the chemistry,
# get the area-weighted mean using the grid weights set on the builder.
# A hidden cell collects `grids`, the unit and grid of each variable, from the inputs and the definitions.

# %% tags=["remove_input"]
grids = {name: ("W/m^2", GridType.FourBox) for name in prescribed_erf}
grids |= {
    name: (unit, GridType.Scalar) for name, (_, unit) in prescribed_emissions.items()
}
for component in components.values():
    for d in component.definitions():
        _, grid = grids.get(d.name, (d.unit, d.grid))
        four_box = GridType.FourBox in (grid, d.grid)
        grids[d.name] = (d.unit, GridType.FourBox if four_box else GridType.Scalar)

# %%
schema = VariableSchema()
for name, (unit, grid) in grids.items():
    schema.add_variable(name, unit, grid)
four_box_count = sum(g == GridType.FourBox for _, g in grids.values())
print(
    f"{len(prescribed_emissions)} scalar and {len(prescribed_erf)} four-box prescribed inputs"
)
print(f"{len(grids)} variables, {four_box_count} on four boxes")

# %% [markdown]
# ## Initial values
#
# Every state needs a value at the first boundary.
#
# - Concentrations start at MAGICC7's 1750 values, which are not zero for some halocarbons.
# - CH4 starts as a natural burden only.
# - The ocean starts in equilibrium with the initial CO2,
#   and the land pools start at their MAGICC7 preindustrial sizes.
# - Sea level rise starts at zero.

# %%
ch4_burden = initial_concentrations["CH4"] * p["ch4_ppb2tgch4"] * p["ch4_mixboxsize"]
initial_values = {
    **{
        f"Atmospheric Concentration|{gas}": c
        for gas, c in initial_concentrations.items()
    },
    **{f"Atmospheric Burden|CH4|{s}": 0.0 for s in ("Fossil", "AFOLU", "Permafrost")},
    "Atmospheric Burden|CH4|Natural": ch4_burden,
    "Ocean Surface pCO2": initial_concentrations["CO2"],
    "Cumulative Ocean Uptake": 0.0,
    **{
        f"Carbon Pool|{pool.title()}": terrestrial[f"{pool}_pool_pi"]
        for pool in ("plant", "detritus", "soil")
    },
    **{
        d.name: 0.0
        for d in components["SeaLevelRise"].definitions()
        if d.role == VariableRole.Output
    },
}

# %% [markdown]
# The climate needs the forcing at the first boundary.
# `ForcingAggregation.aggregate` turns one value per agent, `initial_agents`,
# into the total and four-box forcing,
# and `ClimateUDEB.initial_values` gives the matching climate state.
# The prescribed agents take their first-year forcing and the calculated agents start at zero,
# apart from F-gases, which the next section fills in.

# %% tags=["remove_input"]
first_year = {
    **prescribed_erf,
    f"{ERF}|{full.BIOMASS}": sum(prescribed[name] for name in full.BIOMASS_AGENTS),
}
initial_agents = [
    list(first_year.get(agent["variable"], np.zeros((1, 4)))[0])
    if agent["regional"]
    else [0.0]
    for agent in agents
]

# %% [markdown]
# ## Building the model
#
# The builder steps are in a function so that a short run can reuse them.
# Prescribed values use linear interpolation between the annual labels.


# %%
def build_model(n_years, initial_agents):
    """Build the coupled model over the first `n_years` years."""
    axis = TimeAxis.from_values(years[:n_years])
    builder = (
        ModelBuilder()
        .with_time_axis(axis)
        .with_schema(schema)
        .with_grid_weights(GridType.FourBox, list(area))
    )
    for component in components.values():
        builder.with_component(component)
    for variable in lag:
        builder.with_lag(variable)
    for variable, consumer in lag_for:
        builder.with_lag_for(variable, consumer)
    for name, (values, unit) in prescribed_emissions.items():
        builder.with_prescribed_variable(
            name, Timeseries(values[:n_years], axis, unit, InterpolationStrategy.Linear)
        )
    for name, values in prescribed_erf.items():
        builder.with_prescribed_variable(
            name,
            FourBoxTimeseries(
                values[:n_years], axis, "W/m^2", InterpolationStrategy.Linear
            ),
        )
    total, boxes = aggregation.aggregate(initial_agents)
    builder.with_initial_values({**initial_values, **climate.initial_values(total)})
    builder.with_initial_grid_values({f"{ERF}|Efficacy Adjusted": list(boxes)})
    return builder.build()


# %% [markdown]
# MAGICC7 starts from the second year's F-gas forcing.
# A two-year run of the same model supplies it before the full model is built.

# %%
fgas = [agent["variable"] for agent in agents].index(f"{ERF}|F-Gases")
two_years = build_model(2, initial_agents)
two_years.run()
fgas_boxes = two_years.timeseries().get_four_box_timeseries_by_name(f"{ERF}|F-Gases")
initial_agents[fgas] = list(fgas_boxes.values()[1])

model = build_model(len(years), initial_agents)

# %% [markdown]
# ## The component graph
#
# `model.as_dot()` returns the dependency graph in Graphviz format.
# Its raw labels include every parameter, so a hidden helper, `tidy_graph`, redraws it.
# Colours mark the domains: green for chemistry, blue for the carbon cycle and permafrost,
# orange for forcing and pink for the climate.
# Solid arrows are couplings within a step.
# Red dashed arrows are the lagged couplings from the table above.
# They are not part of the solved graph, which is why the graph has no loops.
# Rendering requires Graphviz to be installed.

# %% tags=["remove_input"]
DOMAINS = {
    "#c8e6c9": ["CH4Chemistry", "CH4Oxidation", "N2OChemistry", "HalocarbonChemistry"],
    "#bbdefb": ["Permafrost", "TerrestrialCarbon", "OceanCarbon", "CO2Budget"],
    "#ffe0b2": ["GhgForcing", "OzoneForcing", "ForcingAggregation"],
    "#f8bbd0": ["ClimateUDEB", "SeaLevelRise"],
}
COLOURS = {name: colour for colour, names in DOMAINS.items() for name in names}


def tidy_graph(model, components, lagged):
    """Component graph coloured by domain, with lagged couplings as dashed edges."""
    raw = pydot.graph_from_dot_data(model.as_dot())[0]
    names = {
        node.get_name(): node.get_label().strip('"').split(" ")[0].split("{")[0]
        for node in raw.get_nodes()
        if node.get_label()
    }
    nodes = {label: node for node, label in names.items() if label in components}
    edges = {
        (edge.get_source(), edge.get_destination())
        for edge in raw.get_edges()
        if edge.get_label().strip('"') and names.get(edge.get_source()) in nodes
    }
    lags = {(nodes[producer], nodes[reader]) for producer, _, reader in lagged}

    graph = pydot.Dot(graph_type="digraph", nodesep="0.2", ranksep="0.35")
    for label, node in nodes.items():
        graph.add_node(
            pydot.Node(
                node,
                label=label,
                shape="box",
                style="filled,rounded",
                fillcolor=COLOURS[label],
                fontsize="16",
            )
        )
    for source, target in sorted(edges):
        graph.add_edge(pydot.Edge(source, target))
    for source, target in sorted(lags):
        graph.add_edge(
            pydot.Edge(
                source, target, style="dashed", color="#c62828", constraint="false"
            )
        )
    return graph


def show_svg(graph):
    """Display a graph as SVG that scales to the page width."""
    svg = graph.create_svg().decode()
    size = r'<svg width="[^"]*" height="[^"]*"'
    display(SVG(re.sub(size, "<svg", svg, count=1)))


# %%
graph = tidy_graph(model, components, lagged)
show_svg(graph)

# %% [markdown]
# Reading the graph:
#
# - `Permafrost`, `N2OChemistry`, `TerrestrialCarbon` and `CH4Chemistry` read only prescribed emissions
#   and lagged values, apart from the permafrost CH4 that `CH4Chemistry` takes from `Permafrost`.
# - The carbon chain runs from `TerrestrialCarbon` through `OceanCarbon` to `CO2Budget`.
#   The ocean needs every CO2 source because it evolves atmospheric CO2 monthly within the year.
# - `GhgForcing` waits for all three concentrations, and `OzoneForcing` waits for CH4, N2O and EESC.
# - `ForcingAggregation` collects every agent, including the prescribed ones,
#   into the forcing that drives `ClimateUDEB`.
#   The prescribed emissions and forcing, and the monthly volcanic forcing, are not drawn.
# - `SeaLevelRise` reads the annual mean warming and thermal expansion from `ClimateUDEB`.
# - The red arrows close the loops one step later.
#   Temperature reaches seven components this way, CO2 reaches the land and ocean,
#   and climate sensitivity reaches `ForcingAggregation`.

# %% [markdown]
# ## Running the model
#
# `series` reads a variable from the results, taking the area-weighted global mean of four-box variables.

# %%
model.run()
results = model.timeseries()


def series(name):
    """Return a variable from the hand-built run, as the global mean for four-box variables."""
    scalar = results.get_timeseries_by_name(name)
    if scalar is not None:
        return scalar.values()
    return results.get_four_box_timeseries_by_name(name).values() @ area


regional_temperature = results.get_four_box_timeseries_by_name(
    "Surface Temperature"
).values()
global_temperature = regional_temperature @ area
print(f"Warming in 2100: {global_temperature[years == 2100][0]:.2f} K")

# %% [markdown]
# ### Agreement with MAGICC7
#
# MAGICC7 v7.5.3 was run with the same emissions and configuration.
# Its results are in the reference file next to the one with the emissions.
# The largest relative differences over the whole run are at the level of floating-point rounding,
# far inside the 1e-9 tolerance the regression tests use.

# %%
magicc = reference_run("02_ssp245_emissions_permafrost")


def magicc_series(name):
    """Return a global variable from the MAGICC7 run."""
    return magicc.loc[(name, "World")].to_numpy()


differences = {
    **{
        gas: series(f"Atmospheric Concentration|{gas}")
        / magicc_series(f"Atmospheric Concentrations|{gas}")
        - 1
        for gas in ("CO2", "CH4", "N2O")
    },
    "Surface temperature": (global_temperature - magicc_series("Surface Temperature"))
    / np.abs(magicc_series("Surface Temperature")).max(),
}
largest = pd.Series(
    {k: np.abs(v).max() for k, v in differences.items()},
    name="Largest relative difference",
)
assert (largest < 1e-9).all()
largest.map("{:.1e}".format).to_frame()

# %% [markdown]
# ### Concentrations
#
# CO2 and CH4 stay close to their 1750 values until anthropogenic emissions grow in the nineteenth century.
# N2O falls by about 20 ppb first.
# Its natural emissions close the budget over 1981 to 1991.
# At 8.04 Mt N/yr they are below the 1750 sink of about 9.4 Mt N/yr,
# so N2O declines until anthropogenic emissions grow.
# MAGICC7 shows the same dip, and its dashed lines sit on top of the RSCM ones.
# EESC (equivalent effective stratospheric chlorine), which drives stratospheric ozone depletion,
# peaks after the Montreal Protocol cuts emissions.

# %% tags=["remove_input"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
for ax, gas, unit in zip(axes, ["CO2", "CH4", "N2O"], ["ppm", "ppb", "ppb"]):
    ax.plot(years, series(f"Atmospheric Concentration|{gas}"), label="RSCM")
    ax.plot(
        years,
        magicc_series(f"Atmospheric Concentrations|{gas}"),
        "k--",
        linewidth=1,
        label="MAGICC7",
    )
    ax.set_title(f"{gas} ({unit})")
axes[0].legend()
axes[3].plot(years, series("EESC"))
axes[3].set_title("EESC (ppt)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Forcing by agent
#
# The stacked areas are each agent's forcing before efficacies,
# positive agents above zero and negative agents below it.
# The black line is the total ERF from `ForcingAggregation`, after efficacies.
# Volcanic forcing is not in either, because it enters `ClimateUDEB` monthly.

# %% tags=["remove_input"]
ozone = (
    "Stratospheric Ozone",
    "Tropospheric Ozone",
    "Ozone due to N2O",
    "Ozone due to Temperature",
)
forcing = {
    "CO2": series(f"{ERF}|CO2"),
    "CH4": series(f"{ERF}|CH4"),
    "N2O": series(f"{ERF}|N2O"),
    "Halocarbons": sum(
        series(name)
        for group in full.HALOCARBON_GROUPS
        for name in full.halocarbon_forcing(group)
    ),
    "Ozone": sum(series(f"{ERF}|{name}") for name in ozone),
    "Stratospheric H2O": series(f"{ERF}|CH4 Oxidation Stratospheric H2O"),
    **{
        label: sum(prescribed[name] for name in names) @ area
        for label, names in groups.items()
    },
}
fig, ax = plt.subplots(figsize=(10, 5))
positive = np.zeros(len(years))
negative = np.zeros(len(years))
for label, values in forcing.items():
    up, down = np.clip(values, 0, None), np.clip(values, None, 0)
    colour = ax.fill_between(
        years, positive, positive + up, label=label
    ).get_facecolor()
    ax.fill_between(years, negative, negative + down, color=colour)
    positive += up
    negative += down
ax.plot(years, series(ERF), color="black", label="Total ERF")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_ylabel("W/m^2")
ax.set_title("Effective radiative forcing by agent")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# ### Surface temperature
#
# `ClimateUDEB` resolves four boxes.
# Northern land warms fastest and the southern ocean slowest.
# The dips are volcanic eruptions.

# %% tags=["remove_input"]
fig, ax = plt.subplots(figsize=(10, 4))
for i, box in enumerate(
    ["Northern ocean", "Northern land", "Southern ocean", "Southern land"]
):
    ax.plot(years, regional_temperature[:, i], label=box, linewidth=1)
ax.plot(
    years,
    global_temperature,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Global (area-weighted)",
)
ax.set_ylabel("K relative to 1750")
ax.set_title("Surface temperature")
ax.legend()
plt.show()

# %% [markdown]
# ### Carbon cycle and permafrost
#
# The left panel shows the sources and sinks that `CO2Budget` integrates.
# Positive land and ocean fluxes take carbon out of the atmosphere.
# The net land flux includes land-use emissions, so it is negative until the mid-twentieth century.
# The middle panel shows the carbon that thawing permafrost releases as CO2 and CH4.
# The right panel shows the lifetimes, which respond to temperature, to the concentrations themselves
# and, for CH4, to NOx, CO and NMVOC emissions.

# %% tags=["remove_input"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name, label in [
    ("Emissions|CO2|Fossil", "Fossil emissions"),
    ("Carbon Flux|Terrestrial", "Net land uptake"),
    ("Carbon Flux|Ocean", "Ocean uptake"),
]:
    axes[0].plot(years, series(name), label=label)
axes[0].axhline(0, color="grey", linewidth=0.5)
axes[0].set_ylabel("GtC/yr")
axes[0].set_title("Carbon fluxes")
axes[0].legend()
axes[1].plot(years, series("Emissions|CO2|Permafrost"), label="CO2 (GtC/yr)")
axes[1].plot(
    years, series("Emissions|CH4|Permafrost") / 100, label="CH4 (100 Mt CH4/yr)"
)
axes[1].set_title("Permafrost emissions")
axes[1].legend()
axes[2].plot(years, series("Lifetime|CH4"), label="CH4")
axes[2].plot(years, series("Lifetime|N2O") / 10, label="N2O / 10")
axes[2].set_title("Lifetimes (yr)")
axes[2].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Sea level rise
#
# `ClimateUDEB` calculates thermal expansion from the ocean temperature profile.
# `SeaLevelRise` adds glaciers and the Greenland and Antarctic ice sheets.
# MAGICC7 counts sea level rise from 1850, so every term is zero before then.

# %% tags=["remove_input"]
fig, ax = plt.subplots(figsize=(10, 4))
for name in (
    "Thermal Expansion",
    "Glaciers",
    "Greenland|Surface Mass Balance",
    "Greenland|Solid Ice Discharge",
    "Antarctica|Surface Mass Balance",
    "Antarctica|Solid Ice Discharge",
):
    ax.plot(years, series(f"Sea Level Rise|{name}"), label=name)
ax.plot(years, series("Sea Level Rise"), color="black", label="Total")
ax.set_ylabel("mm relative to 1750")
ax.set_title("Sea level rise")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## The convenience path: `run_magicc7`
#
# `rscm.magicc.run_magicc7` builds this whole chain for you, with the same lags.
# It takes the emissions and MAGICC7 parameter names, and returns the run model.
# With the same parameters it gives the same result as the hand-built model.

# %%
convenience = run_magicc7(years, emissions, parameters=parameters)
convenience_temperature = (
    convenience.timeseries()
    .get_four_box_timeseries_by_name("Surface Temperature")
    .values()
)
print(
    "Largest temperature difference: "
    f"{np.abs(convenience_temperature - regional_temperature).max():.1e} K"
)
np.testing.assert_array_equal(convenience_temperature, regional_temperature)

# %% [markdown]
# Without `parameters`, `run_magicc7` uses MAGICC7's default configuration.
# It prescribes concentrations from MAGICC7's packaged files until 2015
# and calculates them from emissions after.
# Natural emissions close the budget against those observed concentrations,
# so CO2, CH4 and N2O follow the observations until 2015 rather than drifting.
#
# Overriding single parameters runs experiments.
# A small sweep over `core_climatesensitivity` (ECS) shows the carbon-cycle feedback at work:
# a more sensitive climate warms more, which weakens the land and ocean sinks and leaves more CO2 in the air.
# The sweep uses the emissions-driven configuration so that CO2 responds from 1750.

# %%
default = run_magicc7(years, emissions).timeseries()
sweep = {}
for ecs in (2.0, 3.0, 4.5):
    run = run_magicc7(
        years, emissions, parameters=parameters | {"core_climatesensitivity": ecs}
    )
    sweep[ecs] = run.timeseries()

# %% tags=["remove_input"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for gas, unit, ax in zip(("CO2", "N2O"), ("ppm", "ppb"), axes[:2]):
    name = f"Atmospheric Concentration|{gas}"
    ax.plot(years, series(name), label="Emissions-driven")
    ax.plot(
        years, default.get_timeseries_by_name(name).values(), label="MAGICC7 default"
    )
    ax.set_title(f"{gas} concentration ({unit})")
    ax.legend()
for ecs, timeseries in sweep.items():
    boxes = timeseries.get_four_box_timeseries_by_name("Surface Temperature").values()
    axes[2].plot(years, boxes @ area, label=f"ECS {ecs} K")
axes[2].set_title("Surface temperature (K)")
axes[2].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## The smaller runners
#
# `rscm.magicc` also has `run_magicc` and `run_magicc_emissions`, which couple only part of the chain
# and take one global series for all other forcing.
# [MAGICC runs](../../magicc_runs/) compares the three runners.
#
# Feeding `run_magicc` the concentrations from the hand-built run, with the rest of the forcing as one series,
# shows the size of that simplification.
# Outputs calculated during a step have no value in the first year,
# so the first year of the other forcing comes from the initial forcing instead.

# %%
ghg_forcing = sum(series(f"{ERF}|{gas}") for gas in ("CO2", "CH4", "N2O"))
other_forcing = series(ERF) - ghg_forcing
other_forcing[0], _ = aggregation.aggregate(initial_agents)
result = run_magicc(
    years,
    {gas: series(f"Atmospheric Concentration|{gas}") for gas in ("CO2", "CH4", "N2O")},
    other_forcing=other_forcing,
    forcing_parameters={
        f"{gas.lower()}_pi": initial_concentrations[gas]
        for gas in ("CO2", "CH4", "N2O")
    },
    climate_parameters={"ecs": parameters["core_climatesensitivity"]},
)

# %% [markdown]
# Without monthly volcanic forcing, its temperature misses the eruptions but otherwise stays close.

# %% tags=["remove_input"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, global_temperature, label="Full chain")
ax.plot(result.years, result.values["Surface Temperature"], label="run_magicc")
ax.set_ylabel("K relative to 1750")
ax.set_title("Surface temperature")
ax.legend()
plt.show()

# %% [markdown]
# ## What is validated
#
# This notebook checks one configuration, the emissions-driven SSP2-4.5 run with permafrost.
# The regression tests hold `run_magicc7` to MAGICC7 in several more,
# and [Agreement with MAGICC7](../../magicc_runs/#agreement-with-magicc7) lists them.
# The [known gaps](../../modules/#known-gaps) list what the port leaves out.
