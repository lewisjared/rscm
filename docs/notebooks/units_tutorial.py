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
# # RSCM Units Module
#
# The RSCM units module parses, normalises and converts the physical units used in climate modelling.
#
# ## Features
#
# - **Flexible parsing**: `W/m^2`, `W / m ^ 2` and `W m^-2` are the same unit.
# - **Dimensional analysis**: conversions are only allowed between units of the same dimension.
# - **Species**: a mass unit can name what it counts, as in `Mt CH4/yr` or `Gt C/yr`.
# - **Climate-specific units**: concentrations (ppm, ppb), radiative forcing (W/m²) and emission rates (GtC/yr).
# - **Conversion factors**: multipliers between compatible units,
#   including molecular mass ratios such as CO₂ to C (44/12).

# %%
import numpy as np

from rscm.components import CarbonCycleBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    Unit,
    VariableSchema,
)

# %% [markdown]
# ## Parsing Units
#
# The `Unit` class parses unit strings with flexible syntax.
# Whitespace is normalised and several notations are supported.

# %%
units = [Unit(s) for s in ["W/m^2", "W / m ^ 2", "W m^-2", "W per m^2"]]
for u in units:
    print(f"{u.original!r:14} -> {u}")

assert all(u == units[0] for u in units)

# %% [markdown]
# ### Supported Syntax
#
# The parser accepts several equivalent notations:
#
# | Notation | Meaning |
# |----------|---------|
# | `m^2`, `m**2`, `m2` | Square metres |
# | `W/m^2`, `W m^-2`, `W per m^2` | Watts per square metre |
# | `kg m`, `kg*m`, `kg·m` | Kilogram-metres |
# | `Gt C/yr`, `GtC / yr` | Gigatonnes of carbon per year |
# | `Mt CO2/yr`, `MtCO2/yr` | Megatonnes of CO₂ per year |
# | `Mt CH4/yr` | Megatonnes of methane per year |

# %%
for notation in ["m^2", "m**2", "m2", "W/m^2", "W m^-2", "W per m^2"]:
    print(f"{notation!r:12} -> {Unit(notation).normalised()}")

# %% [markdown]
# ### Original vs Normalised
#
# The `Unit` class keeps the original input string and provides a normalised form for comparison.
# The normalised form is compact, with no spaces around `/` or `^`.
# Carbon is fused to its mass unit (`GtC/yr`) and every other species is spaced (`Mt CO2/yr`).

# %%
for s in ["  GtC  /  yr  ", "Gt C / yr", "MtCO2/yr", "Mt CH4 / yr", "Mt NMVOC/yr"]:
    u = Unit(s)
    print(f"{u.original!r:18} -> {u.normalised()!r}")

# %% [markdown]
# ## Climate-Specific Units
#
# The module includes units commonly used in climate modelling:
#
# | Category | Units |
# |----------|-------|
# | Mass prefixes | `Gt` (giga-tonne), `Mt` (mega-tonne), `kt`, `t`, etc. |
# | Concentrations | `ppm`, `ppb`, `ppt` |
# | Time | `yr` (365.25 days), `day`, `h`, `min`, `s` |
# | Energy/Power | `W`, `J` with SI prefixes |
# | Temperature | `K`, `degC` (for differences) |
#
# ## Species
#
# A species token after a mass unit names what the mass counts: `Mt CH4/yr` is megatonnes of methane per year.
# The carbon family may also be written fused, so `Gt C/yr` and `GtC/yr` are the same unit.
#
# Species group into families, and units of one family convert by molecular mass ratio:
#
# | Family | Species | Ratio |
# |--------|---------|-------|
# | Carbon | `C`, `CO2` | 44/12 |
# | Sulfur | `S`, `SO2` | 64/32 |
# | Nitrogen oxides | `N`, `NO2`, `NOx` (counted as NO₂) | 46/14 |
# | Nitrous oxide | `N2ON` (the nitrogen in N₂O), `N2O` | 44/28 |
#
# Every other species converts only to itself:
# `CH4`, `CO`, `NH3`, `BC`, `OC`, `VOC` (or `NMVOC`) and the halocarbons, such as `CFC11` and `HFC134a`.
# `HFC4310` is a synonym of `HFC4310mee`.

# %%
for source, target in [
    ("Mt S/yr", "Mt SO2/yr"),
    ("Mt N/yr", "Mt NO2/yr"),
    ("Mt N2ON/yr", "kt N2O/yr"),
    ("Mt NMVOC/yr", "Mt VOC/yr"),
]:
    factor = Unit(source).conversion_factor(Unit(target))
    print(f"{source:<11} -> {target:<10} {factor:>10.4f}")

# %% [markdown]
# Units of different species never convert, even though both are a mass per time.
# An unknown species is a parse error that names the token.

# %%
print(Unit("Mt CH4/yr").is_compatible(Unit("Mt CO2/yr")))

try:
    Unit("Mt XYZ/yr")
except ValueError as e:
    print(f"Error: {e}")

# %% [markdown]
# ## Checking Compatibility
#
# Units are compatible if they have the same physical dimension and count species of the same family.
# `GtC/yr` and `MtCO2/yr` are both carbon-family mass per time, so one converts to the other.

# %%
gtc_yr = Unit("GtC/yr")
mtco2_yr = Unit("MtCO2/yr")
flux = Unit("W/m^2")

print(f"GtC/yr compatible with MtCO2/yr? {gtc_yr.is_compatible(mtco2_yr)}")
print(f"GtC/yr compatible with W/m^2?    {gtc_yr.is_compatible(flux)}")

# %% [markdown]
# ### Dimensionless Units
#
# Concentrations such as ppm and ppb are dimensionless.

# %%
ppm = Unit("ppm")
ppb = Unit("ppb")
watt = Unit("W")

print(f"ppm is dimensionless: {ppm.is_dimensionless()}")
print(f"ppb is dimensionless: {ppb.is_dimensionless()}")
print(f"W is dimensionless:   {watt.is_dimensionless()}")
print(f"ppm compatible with ppb: {ppm.is_compatible(ppb)}")

# %% [markdown]
# ## Unit Conversion
#
# `conversion_factor()` returns the multiplier that converts a value from one unit to another.
# From GtC/yr to MtCO2/yr it combines the prefix change (Gt to Mt, 1000)
# with the molecular weight ratio of CO₂ to C (44/12).

# %%
factor = gtc_yr.conversion_factor(mtco2_yr)
print(f"GtC/yr to MtCO2/yr: {factor:.2f}")

# %% [markdown]
# `convert()` converts a value.

# %%
value_mtco2 = gtc_yr.convert(10.0, mtco2_yr)
print(f"10.0 GtC/yr = {value_mtco2:.1f} MtCO2/yr")
print(f"{value_mtco2:.1f} MtCO2/yr = {mtco2_yr.convert(value_mtco2, gtc_yr)} GtC/yr")

# %% [markdown]
# ## Common Conversions
#
# The unit system supports the SI prefixes from yocto (10⁻²⁴) to yotta (10²⁴).
# Some conversions that come up in climate modelling:

# %%
conversions = [
    ("GtC", "MtCO2"),
    ("GtCO2", "GtC"),
    ("GtC/yr", "GtC/s"),
    ("km", "m"),
    ("ppm", "ppb"),
    ("TW", "W"),
    ("GW", "MW"),
    ("mW", "W"),
]

print(f"{'From':<8} {'To':<8} {'Factor':>12}")
for from_str, to_str in conversions:
    factor = Unit(from_str).conversion_factor(Unit(to_str))
    print(f"{from_str:<8} {to_str:<8} {factor:>12.6g}")

# %% [markdown]
# ## Carbon-CO₂ Conversions
#
# Climate models often convert between carbon mass and CO₂ mass.
# The molecular mass ratio is 44/12 ≈ 3.667:
#
# - **C → CO₂**: multiply by 44/12.
# - **CO₂ → C**: multiply by 12/44.
#
# The two factors multiply to 1.

# %%
gtc = Unit("GtC")
gtco2 = Unit("GtCO2")

c_to_co2 = gtc.conversion_factor(gtco2)
co2_to_c = gtco2.conversion_factor(gtc)

print(f"C   -> CO₂: {c_to_co2:.6f}")
print(f"CO₂ -> C:   {co2_to_c:.6f}")
assert np.isclose(c_to_co2 * co2_to_c, 1.0)

# %% [markdown]
# ### Practical Example: Emissions Pathway Conversion
#
# Climate scenarios often provide emissions in GtCO₂/yr, but carbon cycle models may use GtC/yr internally.

# %%
years = [2020, 2030, 2040, 2050]
emissions_gtco2 = [40.0, 35.0, 25.0, 10.0]

gtco2_yr = Unit("GtCO2/yr")
emissions_gtc = [gtco2_yr.convert(e, gtc_yr) for e in emissions_gtco2]

print(f"{'Year':<6} {'GtCO2/yr':>10} {'GtC/yr':>10}")
for year, e_co2, e_c in zip(years, emissions_gtco2, emissions_gtc):
    print(f"{year:<6} {e_co2:>10.1f} {e_c:>10.2f}")

# %% [markdown]
# ## Error Handling
#
# Converting between incompatible units raises a `ValueError`.

# %%
for target in [flux, Unit("Mt CH4/yr")]:
    try:
        gtc_yr.conversion_factor(target)
    except ValueError as e:
        print(f"Error: {e}")

# %% [markdown]
# ## Integration with Model Building
#
# When a model has a `VariableSchema`, the builder compares each component's declared units with the schema:
#
# 1. **Validates compatibility**: the schema unit and component unit must have the same dimension
#    and count the same species family.
# 2. **Calculates conversion factors**: the builder computes the multiplier needed at runtime.
# 3. **Applies conversion at runtime**: values are converted as Rust components read their inputs.
#
# The builder also converts prescribed data into the unit the model stores it in:
# the schema's unit, or without a schema the unit the components declare.
# Data in a unit with a different dimension or species, or a unit the parser cannot read, fails `build()`.
#
# Below, the schema stores CO₂ emissions in `MtCO2/yr`.
# The carbon cycle component expects `GtC/yr`, so the model converts the values as it reads them.
# 36,666.7 MtCO2/yr is 10 GtC/yr.

# %%
schema = VariableSchema()
schema.add_variable("Emissions|CO2", "MtCO2/yr")
schema.add_variable("Surface Air Temperature Change", "K")
schema.add_variable("Atmospheric Concentrations|CO2", "ppm")
schema.add_variable("Cumulative Carbon Uptake|Land", "Gt C")
schema.add_variable("Cumulative Emissions|CO2", "Gt C")

constant = TimeAxis.from_bounds(np.array([2000.0, 2011.0]))
emissions = Timeseries(
    np.array([36_666.7]), constant, "MtCO2/yr", InterpolationStrategy.Previous
)
temperature = Timeseries(np.array([0.0]), constant, "K", InterpolationStrategy.Previous)

model = (
    ModelBuilder()
    .with_time_axis(TimeAxis.from_values(np.arange(2000.0, 2011.0)))
    .with_schema(schema)
    .with_component(
        CarbonCycleBuilder.from_parameters(
            dict(tau=20.3, co2_pi=280.0, alpha_temperature=0.0)
        ).build()
    )
    .with_prescribed_variable("Emissions|CO2", emissions)
    .with_prescribed_variable("Surface Air Temperature Change", temperature)
    .with_initial_values(
        {
            "Cumulative Carbon Uptake|Land": 0.0,
            "Cumulative Emissions|CO2": 0.0,
            "Atmospheric Concentrations|CO2": 280.0,
        }
    )
).build()

# %% [markdown]
# The "Unit Conversions" section of `coupling()` lists the factor applied to each component input.

# %%
print(model.coupling("plain"))

# %%
model.run()
concentration = model.timeseries().get_timeseries_by_name(
    "Atmospheric Concentrations|CO2"
)
print(f"CO2 concentration in 2010: {concentration.values()[-1]:.1f} ppm")

# %% [markdown]
# Python components receive values in the schema's units, without conversion.

# %% [markdown]
# ## Summary
#
# | Method | Description |
# |--------|-------------|
# | `Unit(unit_str)` | Parse a unit string |
# | `unit.normalised()` | Get the normalised string representation |
# | `unit.is_dimensionless()` | Check if unit is dimensionless |
# | `unit.is_compatible(other)` | Check if conversion is possible |
# | `unit.conversion_factor(other)` | Get the conversion multiplier |
# | `unit.convert(value, other)` | Convert a value to another unit |
#
# In a model, declare units in the `VariableSchema` and the builder converts Rust component inputs for you.
