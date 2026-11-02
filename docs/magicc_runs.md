# Partial MAGICC runs

For an offline SSP245 comparison and its interpretation limits, see the
[concentration baseline](magicc_baseline.md).

`rscm.magicc.run_magicc` runs concentrations through GHG forcing, forcing
aggregation and ClimateUDEB. It also supports a partial CO2 emissions mode using
CO2Budget with prescribed land and ocean uptake.
`rscm.magicc.run_magicc_emissions` calculates CO2, CH4 and N2O from emissions
with the coupled carbon cycle and chemistry.

## Concentration-driven run

```python
import numpy as np
from rscm.magicc import run_magicc

years = np.arange(1750.0, 1851.0)
concentrations = {
    "CO2": 278.0 * 1.01 ** np.arange(len(years)),  # ppm
    "CH4": np.full(len(years), 722.0),           # ppb
    "N2O": np.full(len(years), 270.0),           # ppb
}

result = run_magicc(
    years,
    concentrations,
    other_forcing=np.zeros(len(years)),
    forcing_species=("CO2",),
    forcing_parameters={"method": "Ipcctar", "adjust_co2": 1.0},
)
temperature = result.values["Surface Temperature"]
forcing = result.values["Effective Radiative Forcing"]
```

All three concentrations are required, including in a CO2-only experiment,
because forcing methods can include absorption overlap. `forcing_species`
selects which calculated terms contribute to total forcing. Its default is
`("CO2", "CH4", "N2O")`.

`other_forcing` must be supplied explicitly in W/m². It covers all forcing
outside the selected GHG terms. Zeros exclude those processes from the
experiment. To include unfinished processes, supply their combined prescribed
ERF. A reference residual can be constructed as reference total ERF minus the
reference contributions for the selected gases. Do not supply reference total
ERF as the residual, since that would count the selected gases twice.

## Partial emissions-driven run

```python
from rscm.magicc import CO2Emissions

intervals = len(years) - 1
emissions_result = run_magicc(
    years,
    {"CH4": concentrations["CH4"], "N2O": concentrations["N2O"]},
    other_forcing=np.zeros(len(years)),
    co2_emissions=CO2Emissions(
        initial_concentration=278.0,
        fossil=np.full(intervals, 8.0),
        land_use=np.full(intervals, 1.0),
        land_uptake=np.full(intervals, 2.0),
        ocean_uptake=np.full(intervals, 2.0),
    ),
)
co2 = emissions_result.values["Atmospheric Concentration|CO2"]
```

The four emissions and uptake arrays contain annual mean rates in GtC/yr.
Positive uptake removes carbon from the atmosphere. Each entry applies over
`[years[i], years[i + 1])`; the budget uses 2.123 GtC per ppm. CO2 must not also
be supplied as a prescribed concentration. CH4 and N2O remain prescribed.
The budget receives the net land flux, `land_uptake - land_use`, because land-use
emissions leave the land pools.

This mode exercises emissions → calculated CO2 → forcing → temperature.
It does not calculate land or ocean uptake, couple temperature back into those
sinks, or infer emissions from prescribed concentrations.

## Emissions-driven run

```python
from rscm.magicc import Emissions, run_magicc_emissions

intervals = len(years) - 1
result = run_magicc_emissions(
    years,
    Emissions(
        co2_fossil=np.full(intervals, 8.0),     # GtC/yr
        co2_land_use=np.full(intervals, 1.0),   # GtC/yr
        ch4=np.full(intervals, 300.0),          # Mt CH4/yr, anthropogenic
        ch4_fossil=np.full(intervals, 120.0),   # Mt CH4/yr, part of ch4
        n2o=np.full(intervals, 7.0),            # Mt N/yr
        nox=np.full(intervals, 40.0),           # Mt N/yr
        co=np.full(intervals, 900.0),           # Mt CO/yr
        nmvoc=np.full(intervals, 200.0),        # Mt NMVOC/yr
    ),
    initial_concentrations={"CO2": 278.0, "CH4": 722.0, "N2O": 270.0},
    other_forcing=np.zeros(len(years)),
)
co2 = result.values["Atmospheric Concentration|CO2"]
```

Emissions are annual mean rates, one per interval, as in the partial mode.
The run couples these components, registered in this order:

- `CH4Chemistry` and `N2OChemistry` calculate concentrations from emissions.
  Natural emissions are parameters that default to the budget-closing values
  MAGICC7 derives for its standard historical concentrations.
- `CH4Oxidation` tracks fossil CH4 and the CO2 released as it is oxidised.
- `TerrestrialCarbon` calculates the net land flux, including land-use emissions.
- `OceanCarbon` calculates the uptake while CO2 evolves monthly within the year.
- `CO2Budget` closes the atmospheric CO2 balance.
- `GhgForcing` and `ClimateUDEB` calculate forcing and temperature.

As in MAGICC7, chemistry and the carbon cycle use the temperature at the start of
each year, and forcing uses the concentrations at its end. The carbon-cycle
feedbacks use the warming since 1900 and the CH4 and N2O feedbacks are measured
from fixed reference years (1926 and 1924), as in MAGICC7. Temperature reaching
these components is the area-weighted global mean.

The first year should be preindustrial. The ocean starts in equilibrium with the
initial CO2, the land pools at their preindustrial sizes and the climate at zero
anomaly. Forcing outside CO2, CH4 and N2O, including ozone, aerosols,
halocarbons and natural forcing, is supplied through `other_forcing`.
`parameters` maps component names to their Rust parameter dictionaries:

```python
parameters = {
    "GhgForcing": {"co2_pi": 277.147, "ch4_pi": 731.406, "n2o_pi": 273.865},
    "CH4Chemistry": {"natural_emissions": 182.19},
    "ClimateUDEB": {"ecs": 3.0},
}
```

### Agreement with MAGICC7

The regression case `03_emissions_driven` runs MAGICC7 v7.5.3 SSP245 emissions-driven
from 1750, with the RSCM-calculated forcing replaced by the reference residual.
From 1750 to 2100, RSCM stays within 0.5 ppm CO2, 3 ppb CH4, 0.03 ppb N2O and
0.02 K of MAGICC7. That case disables processes RSCM cannot represent:

- Permafrost is not implemented. MAGICC7 enables it by default, adding about
  0.8 GtC/yr of CO2 by 2100.
- Volcanic and solar forcing are disabled. MAGICC7 applies volcanic forcing
  within the year, which annual residual forcing cannot carry. With it enabled,
  concentrations still agree within 0.6 ppm and 3 ppb, but temperature differs
  by up to 0.3 K in eruption years.

MAGICC7's default configuration also prescribes concentrations until 2015,
calculating emissions-driven only afterwards. RSCM has no such mode switch.

## Time and initialization

- Years are consecutive annual boundaries, with at least two entries.
- Concentrations and forcing have one value per boundary, including the final
  year. The model integrates only the intervals between these boundaries.
- The runner evaluates initial forcing using the same Rust calculation as
  subsequent timesteps. Each forcing output uses concentrations at its own
  boundary; there is no one-year shift between them.
- Initial heat uptake includes the supplied forcing and the configured climate
  efficacy adjustment, even when starting above preindustrial concentrations.
- Climate starts at zero temperature anomaly with an unperturbed ocean.
  Starting with elevated concentrations applies nonzero forcing immediately;
  it does not reconstruct a historical climate state or perform a spin-up.
- Preindustrial reference concentrations are component parameters, independent
  of the first supplied year. Set `co2_pi`, `ch4_pi` and `n2o_pi` explicitly
  when the experiment uses different references.

`climate_parameters` accepts ClimateUDEB parameter names such as `ecs` and
`rf_2xco2`. `forcing_parameters` accepts GhgForcing parameter names such as
`method` and `delq2xco2`. As in MAGICC7, `rf_2xco2` defaults to the ERF of
doubling CO2 under the forcing configuration, which is about 3.9 W/m² for
OLBL. Set it explicitly only to decouple the climate feedback from the forcing.

## Results and verification

`result.years` labels the returned arrays. `result.values` and `result.units`
contain concentrations, individual and total forcing, surface temperature,
SST, heat uptake and ocean heat content. Global surface temperature uses the
configured hemispheric land fractions; `result.regional_temperature` retains
the four boxes in northern ocean, northern land, southern ocean, southern land
order. Prescribed interval rates are also retained at their starting boundary;
their final storage entry repeats the last rate and is never integrated.

`result.sources` identifies prescribed inputs and calculated outputs.
`result.forcing_contributors` records which forcing terms were included.
Inputs must be complete and finite. Required concentrations must be positive.
The runner checks returned outputs for nonfinite values, and component errors
raise exceptions rather than allowing a failed run to appear successful.

Run the execution tests and the reference comparisons with:

```sh
make build-dev
uv run pytest tests/test_magicc_runner.py tests/regression/test_ghg_forcing.py
```

The execution tests include equilibrium, forcing boundary alignment, comparison
with prescribed-forcing climate runs, carbon mass balance and concentration
replay. The reference concentration cases use this same runner and check GHG
forcing parity. The emissions-driven case checks concentrations, temperature,
carbon fluxes and lifetimes against MAGICC7.

UDEB currently applies its CO2 regional forcing pattern to total ERF, including
the prescribed residual; agent-specific spatial patterns are not represented.
Ozone and aerosol forcing are not yet calculated in the coupled run.
