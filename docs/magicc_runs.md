# Partial MAGICC runs

`rscm.magicc.run_magicc` runs concentrations through GHG forcing, forcing
aggregation and ClimateUDEB. It also supports a partial CO2 emissions mode using
CO2Budget with prescribed land and ocean uptake. These configurations let us test
the model end to end while chemistry and carbon-cycle work continues.

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

This mode exercises emissions → calculated CO2 → forcing → temperature.
It does not calculate land or ocean uptake, couple temperature back into those
sinks, or infer emissions from prescribed concentrations.

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
`method` and `delq2xco2`. For an IPCCTAR experiment with a custom CO2-doubling
forcing, set both `delq2xco2` and `rf_2xco2` consistently.

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

Run the mandatory execution tests and the existing reference comparisons with:

```sh
make build-dev
uv run pytest tests/test_magicc_runner.py tests/regression/test_ghg_forcing.py
```

The execution tests include equilibrium, forcing boundary alignment, comparison
with prescribed-forcing climate runs, carbon mass balance and concentration
replay. The reference concentration cases use this same runner and check GHG
forcing parity. Their finite climate outputs establish successful composition;
they do not establish full MAGICC7 temperature parity.

The older fully coupled emissions parity test remains an expected failure.
UDEB currently applies its CO2 regional forcing pattern to total ERF, including
the prescribed residual; agent-specific spatial patterns are not represented.
Explicit lagged feedback dependencies, chemistry integration, carbon-cycle
initialization and complete reference input fixtures remain separate work.
