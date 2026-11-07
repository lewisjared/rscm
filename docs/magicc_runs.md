# MAGICC runs

`rscm.magicc` provides three runners that chain the MAGICC components on annual steps.

| Runner                 | Couples                                                | Returns        | Tested against MAGICC7 |
| ---------------------- | ------------------------------------------------------ | -------------- | ---------------------- |
| `run_magicc7`          | The full MAGICC7 chain, with its input files           | `Model`        | Yes                    |
| `run_magicc_emissions` | Chemistry, carbon cycle, GHG forcing and climate       | `MagiccResult` | No                     |
| `run_magicc`           | GHG forcing and climate from prescribed concentrations | `MagiccResult` | No                     |

- Use `run_magicc7` to reproduce MAGICC7.
- Use `run_magicc_emissions` for a small coupled chain that needs no input files
  and can start in any preindustrial year.
  All forcing outside CO2, CH4 and N2O enters as a prescribed residual.
- Use `run_magicc` for forcing and climate experiments.
  An optional partial emissions mode calculates CO2 with `CO2Budget` from prescribed land and ocean uptake.

All three runners take one annual mean emission rate per year,
labelled by the year it starts, as MAGICC7 input files are.

The [coupled MAGICC model](notebooks/magicc_coupled_model.py) tutorial wires the components in by hand.
The [module specifications](modules/README.md) describe each MAGICC7 module and its implementation status.
For an offline SSP2-4.5 concentration-driven comparison of `run_magicc` and its interpretation limits,
see the [concentration baseline](magicc_baseline.md).

## Full MAGICC7 chain

`rscm.magicc.run_magicc7` couples every MAGICC7 component RSCM implements:
chemistry, the carbon cycle, permafrost, GHG and ozone forcing, forcing aggregation
with efficacies and regional patterns, and ClimateUdeb with monthly volcanic forcing.

```python
import numpy as np
from rscm.core import TimeAxis
from rscm.magicc import full, prescribed, run_magicc7

years = np.arange(1750.0, 2101.0)
# A linear ramp stands in for a scenario such as MAGICC7's SSP2-4.5 emissions.
ramp = np.linspace(0.1, 1.0, len(years))
emissions = {
    f"{species}|{sector}": ramp
    for species in full.SECTORAL_EMISSIONS
    for sector in ("Fossil", "Biosphere")
}
emissions |= {species: np.zeros(len(years)) for species in full.halocarbon_species()}

model = run_magicc7(
    TimeAxis.from_values(years), emissions, parameters={"core_climatesensitivity": 3.0}
)
timeseries = model.timeseries()
co2 = timeseries.get_timeseries_by_name("Atmospheric Concentrations|CO2").values()
boxes = timeseries.get_four_box_timeseries_by_name("Surface Air Temperature Change").values()
temperature = boxes @ prescribed.area_fractions()
```

The time axis has annual steps and should start in 1750, the year of MAGICC7's initial state.
With the default parameters they must reach 2019,
the year to which MAGICC7 scales its prescribed aerosol and land-use forcing.
`emissions` holds one value per year for `"<species>|Fossil"` and `"<species>|Biosphere"`
of each species in `full.SECTORAL_EMISSIONS`, in the units that mapping gives.
These are the gcages units, such as `Mt CO2/yr` and `Mt SO2/yr`.
It also holds one array per halocarbon in `full.halocarbon_species()`, in kt of the gas per year.
`convention=Convention.MAGICC7`, with `Convention` from `rscm.names`,
takes the emissions in the units MAGICC7 reads instead, such as `GtC/yr` and `Mt S/yr`.
Missing, unknown or misshapen emissions raise `ValueError`.

The run returns the `Model` after its last step.
Read outputs from `model.timeseries()`.
Surface temperature and other regional variables are four-box timeseries
in northern ocean, northern land, southern ocean, southern land order.
Their area weights are always MAGICC7's default land fractions, 42% of the northern hemisphere
and 21% of the southern, as `prescribed.area_fractions()` gives.
Overriding the `ClimateUdeb` land fractions through `component_parameters` does not change them.

The keyword arguments override MAGICC7's setup:

- `parameters` takes MAGICC7 configuration names in lower case, such as `core_climatesensitivity`.
  `full.default_parameters()` gives MAGICC7 v7.5.3's defaults for every name the chain reads.
  Names the chain does not read are ignored without a warning, so a misspelt name changes nothing
  ([#162](https://github.com/lewisjared/rscm/issues/162)).
- `initial_concentrations` overrides MAGICC7's 1750 concentrations,
  which `full.default_initial_concentrations()` gives.
  A gas it does not name raises `ValueError`.
- `component_parameters` overrides the Rust parameters of single components, keyed by component name
  such as `"TerrestrialCarbon"`.
  A name outside `full.COMPONENTS` raises `ValueError`.
- `inputs` replaces the input files that `full.load_inputs` reads.

The forcing MAGICC7 prescribes before the run comes from `rscm.magicc.prescribed`
and MAGICC7's input files packaged in `rscm.magicc`.
This covers aerosols, black carbon on snow, land-use albedo, solar and volcanic forcing.
`full.prescribed_forcing` returns the four-box ERF of each of these agents before efficacies.
`full.natural_emissions` calculates the natural CH4 and N2O emissions that close MAGICC7's budget.

### Concentration-driven years

Like MAGICC7's default configuration, `run_magicc7` prescribes the concentrations of CO2,
CH4, N2O and every halocarbon up to 2015 from the packaged input files,
and calculates them from emissions afterwards.
`co2_switchfromconc2emis_year` and its `ch4`, `n2o`, `fgas` and `mhalo` counterparts move the switch.
A switch year after the first year must fall within `years`.
While concentrations are prescribed, the ocean sees CO2 interpolated within the year
and fossil CH4 oxidation scales with the inverse CH4 emissions, as in MAGICC7.
With every switch year at or before the first year, the chain runs from emissions throughout.

### MAGICC7 switches

`run_magicc7` maps these MAGICC7 switches:

- `core_co2ch4n2o_rfmethod` selects the GHG forcing formulas:
  OLBL (after Etminan et al., 2016, the default) or IPCCTAR (from the IPCC Third Assessment Report).
  With IPCCTAR it applies no rapid adjustment, because MAGICC7 ignores `core_rfrapidadjust_*` there.
- `co2_fertilization_method` and `co2_plantboxresp_fertscale` set CO2 fertilisation.
- `co2_feedbackfactor_*` and `co2_tempfeedback_switch` set the carbon cycle temperature feedbacks.
- `co2_plantboxresp_method` must be 2, the only respiration method implemented.
  Other values raise `ValueError`.
- `rf_efficacy_apply` sets how efficacies apply, including to the monthly volcanic forcing.
  `full.EFFICACY_METHODS` maps its 0, 1 and 2 to the `none`, `unit` and `ar6` efficacy methods.
  With 1, forcing is divided by each agent's internal efficacy and prescribed efficacies are ignored,
  as in MAGICC7.
- `pf_apply=0` switches permafrost off.
  MAGICC7 enables it by default.
- `ncycle_apply=1` limits NPP by nitrogen, reading fixation from MAGICC7's packaged `NCYCLE_NFIX3.IN`.
  `ncycle_apply_ndep_from_emis=1` adds deposition from NOx and NH3.
- `out_sealevel=1` adds `SeaLevelRise` and thermal expansion in `ClimateUdeb`.

In a model built by hand, add `PermafrostBuilder` and set `include_permafrost` on
`CH4Chemistry`, `CH4Oxidation`, `OceanCarbon` and `CO2Budget` to feed permafrost emissions back.
They then read `Emissions|CH4|Permafrost` and `Emissions|CO2|Permafrost`.

### Agreement with MAGICC7

`tests/regression/test_full_run.py` compares `run_magicc7` with MAGICC7 v7.5.3 runs of SSP2-4.5 emissions
in every year from 1750 to 2105.
It checks four-box surface temperature, the CO2, CH4 and N2O concentrations and total ERF
at a relative tolerance of 1e-9 and an absolute tolerance of 1e-12, set in `tests/regression/full_run.py`.
The tolerance asks for scientific parity rather than bit-for-bit agreement,
but most cases currently agree to the last bit.
The coupled cases are:

- MAGICC7's default run, which prescribes concentrations until 2015 and includes permafrost.
  This case also checks every halocarbon concentration.
- The emissions-driven run with every switch year at 1750, with and without permafrost.
- The emissions-driven run with nitrogen limitation and deposition from emissions.
  Its nitrogen pool, limitation factor and NPP agree within a relative 1e-4,
  because MAGICC7 writes them only at ASCII precision.
- 15 variants of the emissions-driven run without permafrost,
  each changing one climate, efficacy or method switch.
  `VARIANTS` in `tests/regression/full_run.py` lists them.

In the default, emissions-driven and permafrost runs, temperature, concentrations and total ERF
match MAGICC7 to the last bit in every year.
So do temperature and concentrations in the nitrogen-limited run.
Seven of the 15 variants also match to the last bit.
The others stay within 7.1e-15 K in temperature and a relative 1.4e-15 in concentrations.

The bisection on the nitrogen pool makes the nitrogen-limited run chaotic in MAGICC7 too:
a 1e-13 relative change in one year's emissions moves CO2 by 0.03 ppm.

The same file also drives single components with MAGICC7's own inputs,
so a difference points at that component alone.
`tests/regression/test_sea_level.py` checks each sea level contribution of the `out_sealevel=1` run.
Only SSP2-4.5 and these configurations have a reference run.
Other scenarios and parameter sets use the same code without a MAGICC7 comparison.

## Emissions-driven run

```python
import numpy as np
from rscm.core import TimeAxis
from rscm.magicc import Emissions, run_magicc_emissions

years = np.arange(1750.0, 1851.0)
n = len(years)
result = run_magicc_emissions(
    TimeAxis.from_values(years),
    Emissions(
        co2_fossil=np.full(n, 29000.0),   # Mt CO2/yr
        co2_biosphere=np.full(n, 3700.0), # Mt CO2/yr
        ch4=np.full(n, 300.0),            # Mt CH4/yr, anthropogenic
        ch4_fossil=np.full(n, 120.0),     # Mt CH4/yr, part of ch4
        n2o=np.full(n, 11000.0),          # kt N2O/yr
        nox=np.full(n, 130.0),            # Mt NO2/yr
        co=np.full(n, 900.0),             # Mt CO/yr
        nmvoc=np.full(n, 200.0),          # Mt VOC/yr
    ),
    initial_concentrations={"CO2": 278.0, "CH4": 722.0, "N2O": 270.0},
    other_forcing=np.zeros(len(years)),
)
co2 = result.values["Atmospheric Concentrations|CO2"]
```

Emissions are annual mean rates, one per supplied year.
The value for each year applies from that year to the next.
The run couples these components:

- `CH4Chemistry` calculates the CH4 concentration from emissions.
- `CH4Oxidation` tracks fossil CH4 and the CO2 released as it is oxidised.
- `N2OChemistry` calculates the N2O concentration from emissions.
- `TerrestrialCarbon` calculates the net land flux, including land-use emissions.
- `OceanCarbon` calculates the uptake while CO2 evolves monthly within the year.
- `CO2Budget` closes the atmospheric CO2 balance.
- `GhgForcing` and `ClimateUdeb` calculate forcing and temperature.

Natural CH4 and N2O emissions are chemistry parameters.
They default to the budget-closing values MAGICC7 derives for its standard historical concentrations.

The runner lags the feedback loops with `with_lag` and `with_lag_for`.
So, as in MAGICC7, chemistry and the carbon cycle use the temperature and CO2 at the start of each year,
and forcing uses the concentrations at its end.
The carbon-cycle feedbacks use the warming since 1900.
The CH4 lifetime feedbacks use the concentration and temperature in 1926
and the NOx, CO and NMVOC emissions in 1927.
The N2O lifetime feedbacks use the concentration in 1924 and the warming since 1980.
Runs ending before a reference year use their last year instead.
Temperature reaching these components is the area-weighted global mean.

The first year should be preindustrial.
The ocean starts in equilibrium with the initial CO2, the land pools at their preindustrial sizes
and the climate at zero anomaly.
Forcing outside CO2, CH4 and N2O, including ozone, aerosols, halocarbons and natural forcing,
is supplied through `other_forcing`.
`forcing_species` selects the calculated gases that contribute to total forcing, as in `run_magicc`.
`parameters` maps component names to their Rust parameter dictionaries:

```python
parameters = {
    "GhgForcing": {"co2_pi": 277.147, "ch4_pi": 731.406, "n2o_pi": 273.865},
    "CH4Chemistry": {"natural_emissions": 182.19},
    "ClimateUdeb": {"ecs": 3.0},
}
```

A `gtc_per_ppm` given for `CO2Budget` or `OceanCarbon` applies to both,
because the ocean evolves atmospheric CO2 within the year and must match the budget.
`tg_per_ppb`, `mixing_box_size` and `natural_emissions` are shared the same way
between `CH4Chemistry` and `CH4Oxidation`, because the oxidation repeats the CH4 budget.
Conflicting values raise `ValueError`.

### Differences from MAGICC7

No regression test compares `run_magicc_emissions` with MAGICC7.
Use `run_magicc7` where parity matters.
It differs from MAGICC7's default configuration in three ways:

- It has no permafrost.
  MAGICC7 enables permafrost by default, adding about 0.8 GtC/yr of CO2 by 2100.
- It takes forcing outside CO2, CH4 and N2O as one annual global residual.
  This cannot carry volcanic forcing within the year, efficacies or agent-specific regional patterns.
- It calculates concentrations from emissions in every year.
  MAGICC7 prescribes them until 2015.

## Concentration-driven run

```python
import numpy as np
from rscm.core import TimeAxis
from rscm.magicc import run_magicc

years = np.arange(1750.0, 1851.0)
concentrations = {
    "CO2": 278.0 * 1.01 ** np.arange(len(years)),  # ppm
    "CH4": np.full(len(years), 722.0),           # ppb
    "N2O": np.full(len(years), 270.0),           # ppb
}

result = run_magicc(
    TimeAxis.from_values(years),
    concentrations,
    other_forcing=np.zeros(len(years)),
    forcing_species=("CO2",),
    forcing_parameters={"method": "IPCCTAR", "adjust_co2": 1.0},
)
temperature = result.values["Surface Air Temperature Change"]
forcing = result.values["Effective Radiative Forcing"]
```

All three concentrations are required, including in a CO2-only experiment,
because forcing methods can include absorption overlap.
`forcing_species` selects which calculated terms contribute to total forcing.
Its default is `("CO2", "CH4", "N2O")`.

`other_forcing` must be supplied explicitly in W/m².
It covers all forcing outside the selected GHG terms.
Zeros exclude those processes from the experiment.
To include other processes, supply their combined prescribed ERF.
A reference residual can be constructed as reference total ERF minus the reference contributions
for the selected gases.
Do not supply reference total ERF as the residual, since that would count the selected gases twice.

## Partial emissions-driven run

```python
from rscm.magicc import CO2Emissions

n = len(years)
emissions_result = run_magicc(
    TimeAxis.from_values(years),
    {"CH4": concentrations["CH4"], "N2O": concentrations["N2O"]},
    other_forcing=np.zeros(n),
    co2_emissions=CO2Emissions(
        initial_concentration=278.0,
        fossil=np.full(n, 29000.0),
        biosphere=np.full(n, 3700.0),
        land_uptake=np.full(n, 2.0),
        ocean_uptake=np.full(n, 2.0),
    ),
)
co2 = emissions_result.values["Atmospheric Concentrations|CO2"]
```

The arrays contain annual mean rates.
The emissions are in Mt CO2/yr and the uptakes in GtC/yr.
Positive uptake removes carbon from the atmosphere.
Each entry applies over `[years[i], years[i + 1])`, so the last entry is never integrated.
The budget uses 2.123 GtC per ppm.
CO2 must not also be supplied as a prescribed concentration.
CH4 and N2O remain prescribed.
The budget receives the net land flux, `land_uptake` less `biosphere` in GtC/yr,
because biosphere emissions leave the land pools.

This mode exercises emissions → calculated CO2 → forcing → temperature.
It does not calculate land or ocean uptake, couple temperature back into those sinks,
or infer emissions from prescribed concentrations.

## Time and initialisation

These conventions apply to `run_magicc` and `run_magicc_emissions`.

- The time axis has at least two annual steps, and each step starts at a boundary.
- Concentrations and forcing have one value per boundary, including the final year.
  The model integrates only the intervals between these boundaries.
- The runners evaluate initial forcing using the same Rust calculation as subsequent steps.
  Each forcing output uses concentrations at its own boundary, with no one-year shift between them.
- Initial heat uptake includes the supplied forcing and the configured climate efficacy adjustment,
  even when starting above preindustrial concentrations.
- Climate starts at zero temperature anomaly with an unperturbed ocean.
  Starting with elevated concentrations applies nonzero forcing immediately.
  It does not reconstruct a historical climate state or perform a spin-up.
- Preindustrial reference concentrations are component parameters, independent of the first supplied year.
  Set `co2_pi`, `ch4_pi` and `n2o_pi` explicitly when the experiment uses different references.

For `run_magicc`, `climate_parameters` accepts ClimateUdeb parameter names such as `ecs` and `erf_2xco2`,
and `forcing_parameters` accepts GhgForcing parameter names such as `method` and `erf_2xco2`.
`run_magicc_emissions` takes both through `parameters["ClimateUdeb"]` and `parameters["GhgForcing"]`.
As in MAGICC7, `erf_2xco2` defaults to the ERF of doubling CO2 under the forcing configuration,
which is about 3.9 W/m² for OLBL.
Set it explicitly only to decouple the climate feedback from the forcing.

## Results and verification

`run_magicc` and `run_magicc_emissions` return a `MagiccResult`.
`result.years` labels the returned arrays.
`result.values` and `result.units` contain concentrations, individual and total forcing,
surface temperature, SST, heat uptake and ocean heat content.
Emissions-driven results also contain the carbon pools and fluxes, the CH4 and N2O lifetimes
and the CO2 from CH4 oxidation.
The end-of-year CO2 and the airborne fraction are `NaN` at the first year.
Global surface temperature uses the configured hemispheric land fractions.
`result.regional_temperature` retains the four boxes in northern ocean, northern land, southern ocean,
southern land order.
Points, such as concentrations and temperature, are labelled by the time they hold at.
Averages, such as emissions, fluxes and lifetimes, are labelled by the year they start.
A calculated average is `NaN` at the last year, because no step starts there.

`result.sources` identifies prescribed inputs and calculated outputs.
`result.forcing_contributors` records which forcing terms were included.
Inputs must be complete and finite.
Required concentrations must be positive.
The runners check returned outputs for nonfinite values,
and component errors raise exceptions rather than allowing a failed run to appear successful.

These two runners sum forcing in a `VariableSchema` aggregate rather than `ForcingAggregation`.
UDEB applies its CO2 regional forcing pattern to total ERF, including the prescribed residual.
Agent-specific spatial patterns and efficacies are not represented.

Run the execution tests and the reference comparisons with:

```sh
make build-dev
uv run pytest tests/test_magicc_runner.py tests/test_magicc_emissions.py tests/test_magicc_full.py \
    tests/test_magicc_prescribed.py tests/regression/test_full_run.py tests/regression/test_sea_level.py
```

The execution tests include equilibrium, forcing boundary alignment, comparison with prescribed-forcing
climate runs, carbon mass balance and concentration replay.
`tests/test_magicc_full.py` and `tests/test_magicc_prescribed.py` check the set-up of the full chain.
The regression tests compare the full chain with MAGICC7, as described above.
