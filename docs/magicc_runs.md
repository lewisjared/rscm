# MAGICC runs

`rscm.magicc` builds two MAGICC models on annual steps.
MAGICC7 runs both from one model with switches.
RSCM keeps them apart, because the concentration-driven model lacks the inverse emissions
of every gas but CH4.

| Model                              | Concentrations                                     | Parity tested |
| ---------------------------------- | -------------------------------------------------- | ------------- |
| `rscm.magicc.emissions_driven`     | From emissions, optionally prescribed until a year | Yes           |
| `rscm.magicc.concentration_driven` | Prescribed for the whole run                       | Yes           |

As in FaIR, building a model is separate from running it.
Each module has `setup_model(config, inputs, time_axis)`, which returns a `rscm.spec.ModelSetup`.
`setup.build()` returns an unrun `Model` and `setup.run()` a run one.
`rscm.spec.run_all` runs many setups in parallel, from either model or from FaIR.
`rscm.magicc.full.from_magicc7` translates MAGICC7's configuration, input files and emissions
into the `config` and `inputs` both models take.

The [coupled MAGICC model](notebooks/magicc_coupled_model.py) tutorial wires the components in by hand.
The [module specifications](modules/README.md) describe each MAGICC7 module and its implementation status.

## Running a MAGICC7 configuration

```python
import numpy as np
from rscm.core import TimeAxis
from rscm.magicc import emissions_driven, full, prescribed

years = np.arange(1750.0, 2101.0)
time_axis = TimeAxis.from_values(years)
# A linear ramp stands in for a scenario such as MAGICC7's SSP2-4.5 emissions.
ramp = np.linspace(0.1, 1.0, len(years))
emissions = {
    f"{species}|{sector}": ramp
    for species in full.SECTORAL_EMISSIONS
    for sector in ("Fossil", "Biosphere")
}
emissions |= {species: np.zeros(len(years)) for species in full.halocarbon_species()}

config, inputs = full.from_magicc7(
    time_axis, emissions, parameters={"core_climatesensitivity": 3.0}
)
model = emissions_driven.setup_model(config, inputs, time_axis).run()
timeseries = model.timeseries()
co2 = timeseries.get_timeseries_by_name("Atmospheric Concentrations|CO2").values()
boxes = timeseries.get_four_box_timeseries_by_name("Surface Air Temperature Change").values()
temperature = boxes @ prescribed.area_fractions()
```

The time axis has annual steps and should start in 1750, the year of MAGICC7's initial state.
With the default parameters it must reach 2019,
the year to which MAGICC7 scales its prescribed aerosol and land-use forcing.
`emissions` holds one value per year for `"<species>|Fossil"` and `"<species>|Biosphere"`
of each species in `full.SECTORAL_EMISSIONS`, in the units that mapping gives.
These are the gcages units, such as `Mt CO2/yr` and `Mt SO2/yr`.
It also holds one array per halocarbon in `full.halocarbon_species()`, in kt of the gas per year.
`convention=Convention.MAGICC7`, with `Convention` from `rscm.names`,
takes the emissions in the units MAGICC7 reads instead, such as `GtC/yr` and `Mt S/yr`.
Missing, unknown or misshapen emissions raise `ValueError`.
Emissions are annual mean rates, labelled by the year they start, as MAGICC7 input files are.

`run()` returns the `Model` after its last step.
Read outputs from `model.timeseries()`.
Surface air temperature change and other regional variables are four-box timeseries
in northern ocean, northern land, southern ocean, southern land order.
Their grid weights are always MAGICC7's default land fractions, 42% of the northern hemisphere
and 21% of the southern, as `prescribed.area_fractions()` gives.
Overriding the `ClimateUDEB` land fractions in the configuration does not change them.

The keyword arguments of `from_magicc7` override MAGICC7's setup:

- `parameters` takes MAGICC7 configuration names in lower case, such as `core_climatesensitivity`.
  `full.default_parameters()` gives MAGICC7 v7.5.3's defaults for every name the translation reads.
  Names it does not read are ignored without a warning, so a misspelt name changes nothing
  ([#162](https://github.com/lewisjared/rscm/issues/162)).
- `initial_concentrations` overrides MAGICC7's 1750 concentrations,
  which `full.default_initial_concentrations()` gives.
  A gas it does not name raises `ValueError`.
- `input_files` replaces the input files that `full.load_inputs` reads.
  Load them once and pass them to every translation of an ensemble.
- `concentration_driven=True` prescribes every concentration for the whole run,
  for the concentration-driven model.

The forcing MAGICC7 prescribes before the run comes from `rscm.magicc.prescribed`
and MAGICC7's input files packaged in `rscm.magicc`.
This covers aerosols, black carbon on snow, land-use albedo, solar and volcanic forcing.
`full.prescribed_forcing` returns the four-box ERF of each of these agents before efficacies.
`full.natural_emissions` calculates the natural CH4 and N2O emissions that close MAGICC7's budget.

## Configuration and inputs

The configuration is a plain dict that can be edited before building:

- `components` maps each component name, such as `"TerrestrialCarbon"`, to its Rust parameters.
  A model rejects a configuration that lacks one of its `COMPONENTS`
  or names a component outside `COMPONENTS` and `OPTIONAL_COMPONENTS`.
  `Permafrost` and `SeaLevelRise` are optional.
- `initial_values` holds the value at the first time bound of each state variable.
  The build calculates those of `ClimateUDEB` and `ForcingAggregation` from the first year's forcing.
- `initial_forcing`, optional, holds the forcing of `ForcingAggregation` agents at the first time bound.
  As in MAGICC7, `from_magicc7` starts the F-gas forcing from the second year's value,
  which a two-year run of the translated configuration calculates.

```python
config["components"]["ClimateUDEB"]["ecs"] = 2.5
model = emissions_driven.setup_model(config, inputs, time_axis).run()
```

`inputs` maps each prescribed variable to its values and unit, with one value per time bound.
Four-box values have four columns and hemispheric values two.
An input off the time axis raises `ValueError`.

The models carry MAGICC7's feedback lags, `rscm.magicc.model.LAGS`.
So chemistry and the carbon cycle use the temperature and CO2 at the start of each year,
and forcing uses the concentrations at its end.

## Inspecting a model before building

`setup_model` returns the model as data: its components with the parameters they are built from,
its lags, initial values and prescribed data.
Printing it summarises them and names any input that nothing supplies.

```python
from rscm.magicc import concentration_driven
from rscm.spec import diff

setup = emissions_driven.setup_model(config, inputs, time_axis)
print(setup)
```

```text
MAGICC emissions-driven
  time axis  1750 to 2101, 351 steps
  components (12, in the order added)
    Permafrost           18 parameters
    CH4Chemistry         6 parameters
    ...
    ClimateUDEB          26 parameters
  lags (4)
    Surface Air Temperature Change -> every consumer
    Atmospheric Concentrations|CO2 -> TerrestrialCarbon
    Atmospheric Concentrations|CO2 -> OceanCarbon
    Equilibrium Climate Sensitivity -> ForcingAggregation
  prescribed data (113)
    Effective Radiative Forcing            13
    Emissions                              56
    Prescribed Atmospheric Concentrations  44
  missing inputs: none
  initial values: 57 scalar, 1 gridded
```

`setup.spec.components["CO2Budget"].parameters` gives one component's parameters,
and `setup.prescribed_inputs()` lists every input no component produces,
with its unit, its consumers and whether the setup supplies it.
`diff` shows what changes between two setups.
Here the concentration-driven model differs only in its switch years:

```python
print(diff(setup, concentration_driven.setup_model(config, inputs, time_axis)))
```

```text
MAGICC emissions-driven -> MAGICC concentration-driven
  ~ CH4Chemistry.concentration_switch_year: 2015.0 -> inf
  ~ N2OChemistry.concentration_switch_year: 2015.0 -> inf
  ~ HalocarbonChemistry.fgas_switch_year: 2015.0 -> inf
  ~ HalocarbonChemistry.montreal_gases_switch_year: 2015.0 -> inf
  ~ OceanCarbon.concentration_switch_year: 2015.0 -> inf
  ~ CO2Budget.concentration_switch_year: 2015.0 -> inf
```

`setup.build()` returns the unrun model and `setup.run()` the run one.
To change the model, edit the configuration and call `setup_model` again,
because the initial values follow from the parameters and the first year's forcing.

## Emissions-driven model

`rscm.magicc.emissions_driven` couples chemistry, the carbon cycle, permafrost, GHG and ozone forcing,
forcing aggregation with efficacies and regional patterns, and ClimateUDEB with monthly volcanic forcing.
Its module docstring lists the role of each component.

Like MAGICC7's default configuration, `from_magicc7` prescribes the concentrations of CO2,
CH4, N2O and every halocarbon up to 2015 from the packaged input files,
and the model calculates them from emissions afterwards.
`co2_switchfromconc2emis_year` and its `ch4`, `n2o`, `fgas` and `mhalo` counterparts move the switch.
A switch year after the first year must fall within the time axis.
While concentrations are prescribed, the ocean sees CO2 interpolated within the year
and fossil CH4 oxidation scales with the inverse CH4 emissions, as in MAGICC7.
With every switch year at or before the first year, the chain runs from emissions throughout.

## Concentration-driven model

`rscm.magicc.concentration_driven` has the components of the emissions-driven model.
`setup_model` sets every concentration switch year to infinity, whatever the configuration holds,
so CO2, CH4, N2O and every halocarbon follow their prescribed concentrations for the whole run.
The carbon cycle still runs, so land and ocean fluxes and pools follow the prescribed CO2.

```python
from rscm.magicc import concentration_driven

config, inputs = full.from_magicc7(time_axis, emissions, concentration_driven=True)
model = concentration_driven.setup_model(config, inputs, time_axis).run()
```

`CH4Chemistry` writes the anthropogenic CH4 emissions that reproduce the prescribed concentration
as `Inverse Emissions|CH4`, net of any permafrost emissions.
The model calculates no other inverse emissions, so `Net Emissions|CO2` of `CO2Budget`
is the net source the prescribed CO2 implies rather than fossil emissions.
The components still read their emissions inputs:
fossil CO2 spreads short-lived F-gases between the hemispheres,
and NOx, CO and NMVOC drive tropospheric ozone.

## MAGICC7 switches

`from_magicc7` maps these MAGICC7 switches:

- `core_co2ch4n2o_rfmethod` selects the GHG forcing formulas:
  OLBL (after Etminan et al., 2016, the default) or IPCCTAR (from the IPCC Third Assessment Report).
  With IPCCTAR it applies no rapid adjustment, because MAGICC7 ignores `core_rfrapidadjust_*` there.
- `co2_fertilization_method` and `co2_plantboxresp_fertscale` set CO2 fertilisation.
- `co2_feedbackfactor_*` and `co2_tempfeedback_switch` set the carbon cycle temperature feedbacks.
- `co2_plantboxresp_method` must be 2, the only respiration method implemented.
  Other values raise `ValueError`.
- `rf_efficacy_apply` sets how efficacies apply, including to the monthly volcanic forcing.
  `full.EFFICACY_METHODS` maps its 0, 1 and 2 to the `none`, `unit` and `ar6` efficacy methods.
  With 1, forcing is divided by each agent's internal efficacy and efficacies are ignored,
  as in MAGICC7.
- `pf_apply=0` leaves `Permafrost` out of the configuration.
  MAGICC7 enables it by default.
- `ncycle_apply=1` limits NPP by nitrogen, reading fixation from MAGICC7's packaged `NCYCLE_NFIX3.IN`.
  `ncycle_apply_ndep_from_emis=1` adds deposition from NOx and NH3.
- `out_sealevel=1` adds `SeaLevelRise` and thermal expansion in `ClimateUDEB`.

With `Permafrost`, set `include_permafrost` on `CH4Chemistry`, `CH4Oxidation`, `OceanCarbon`
and `CO2Budget` to feed permafrost emissions back, as `from_magicc7` does.
They then read `Emissions|CH4|Permafrost` and `Emissions|CO2|Permafrost`.

## Agreement with MAGICC7

`tests/regression/test_full_run.py` compares both models with MAGICC7 v7.5.3 runs of SSP2-4.5
in every year from 1750 to 2105.
It checks four-box surface air temperature change, the CO2, CH4 and N2O concentrations and total ERF
at a relative tolerance of 1e-9 and an absolute tolerance of 1e-12, set in `tests/regression/full_run.py`.
The tolerance asks for scientific parity rather than bit-for-bit agreement,
but most cases currently agree to the last bit.
The cases are:

- MAGICC7's default run, which prescribes concentrations until 2015 and includes permafrost.
  This case also checks every halocarbon concentration.
- The emissions-driven run with every switch year at 1750, with and without permafrost.
- The emissions-driven run with nitrogen limitation and deposition from emissions.
  Its nitrogen pool, limitation factor and NPP agree within a relative 1e-4,
  because MAGICC7 writes them only at ASCII precision.
- 15 variants of the emissions-driven run without permafrost,
  each changing one climate, efficacy or method switch.
  `VARIANTS` in `tests/regression/full_run.py` lists them.
- The concentration-driven run, with every switch year after the run and permafrost.
  This case also checks every halocarbon concentration, the land and ocean CO2 fluxes
  and the inverse CH4 emissions.

In the default, emissions-driven, permafrost and concentration-driven runs,
temperature, concentrations and total ERF match MAGICC7 to the last bit in every year.
So do temperature and concentrations in the nitrogen-limited run.
Seven of the 15 variants also match to the last bit.
The others stay within 7.1e-15 K in temperature and a relative 1.4e-15 in concentrations.
The inverse CH4 emissions agree within 0.2% in the 1750s and to about a relative 1e-6 after.

The bisection on the nitrogen pool makes the nitrogen-limited run chaotic in MAGICC7 too:
a 1e-13 relative change in one year's emissions moves CO2 by 0.03 ppm.

The same file also drives single components with MAGICC7's own inputs,
so a difference points at that component alone.
`tests/regression/test_sea_level.py` checks each sea level contribution of the `out_sealevel=1` run.
Only SSP2-4.5 and these configurations have a reference run.
Other scenarios and parameter sets use the same code without a MAGICC7 comparison.

Run the tests with:

```sh
make build-dev
uv run pytest tests/test_magicc_model.py tests/test_magicc_full.py tests/test_magicc_prescribed.py \
    tests/regression/test_full_run.py tests/regression/test_sea_level.py
```
