# Naming Conventions

These rules keep names consistent across the Rust crates, the Python bindings, the docs and the tests.
Term meanings live in the [glossary](https://github.com/lewisjared/rscm/blob/main/GLOSSARY.md).
This page only says how to spell them.
When a rule and the code disagree, fix one of them in the same PR.

## API identifiers

### Rust and Python

- A variable has one `unit`, never `units`: `Input(..., unit="K")`, `Timeseries.unit`.
- The grid field and keyword is `grid`, and its enum is `GridType`: `Input(..., grid="four_box")`.
- The four-box grid is `FourBox` in types, `four_box` in snake case and four-box in prose.
- Every entry point takes the time axis as `time_axis: TimeAxis`: `from_magicc7(time_axis, ...)`.
- `solve` takes `t_current` and `t_next`, and every index argument is `time_index`: `Timeseries.at`.
- Prose says step, and substep for a component's inner steps, never timestep or time step.
- Type names say `Parameters` and public arguments say `parameters`: `ParameterSet(parameters=...)`.
- Counts and lists of parameters follow: `parameter_names`, `best_parameters`, `n_parameters`.
- `params` is fine for a local variable only.
- `with_prescribed_variable` supplies prescribed data on every grid, dispatching on the timeseries type.
- Initial values are `initial_values` and `initial_grid_values`: `build_from_components(initial_values=...)`.
- `Component` is only the author's base class, `rscm.component.Component`.
- `RustComponent` is the protocol a Rust component satisfies in Python.
- A builder adds any component with `with_component`, in Rust and in Python.
- `Builder` names an object, and `build_` a function that returns a `Model`: `ModelBuilder`, `build_model`.
- A function that returns a list of components is `components`: `rscm.fair.config.components`.
- A function that returns components as specs by name is `component_specs`: `rscm.fair.config.component_specs`.
- Each model package has `setup_model`, `build_model` and `run_model` with the same arguments.
  `setup_model` returns a `ModelSetup`, the others an unrun and a run `Model`.
- A component declares a `VariableDefinition` with a `VariableRole` of `Input`, `Output` or `State`.
- Each model package publishes its lags as `Lag` data, `LAGS` or `lags(config)`,
  and applies them to a builder with `with_lags(builder, ...)`: `rscm.magicc.model.LAGS`.
- Say prescribed and modelled, never exogenous and endogenous: `VariableSource.Prescribed`.
- Log probabilities start with `log_`: `log_pdf`, `log_prior`, `log_likelihood`.
- A standard deviation is `std_dev`, and log-space parameters are `mu` and `sigma`: `LogNormal`.
- Counts are `n_<plural noun>`: `n_parameters`, `n_iterations`, never `num_` or `total_`.
- Python keyword arguments mirror Rust field names: `harmonisation_target`, `harmonisation_reference_year`.

### Type names

- Acronyms and chemical formulae stay upper case in type and enum member names, as PEP 8 asks:
  `CO2ERF`, `ClimateUDEB`, `RSCMError`, `EfficacyMethod::AR6`.
- A name with its own mixed case keeps it at the start of a type: `FairConfig` for FaIR.
- Component types carry no model prefix, so `rscm_fair::GHGForcing` and `rscm_magicc::GHGForcing` both exist.
- A model crate namespaces its typetag names: `#[typetag::serde(name = "fair.GHGForcing")]`.
- A component's top-level parameters type is `<Component>Parameters`: `HalocarbonChemistryParameters`.
- Nested parameter structs are bare: `PrognosticFixation`, `AntarcticRegion`, `ForcingAgent`.
- The component `ModelBuilder` inserts for a schema aggregate is `AggregateComponent`.

## Enums and strings

- An enum that chooses a formulation ends in `Method`: `ForcingMethod`, `OceanCarbonMethod`.
- Its field is `method`, or `<process>_method` beside other choices: `fertilisation_method`.
- What sets a species in a run is a `Driver`, never a mode or kind: `Driver::Emissions`.
- Rust and PyO3 enum members are PascalCase: `GridType::FourBox` in Rust, `GridType.FourBox` in Python.
- Pure Python enums keep upper snake case members: `Convention.RSCM`, `ParameterStatus.SUPPORTED`.
- Every string form of an enum value is snake case: `GridType::FourBox` is `"four_box"`.
- A serde enum carries `#[serde(rename_all = "snake_case")]` unless a source model sets its strings.
  For example `VariableSource::OwnState` is `"own_state"`.
- A value named after another model's method keeps its spelling: `"OLBL"`, `"3D-GFDL"`, `"leach2021"`.
- A MAGICC7 integer switch becomes a named value: `efficacy_method="ar6"`, not `2`.
- Tables in `rscm.magicc` translate MAGICC7 integers and keys into RSCM values: `EFFICACY_METHODS`.
- A weighted aggregate is a weighted sum, `AggregateOp::WeightedSum`, never a weighted average.

## Spelling and abbreviations

- Identifiers the project owns use British spelling: `normalised`, `Optimiser`, `harmonisation_target`.
- External APIs keep their own spelling, such as serde's `Serialize`.
- `species` is singular and plural, never `specie`.
- Identifiers spell words out: `temperature_sensitivity`, not `temp_sensitivity`.
- Allowed abbreviations are `co2`, `ch4`, `n2o`, `erf`, `ecs`, `npp`, `oh`, `sst`, `pi`, `nh`, `sh`, `irf`.
- A source model's abbreviation is no reason to keep a name, because parity is judged on results.
  For example `initial_upwelling_rate`, not `w_initial`.
- Element symbols in a chemical quantity stay as symbols: `cl_atoms`, `br_cl_ratio`.
- A conversion constant between compound units may keep a compact form: `DIFFUSIVITY_CM2S_TO_M2YR`.
- Variable names spell every quantity out: `Equivalent Effective Stratospheric Chlorine`, not `EESC`.

## Parameter fields

- `temperature` is written in full: `temperature_sensitivity`, `npp_temperature_sensitivity`.
- `lifetime` is an atmospheric lifetime and `tau` a response time constant: `oh_lifetime`, `gas_exchange_tau`.
- An initial value leads with `initial_`: `initial_upwelling_rate`, `initial_ch4_lifetime`.
- A pre-industrial reference value ends in `_pi`: `co2_pi`, `sst_pi`.
- A year when a process turns on ends in `_start_year`: `temperature_feedback_start_year`.
- A year when a mode changes ends in `_switch_year`: `fgas_switch_year`.
- A fixed reference year ends in `_reference_year`: `meridional_flux_reference_year`.
- A boolean switch is `include_<process>`: `include_temperature_feedback`, `include_land_heat_capacity`.
- `radiative_efficiency` is written in full: `ch4_radiative_efficiency`.
- A share ends in `_fraction`: `npp_to_plant_fraction`.
- A conversion factor is `<to>_per_<from>`: `tg_per_ppb`, `gtc_per_ppm`.
- A component takes a grid rather than per-box fractions: `LAMCALCParameters.grid: FourBoxGrid`.
- The forcing of doubled CO2 is `erf_2xco2` in every component: `time_varying_erf_2xco2`.
- Halogen atom counts are `cl_atoms`, `br_atoms` and `br_cl_ratio` in every crate.
- One quantity has one field name across models: `deep_ocean_efficacy`, `ocean_heat_transfer`.
- `efficacy` alone means forcing efficacy, so ocean heat uptake efficacy is `deep_ocean_efficacy`.
- A field's doc comment names its MAGICC7 or FaIR key: ``(MAGICC7 `RF_EFFICACY_APPLY`)``.
- The runner maps source keys to fields in one table: `CLIMATE_PARAMETERS` in `rscm.magicc.full`.

## Variable names

[Key Concepts](../key_concepts.md#variable-names) shows how names are read.
The constants in `rscm_core::names` are the settled names, so look there before inventing one.

### Hierarchy

- A name is `Quantity|Species|Sector`, then any further breakdown: `Emissions|CO2|Fossil`.
- A child segment is always a part of its parent: `Effective Radiative Forcing|Ozone|Stratospheric`.
- The bare parent is the total, with no `Total` or `Aggregated` segment: `Emissions|CO2`.
- Take a name from gcages first, then from MAGICC7 and RCMIP: `Atmospheric Concentrations|CO2`.
- The grid is metadata, never a segment: `Surface Air Temperature Change` on any grid.

### Qualifiers

- A qualifier leads the first segment: `Net Emissions|CO2`, `Inverse Emissions|CH4`.
- Multi-word qualifiers lead the same way: `Efficacy Adjusted Effective Radiative Forcing`.
- `2xCO2` is a qualifier: `2xCO2 Effective Radiative Forcing`.
- A qualifier may name the producer of a second copy: `Ocean Carbon Atmospheric Concentrations|CO2`.
- A name never encodes a position in the step, such as end of year.
- A statistic appears only where point and average forms coexist.
  It leads as a qualifier: `Annual Mean Surface Air Temperature Change`.

### Prescribed marker

- Prescribed data for a quantity a component also produces leads with the word `Prescribed`.
  For example `Prescribed Atmospheric Concentrations|CO2`.
- Build it with `names::prescribed(...)` in Rust or `names.prescribed(...)` in Python.
- Prescribed data with no modelled counterpart keeps its plain name: `Emissions|CO2|Fossil`.

### Sectors

- The sectors are `Fossil` and `Biosphere` for every species: `Emissions|CH4|Biosphere`.
- `rscm.names.SECTORS` lists them.
- Other labels, such as `AFOLU` and `MAGICC Fossil and Industrial`, are mapped at file input and output.
- `rscm.names.convert_variable_name` does that mapping.

### Species tokens

- Species tokens follow gcages and carry no hyphens: `HFC4310mee`, `CFC11`, `HFC134a`, `SOx`, `NMVOC`.
- Chemical formulae stay as formulae: `CO2`, `NOx`, `N2O`.
- `convert_variable_name` reads older tokens such as `HFC43-10` and `CFC-11`, so old input data still loads.
- `rscm.fair.naming` maps FaIR species names to RSCM tokens.

### Names that are easy to get wrong

- The global temperature is `Surface Air Temperature Change`.
  `Sea Surface Temperature` is a different quantity.
- Ocean layers are `Ocean Temperature|Layer N`, counting the surface as layer 1.
  `names.ocean_layer_temperature(2)` builds one.
- `Ocean Heat Content` is a global integral in J.
- `Heat Uptake` and `Net Energy Imbalance` are different quantities.
- Carbon fluxes say their direction, positive for uptake: `Net Atmosphere to Land Flux|CO2`.
- `Airborne Emissions|<species>` and `Atmospheric Burden|CH4|<source>` are different quantities.
- Forcing agents follow gcages: `Aerosols|Direct Effect`, `Land-use Change`, `Aviation|Contrail`.

## Units

- The `unit` field holds one unit, written compactly: `GtC/yr`, `W/m^2`, `Mt CH4/yr`.
- `Unit` normalises a spaced input such as `W / m^2` to the compact form: `Unit.normalised()`.
- Carbon is fused to its mass unit, and other species follow it after a space: `GtC/yr`, `kt N2O/yr`.
- A species token counts the mass of that species, so `Mt SO2/yr` is a mass of SO2.
- Runner emissions use the gcages molecular basis, one per species: `Mt CO2/yr`, `Mt SO2/yr`, `Mt NO2/yr`.
- `rscm.names.EMISSIONS_UNITS` holds that basis, and halocarbons are `kt <species>/yr`.
- A component may declare its source model's basis, such as `Mt S/yr` in `rscm-magicc`.
  Its runner converts with `emissions_conversion_factor`.
- Temperature anomalies are in `K`.

## Where constants live

- `crates/rscm-core/src/names.rs` holds every name that more than one component, crate or module uses.
- Python reads the same constants from `rscm.names`, which adds the convention mapping and unit tables.
- `python/rscm/_lib/core/names.pyi` types the constants for Python.
- Import a constant rather than writing its string: `names::ERF_CO2` in Rust, `names.ERF_CO2` in Python.
- `ComponentIO` attributes need literals.
  Each component crate's `component_names.rs` test checks them against the constants.
- Build derived names with the helpers `child`, `prescribed` and `ocean_layer_temperature`.
- A constant's identifier is its segments in upper snake case, with `ERF` for the forcing.
  For example `ERF_OZONE_STRATOSPHERIC` and `NET_EMISSIONS_CO2`.
- `ERF_2XCO2` puts the qualifier last, because an identifier cannot start with a digit.

## Adding a variable name

1. Look for the quantity in `names.rs` and the glossary, and reuse an existing name if one fits.
2. Take the name from gcages if it has one, then from MAGICC7 or RCMIP, and fit it to the rules above.
3. Add the constant to the `names!` block in `names.rs`, in its group, with a one-line doc comment.
4. Add the same constant and docstring to `names.pyi`.
   `make validate-pyi` and `tests/test_names_drift.py` check the stub.
5. Declare it in `ComponentIO` as a literal, and add the constant to the crate's `component_names.rs` test.
6. Import the constant in every other Rust module, Python module and test.
7. If another convention spells it differently, add the mapping to the tables in `python/rscm/names.py`.
8. Renaming a variable is a breaking change, so it needs a `breaking` changelog fragment.

## Not settled yet

- What a `run_` function returns differs between runners until the build and run rework.
- `parameters=` holds different things across the MAGICC runners, and stays that way for now.
