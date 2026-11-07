# RSCM

RSCM provides a framework for building reduced-complexity climate models.

## Time

**Step**:
The interval from one time bound to the next, over which every component advances once.
_Avoid_: Timestep, time step, timestep index, year (when the step is not annual)

**Point**:
A variable whose value holds at an instant, such as a concentration, a carbon pool or temperature.
_Avoid_: Stock, instantaneous, timebound value

**Average**:
A variable whose value holds over a whole step, such as emissions, a flux or a lifetime.
_Avoid_: Period value, rate, timepoint value, annual value

**Time axis**:
The ordered time bounds that a model or timeseries is defined on, with one more bound than it has steps.
_Avoid_: Temporal grid, time grid, timebounds, years (for the axis itself)

**Time bound**:
The time at one edge of a step, so a step runs from its start bound to its end bound.
_Avoid_: Timebound, boundary, time point (for the bound itself)

**Substep**:
One of the equal parts a component divides a step into to integrate within it,
such as MAGICC's monthly substeps.
Other components never see a substep.
_Avoid_: Sub-step, inner step, timestep (for a substep)

## Coupling

**Prescribed variable**:
A variable whose values are supplied to the model rather than produced by a component.
_Avoid_: Exogenous variable, external input, external forcing, external data,
forcing (when not radiative forcing), prescribed (for a fixed parameter value)

**Modelled variable**:
A variable that a component in the model produces.
_Avoid_: Endogenous variable, computed variable, internal variable

**Producer**:
The one component or schema aggregate that writes a variable.
Every component that reads it is one of its consumers.
_Avoid_: Writer, provider, source component, upstream component (for any producer)

**Feedback loop**:
A chain of components where a variable produced later in the step is needed earlier in the same step.
_Avoid_: Cycle (outside the dependency graph), circular dependency

**Lag**:
Reading a variable's value from the previous step to break a feedback loop.
A lag belongs to the model, so it holds whether the variable is modelled or prescribed.
_Avoid_: Delay, staggering, staggered scheme, offset

**Variable source**:
Where one component's input is read from, as the builder wired it: prescribed data, an upstream producer,
a lagged edge or the component's own state.
_Avoid_: VarSource, UpstreamOutput, variable type, exogenous (for prescribed), origin

**Coupling**:
The wiring the builder chooses for a model: the producer of each variable, the order components run in,
and which edges lag.
_Avoid_: Debug info, dependency ordering, staging, schedule

**Window**:
A component's view of one input's timeseries inside solve,
positioned by the builder so that get() returns the wired value.
_Avoid_: Accessor, input slice, time index (for what a component chooses)

## Variables

**Variable**:
A named quantity with one unit, one grid and one timeseries type that components read or write.
_Avoid_: Field, series, quantity (in API names)

**Variable name**:
The pipe-separated name of a variable, read from general to specific, such as Emissions|CO2|Fossil.
Each segment after the first names a part of the one before it.
_Avoid_: Key, label, path

**Timeseries type**:
Whether a variable is a point or an average.
This determines how a timeseries is operated on.
_Avoid_: Time type, variable kind, storage type

**Variable definition**:
A component's declaration of one variable it uses: its name, unit, grid, timeseries type and role as input,
output or state.
_Avoid_: Requirement, requirement definition, variable metadata, var info

**State variable**:
A variable that a component both reads and writes, carrying its own value from one step into the next.
It must be a point.
_Avoid_: Own output, persistent output, stock

**Initial value**:
The value a variable holds at the first time bound, which every state variable needs before the model runs.
_Avoid_: Initial condition, initial state, init, spin-up value

**Schema**:
The declaration of a model's variables and of the aggregates computed from them.
_Avoid_: Registry, variable catalogue, manifest

**Aggregate**:
A variable the schema computes from its contributors by a sum, a mean or a weighted sum,
with no component of its own.
_Avoid_: Aggregator, derived variable, virtual component, total (as a kind of variable),
weighted average (for a weighted sum)

**Qualifier**:
A word that changes what a quantity means rather than naming a part of it, such as Prescribed,
Inverse or Efficacy Adjusted.
It leads the first segment of a variable name.
_Avoid_: Suffix, modifier, tag

**Sector**:
The source of emissions that a variable name breaks a species down by: Fossil or Biosphere.
_Avoid_: AFOLU, Land Use, MAGICC Fossil and Industrial, Energy and Industrial Processes, source (for a sector)

**Unit**:
The physical unit of a variable's values, such as Mt CH4/yr or W/m^2.
A variable has exactly one.
_Avoid_: Units (for one variable's unit)

**Mass basis**:
The chemical form whose mass an emissions unit counts, such as SO2 in Mt SO2/yr.
Each species has one mass basis, the whole molecule.
_Avoid_: Elemental basis (for the canonical unit), species unit

**Naming convention**:
A consistent set of variable names used by one source, such as gcages, RCMIP or AR6 WG3,
that RSCM names map to and from.
_Avoid_: Vocabulary, dialect, scheme

## Grids

**Grid**:
The spatial layout of a variable's values: scalar for one global value, hemispheric, or four-box.
A variable's grid is not part of its name.
_Avoid_: Grid type (in prose), spatial resolution, resolution

**Region**:
One area of a grid, such as northern land, with a fixed index in the grid's order.
_Avoid_: Cell, box (outside the four-box grid), zone

**Four-box grid**:
The grid of MAGICC's four regions, in order northern ocean, northern land, southern ocean and southern land.
_Avoid_: 4-box, four box, fourbox, FourBox (in prose)

**Grid weights**:
The area fraction of each region of a grid, used to aggregate its values to a coarser grid.
_Avoid_: Area weights, box weights, land fractions (as weights)

## Components and models

**Component**:
A piece of physics that declares the variables it reads and writes and advances them over one step.
_Avoid_: Module (for an RSCM component), block, process, custom component,
Python component (for the author's class)

**Model**:
A set of components coupled on one time axis, with their prescribed data and initial values,
that runs step by step.
Changing any of the components results in a different model.
_Avoid_: Simulation, pipeline, run (for the object)

**Model builder**:
The object that collects components, prescribed data, initial values and lags,
decides the coupling and builds a model.
_Avoid_: Factory, assembler

**Parameter**:
A number or option that configures a component,
fixed when the component is built and constant through the run.
_Avoid_: Param, setting, constant, config (for one component), prescribed (for a fixed value)

**Parity**:
Agreement between an RSCM model and the reference model it ports,
on the same inputs and within a stated tolerance.
_Avoid_: Equivalence, match, validation (for this check)

**Method**:
One of several formulations a component offers for the same physics,
such as a forcing method or an ocean carbon method.
_Avoid_: Kind, mode, model (for a formulation), scheme, profile

## Climate

**Forcing agent**:
A single cause of radiative forcing, such as CO2, sulfate aerosol or solar variability,
that the forcing aggregation combines.
_Avoid_: Forcer, forcing species, forcing term

**Effective radiative forcing**:
The change in the top of atmosphere energy balance caused by a forcing agent,
once rapid adjustments have happened, in W/m^2.
_Avoid_: ERF (in variable names), RF (for effective forcing), radiative forcing (when effective is meant),
delq, q

**Efficacy**:
The warming a forcing agent causes per unit of forcing, relative to the warming from the same forcing by CO2.
_Avoid_: Efficiency, scaling factor, efficacy (for ocean heat uptake efficacy)

**Equilibrium climate sensitivity**:
The global mean warming at equilibrium after atmospheric CO2 doubles.
_Avoid_: Sensitivity (alone), climate sensitivity (when equilibrium is meant), core_climatesensitivity

**Species**:
A chemical compound or aerosol the model tracks, such as CH4, HFC134a or SOx, with its own emissions,
concentration or forcing.
_Avoid_: Specie, gas (for aerosols), agent (for a compound),
hyphenated names such as CFC-11 (in variable names)

**Driver**:
What sets a species' concentration in a run: its emissions, or a prescribed concentration.
A run is emissions driven or concentration driven for each species.
_Avoid_: Mode, gas mode, driver kind

**Inverse emissions**:
The emissions a concentration-driven species implies, calculated back from its prescribed concentration.
_Avoid_: Diagnosed emissions, back-calculated emissions, emissions (for the inverse)

**2xCO2 effective radiative forcing**:
The effective radiative forcing from doubling atmospheric CO2 from its pre-industrial concentration.
_Avoid_: F2x, q2x, delq2xco2, CO2 doubling forcing

**Surface air temperature change**:
The global mean near-surface air temperature relative to a reference period, which the assessment calls GSAT.
_Avoid_: Surface temperature (for this quantity), GSAT (in variable names), GMST (for an air temperature)

## Calibration

**Calibration**:
Fitting a model's parameters to observations,
either by sampling their posterior distribution or by finding a point estimate.
_Avoid_: Tuning, fitting (for the subsystem), optimisation (for sampling)

**Target**:
The observations, grouped by variable, that a calibration compares model output against.
_Avoid_: Observations (for the whole set), dataset, reference data

**Walker**:
One member of an ensemble sampler's population, holding a parameter set that moves at each iteration.
_Avoid_: Particle, member, chain (for one walker)

**Chain**:
The record of every walker's parameter set and log probability at every kept iteration of a sampling run.
_Avoid_: Trace, samples (for the stored object), history
