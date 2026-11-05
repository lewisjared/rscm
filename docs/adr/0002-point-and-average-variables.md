---
status: accepted
---

# Point and average variables

Every variable is either a point or an average, the terms of openscm's `POINT_TIMESERIES` and `AVERAGE_TIMESERIES`.
A point value holds at an instant and is stored at that time.
An average value holds over a step and is stored at the step's start bound.
This applies whoever produced the value, so prescribing a variable instead of modelling it never shifts results.

Before this, `get()` read index N for exogenous data and N+1 for an upstream output.
That lined up averages such as emissions, because input files label them by the step's start
and components wrote them at the step's end.
It misaligned points: a prescribed concentration gave the start of the step and a modelled one gave the end.
The mismatch leaked into the regression tests, which shift MAGICC7 lifetimes by a step,
and into the runners, which disagreed on whether emissions have one value per year or one per interval.

## Considered Options

- **Store everything at the step's end and convert at the model boundary.**
  Rejected because the conversion needs the same point or average metadata,
  and model state would carry labels that match nothing outside the model.
- **Keep keying on the source**, exogenous or upstream, and document it.
  Rejected because points stay misaligned.

## Consequences

- The producer declares the type through `timeseries_type`: a component's output definition,
  or the declaration of an exogenous variable.
  Consumers inherit it, and an input that claims a different type is a build error.
- The type defaults from the unit, as openscm's `guess_parameter_type` does.
  A unit per time, such as `GtC / yr`, means average, and anything else means point,
  including `W / m^2` and `yr`.
  Lifetimes and annual means override it to average.
- The type sets the default interpolation, `Previous` for averages and `Linear` for points.
  `with_interpolation_strategy` can still override it.
- State variables must be points.
  The builder enforces this by default, and a builder flag turns the check off.
- `interval()` is an error on an average, because an average has one value per step.
- Runners take one value per year label for averages, as input files, IAMC data and `run_magicc7` do.
  `run_magicc` and `run_magicc_emissions`, which take one value per interval, change to match.
