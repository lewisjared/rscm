---
status: accepted
---

# The builder owns step coupling

A component reads every coupled input with `get()` and never chooses a time index.
The builder links every producer after all components are registered,
decides which edges lag by one step, and fills each input window to match.
A feedback loop breaks only where the model wiring says so,
with `builder.with_lag(variable)` or `builder.with_lag_for(variable, consumer)`.

Before this, insertion order decided coupling.
The builder linked a consumer only to producers registered before it,
so adding components in another order shifted results by a step (#162).
Components hid the missing ordering with `at_end().unwrap_or(at_start())`,
and each author had to pick between `at_start`, `at_end`, `get` and `previous` to say which value they meant.

## Considered Options

- **Components declare read timing** on each input (this step or last step).
  Rejected because new authors must understand the step's internals to write a component,
  and the same component could not lag in one model and not in another.
- **Stage hints on components** (chemistry, carbon, forcing, climate), where an edge to an earlier stage lags.
  Rejected because it is component knowledge again,
  and loops within one stage, such as land carbon and the CO2 budget, still need wiring.
- **An explicit step schedule** where reads from later stages lag.
  Rejected because a component in the wrong stage lags silently.
- **Start-of-step coupling everywhere** and **iterating loops to convergence within a step**.
  Rejected because both break MAGICC7 parity,
  which lags its feedbacks by a year but couples concentration, forcing and temperature within the year.

## Consequences

- `with_lag(variable)` lags every edge from that variable, and `with_lag_for(variable, consumer)` lags one.
  A consumer left out, such as `SeaLevelRise` reading temperature, still sees this step's value.
- A lag is part of the model, so it holds whether the variable is modelled or prescribed.
  Prescribing temperature to test ozone on its own keeps ozone's lag.
- An unbroken loop is a build error.
  It lists the `with_lag_for` calls that would break it, edges into a component's own state first,
  and never applies one itself.
- Runners carry their models' `with_lag` calls, so `run_magicc7` and `run_fair` keep MAGICC7's and FaIR's lags
  while the components stay free of timing.
- A lagged edge shifts the whole window by one step, so nothing can read a value its producer has not written.
- The window API shrinks to `get()`, `interval()` and `history(offset)` in Rust and Python.
  `history` reads only the past.
  A fixed calendar year, such as MAGICC's 1927 reference for precursor emissions,
  is read by time with `interpolate(year)`.
  It may read ahead of the step, so it suits exogenous data, and a future upstream value is not yet written.
- `model.coupling()` replaces `debug_info()` ordering.
  It shows the order, each lagged edge and each variable's [time type](0002-point-and-average-variables.md).
- A missing exogenous input is a build error unless a builder flag allows it,
  and exogenous units are converted at build time or rejected.
