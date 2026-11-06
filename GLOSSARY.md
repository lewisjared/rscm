# RSCM

RSCM couples reduced-complexity climate model components that advance together one timestep at a time.

## Time

**Step**:
The interval from one time bound to the next, over which every component advances once.
_Avoid_: Timestep index, year (when the step is not annual)

**Point**:
A variable whose value holds at an instant, such as a concentration, a carbon pool or temperature.
_Avoid_: Stock, instantaneous, timebound value

**Average**:
A variable whose value holds over a whole step, such as emissions, a flux or a lifetime.
_Avoid_: Period value, rate, timepoint value, annual value

## Coupling

**Exogenous variable**:
A variable whose values are supplied to the model rather than produced by a component.
_Avoid_: Prescribed input, external input, forcing (when not radiative forcing)

**Feedback loop**:
A chain of components where a variable produced later in the step is needed earlier in the same step.
_Avoid_: Cycle (outside the dependency graph), circular dependency

**Lag**:
Reading a variable's value from the previous step to break a feedback loop.
A lag belongs to the model, so it holds whether the variable is modelled or prescribed.
_Avoid_: Delay, staggering, offset
