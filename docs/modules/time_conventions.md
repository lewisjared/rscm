# ClimateUdeb Time Conventions

Sub-annual stepping and output conventions for the ClimateUdeb
ocean-atmosphere energy balance model.

## Annual Step

The RSCM framework calls `solve(t_current, t_next, input_state)` once per year.
`t_current` and `t_next` are decimal years (e.g. 1750.0, 1751.0). The component
receives forcing at both boundaries and returns temperature at `t_next`.

## Sub-Annual Stepping

ClimateUdeb divides each annual step into 12 monthly substeps (matching
MAGICC7's `STEPSPERYEAR = 12`).
The substep loop runs from step 1 to step 12, or from step 0 in the first year:

```rust
for step_idx in first_step..=self.parameters.steps_per_year {
    let frac = step_idx as FloatValue / steps;
    let erf = erf_start + frac * (erf_end - erf_start);
    // ... solve ocean diffusion-advection ...
}
```

### Forcing interpolation

Forcing is linearly interpolated between the annual boundaries.
Substep `k` uses the forcing at `t_current + k/12`, which is `erf_start + k/12 * (erf_end - erf_start)`.
The last substep uses forcing at exactly `t_next`.

### Temperature output

The output from `solve_impl` is the temperature after the last substep (step 12),
representing the state at `t_next` (January 1 of the next year). This is a
**point-in-time** value, not an annual mean.

## Year boundaries

### First-year boundary handling

MAGICC7 runs 13 substeps (steps 0..12) in the first year instead of the normal 12.
Step 0 uses the forcing at the start of the simulation.
RSCM does the same.

Reference: MAGICC7.f90 lines 2702-2716

### Last-year boundary handling

MAGICC7 runs only 11 substeps in the final year (steps 1..11).
RSCM always runs 12.
The full-chain regression runs still agree with MAGICC7 in every year they report, up to 2105,
so the difference does not reach their output.

### Temperature output semantics

MAGICC7 produces two temperature variables:

- `DAT_SURFACE_TEMP`: Point-in-time at the end of the year (January 1 of next
  year). This is the standard output written by `OUT_TEMPERATURE`.
- `DAT_SURFACE_ANNUALMEANTEMP`: Arithmetic mean of the 12 substep temperatures
  within the year.

RSCM outputs point-in-time values, matching `DAT_SURFACE_TEMP`.
When `thermal_expansion` is set, `ClimateUdeb` also writes `Annual Mean Surface Air Temperature Change`,
matching `DAT_SURFACE_ANNUALMEANTEMP`.
It is an average, so it is stored at the start of the year it covers, as MAGICC7 labels it.
`SeaLevelRise` reads it.

Reference: MAGICC7.f90 lines 3426-3464
