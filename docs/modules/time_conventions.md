# ClimateUDEB Time Conventions

Sub-annual time-stepping and output conventions for the ClimateUDEB
ocean-atmosphere energy balance model.

## Annual Timestep

The RSCM framework calls `solve(t_current, t_next, input_state)` once per year.
`t_current` and `t_next` are decimal years (e.g. 1750.0, 1751.0). The component
receives forcing at both boundaries and returns temperature at `t_next`.

## Sub-Annual Stepping

ClimateUDEB divides each annual timestep into 12 monthly substeps (matching
MAGICC7's `STEPSPERYEAR = 12`). The substep loop runs from step 1 to step 12:

```rust
for step_idx in 1..=self.parameters.steps_per_year {
    let frac = step_idx as FloatValue / steps;
    let erf = erf_start + frac * (erf_end - erf_start);
    // ... solve ocean diffusion-advection ...
}
```

### Forcing interpolation

Forcing is linearly interpolated between the annual boundaries at each substep:

| Step | Fraction | Forcing time | Value |
|------|----------|-------------|-------|
| 1 | 1/12 | t_current + 1/12 | erf_start + 1/12 * (erf_end - erf_start) |
| 2 | 2/12 | t_current + 2/12 | erf_start + 2/12 * (erf_end - erf_start) |
| ... | ... | ... | ... |
| 12 | 12/12 | t_next | erf_end |

The last substep uses forcing at exactly `t_next`.

### Temperature output

The output from `solve_impl` is the temperature after the last substep (step 12),
representing the state at `t_next` (January 1 of the next year). This is a
**point-in-time** value, not an annual mean.

## Known Differences from MAGICC7

### First-year boundary handling

MAGICC7 uses an overlapping year boundary scheme where the first year runs 13
substeps (steps 0..12) instead of the normal 12 (steps 1..12). Step 0 covers
the initial boundary at the start of the simulation. RSCM always runs exactly
12 substeps. This affects the initial transient ("shock" phase) but has
negligible impact on converged temperatures.

Reference: MAGICC7.f90 lines 2702-2716

### Last-year boundary handling

MAGICC7 runs only 11 substeps in the final year (steps 1..11). RSCM always
runs 12. This has minimal impact since the final year is typically well past
the transient phase.

### Temperature output semantics

MAGICC7 produces two temperature variables:

- `DAT_SURFACE_TEMP`: Point-in-time at the end of the year (January 1 of next
  year). This is the standard output written by `OUT_TEMPERATURE`.
- `DAT_SURFACE_ANNUALMEANTEMP`: Arithmetic mean of the 12 substep temperatures
  within the year.

RSCM outputs point-in-time values, matching `DAT_SURFACE_TEMP`. Annual means
are not currently computed.

Reference: MAGICC7.f90 lines 3426-3464
