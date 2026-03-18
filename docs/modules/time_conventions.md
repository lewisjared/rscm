# Time Conventions: MAGICC7 vs RSCM

This document describes the sub-annual time-stepping conventions used by
MAGICC7 and RSCM, and the implications for numerical agreement.

## MAGICC7 Sub-Annual Time Structure

MAGICC7 uses monthly sub-stepping (STEPSPERYEAR = 12) with an **overlapping**
year boundary scheme.

### Time axis

The sub-annual time axis is constructed as:

```
alltimes_d(idx) = year + (step - 1) / 12
```

For year 1750: `1750.0, 1750.083, 1750.167, ..., 1750.917`
For year 1751: `1751.0, 1751.083, 1751.167, ..., 1751.917`

Note: These are the times at which forcing is queried. For a normal year
running steps 1..12, forcing is queried at the time indices corresponding to
`year+0/12` through `year+11/12` (steps 1..12 of the year's allocation in
`alltimes_d`). However, step 12 of year N actually uses the index that falls
at `(year+1)+0/12 = year+1.0` due to the overlapping scheme (see below).

### Substep loop

The loop for each year runs `CURRENT_STEP = STARTSTEP` to `ENDSTEP`:

- **First year (year_idx=1):** `STARTSTEP=0`, `ENDSTEP=12` (13 substeps)
- **Normal years:** `STARTSTEP=1`, `ENDSTEP=12` (12 substeps)
- **Last year (year_idx=NYEARS):** `STARTSTEP=1`, `ENDSTEP=11` (11 substeps)

Reference: MAGICC7.f90 lines 2702-2716

The **overlapping scheme** means:

- Year 1 runs steps 0..12 → covers times 1750.0 to 1751.0 (13 substeps)
- Year 2 runs steps 1..12 → covers times 1751.083 to 1752.0 (12 substeps)
- Year 3 runs steps 1..12 → covers times 1752.083 to 1753.0 (12 substeps)

Step 12 of year N occurs at time `(N+1) + 0/12 = N+1.0`, which is also the
location of step 0 for year N+1 (if it existed). The "January of next year"
step is **included** in the current year's loop, not the next year's.

### Forcing interpolation

Forcing is interpolated to each substep time:

```fortran
q = datastore_get_box_with_interpolation(dat_total_effrf, alltimes_d(current_time_idx))
```

Reference: MAGICC7.f90 line 2724

### Temperature outputs

MAGICC7 produces two temperature outputs:

1. **`DAT_SURFACE_TEMP`** (point-in-time):
   - Stored at `NEXT_YEAR_IDX = CURRENT_YEAR_IDX + 1`
   - Value is `CURRENT_TIME_TEMPERATURE` after the **last substep** (step 12)
   - Represents the temperature at January 1 of the next year
   - Reference: MAGICC7.f90 line 3462

2. **`DAT_SURFACE_ANNUALMEANTEMP`** (annual mean):
   - Stored at `CURRENT_YEAR_IDX`
   - Arithmetic mean of `THISYEAR_TEMPERATURE_STEPS(1:12, :)`
   - Note: step 0 (when it exists) is **excluded** from the average
   - Reference: MAGICC7.f90 lines 3426-3428

The standard output (`OUT_TEMPERATURE`) writes `DAT_SURFACE_TEMP` (point-in-time).

## RSCM Sub-Annual Time Structure

RSCM runs a fixed 12 substeps per year, with forcing times aligned to
MAGICC7's normal-year convention (steps 1..12).

### Substep loop

```rust
for step_idx in 1..=self.parameters.steps_per_year {
    // 12 substeps: step 1, 2, ..., 12
}
```

### Forcing interpolation

```rust
let frac = step_idx as FloatValue / steps;
let erf = erf_start + frac * (erf_end - erf_start);
```

- Step 1: `erf_start + 1/12 * (erf_end - erf_start)` (forcing at t_current + 1/12)
- Step 12: `erf_end` (forcing at t_next)

### Temperature output

The output from `solve_impl` is the temperature after the last substep (step 12).
This represents the temperature at `t_next`, matching MAGICC7's `DAT_SURFACE_TEMP`
(point-in-time at January 1 of the next year).

## Resolved: Substep Forcing Timing

The original RSCM implementation used steps 0..11 with forcing at
`year+0/12` through `year+11/12`, stopping 1/12 year short of the
year boundary. This was resolved by shifting to steps 1..12 with
forcing at `year+1/12` through `year+12/12`, matching MAGICC7's
normal-year convention.

## Remaining Differences

| Aspect | MAGICC7 | RSCM | Impact |
|--------|---------|------|--------|
| Substeps per year (normal) | 12 (steps 1..12) | 12 (steps 1..12) | Matched |
| Last substep forcing time | `year+12/12 = year+1.0` | `year+12/12 = year+1.0` | Matched |
| Output temperature time | January 1 of next year | t_next (= January 1 of next year) | Matched |
| First year substeps | 13 (steps 0..12) | 12 (steps 1..12) | MAGICC7 has extra step |

The remaining first-year difference (13 vs 12 substeps) affects the initial
transient ("shock" phase) but has negligible impact on converged temperatures.

## Potential Future Improvements

1. **Add an extra substep for the first year** (matching MAGICC7's 13-step first year):
   Only affects initial transient but may help shock-phase parity.

2. **Report annual mean** instead of point-in-time: Would need to accumulate
   substep temperatures and average. Only needed if reference data uses annual means.
