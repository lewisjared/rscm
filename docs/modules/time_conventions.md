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

RSCM runs a fixed 12 substeps per year with no overlapping scheme.

### Substep loop

```rust
for step_idx in 0..self.parameters.steps_per_year {
    // 12 substeps: step 0, 1, ..., 11
}
```

### Forcing interpolation

```rust
let frac = step_idx as FloatValue / steps;
let erf = erf_start + frac * (erf_end - erf_start);
```

- Step 0: `erf_start` (forcing at t_current)
- Step 11: `erf_start + 11/12 * (erf_end - erf_start)` (forcing at t_current + 11/12)

### Temperature output

The output from `solve_impl` is the temperature after the last substep (step 11).
This represents the temperature at `t_current + 11/12`, NOT at `t_next`.

## The Timing Mismatch

| Aspect | MAGICC7 | RSCM | Difference |
|--------|---------|------|------------|
| Substeps per year (normal) | 12 (steps 1..12) | 12 (steps 0..11) | Same count |
| Last substep forcing time | `year + 12/12 = year+1.0` | `year + 11/12` | 1/12 year later in MAGICC7 |
| Output temperature time | January 1 of next year | ~December 1 of current year | ~1 month offset |
| First year substeps | 13 (steps 0..12) | 12 (steps 0..11) | MAGICC7 has extra step |
| Energy input (normal year) | Forcing at steps 1..12 | Forcing at steps 0..11 | RSCM starts earlier, ends earlier |

The net effect: RSCM's output temperature is ~1 month behind MAGICC7's.
For slowly-varying forcing this is negligible, but during rapid forcing changes
(the "shock" phase), this creates a systematic cool bias because RSCM
hasn't integrated the final month's worth of forcing.

## Resolution Options

1. **Run 12 substeps from step 1 to step 12** (matching MAGICC7's normal year):
   Interpolate forcing at `year + step/12` for step=1..12, not step=0..11.
   This shifts the forcing window to end at t_next instead of stopping 1/12 short.

2. **Add an extra substep for the first year** (matching MAGICC7's 13-step first year):
   Only affects initial transient but may help shock-phase parity.

3. **Report annual mean** instead of point-in-time: Would need to accumulate
   substep temperatures and average. Only needed if reference data uses annual means.
