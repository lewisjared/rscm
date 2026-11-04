# Ocean Carbon Cycle

## Overview

The ocean carbon cycle takes up CO2 from the atmosphere through air-sea gas exchange.
The ocean is the largest long-term sink for anthropogenic CO2,
so this module shapes atmospheric CO2 over decades to millennia.

MAGICC7 uses an impulse response function (IRF) rather than an ocean circulation model.
The IRF emulates the mixed-layer response of a more complex ocean model (3D-GFDL, 2D-BERN, HILDA or BOXDIFF)
at the cost of a convolution over the flux history.

The module represents:

- air-sea gas exchange, driven by the difference in CO2 partial pressure between atmosphere and surface ocean,
- carbonate chemistry (the Revelle buffer),
  which reduces uptake as dissolved inorganic carbon (DIC) accumulates,
- the temperature dependence of CO2 solubility (warmer water holds less CO2),
- vertical mixing into the deep ocean, implicit in the IRF.

## RSCM implementation

- **Component**: `OceanCarbon`, built from Python with `rscm.magicc.OceanCarbonBuilder`.
- **Source**: `crates/rscm-magicc/src/carbon/ocean.rs`,
  with parameters and IRF forms in `crates/rscm-magicc/src/parameters/ocean_carbon.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
  `run_magicc7` uses the component's defaults, which are MAGICC7's 3D-GFDL settings.
- **Tests**: `tests/regression/test_full_run.py::test_coupled_run` and `test_run_magicc7_defaults`
  check the CO2 concentration against MAGICC7 v7.5.3.
  `crates/rscm-magicc/tests/carbon_cycle_physics.rs` and `conservation.rs` check the
  temperature effect, the Revelle buffer and cumulative uptake.

The equations and the [monthly loop](#monthly-loop) below are the conventions that reproduce MAGICC7's CO2.

## Equations

### Air-sea flux

$$
F = k \left( pCO_2^{atm} - pCO_2^{ocn} \right), \qquad k = \frac{s_k}{\tau_k}
$$

where:

- $F$ is the atmosphere-to-ocean flux (ppm yr⁻¹), positive into the ocean,
- $s_k$ is `OCEANCC_SCALE_GASXCHANGE`,
- $\tau_k$ is the model's gas exchange timescale (years, see [model constants](#model-constants)).

MAGICC7 computes the flux per month with $k / 12$ and multiplies it by 12, which gives the same value.

### Impulse response function

The IRF is the fraction of a carbon pulse still in the mixed layer after time $t$ (years).
Each model has an early form, used before its switch time $t_s$, and a late form.

The 3D-GFDL early form is a polynomial:

$$
IRF_{early}(t) = \sum_{i=0}^{6} c_i \, t^i
$$

2D-BERN and HILDA use exponential sums for both forms, as do all models after the switch:

$$
IRF(t) = a_0 + \sum_{i} a_i \, e^{-t / \tau_i}
$$

MAGICC7 writes the constant $a_0$ as a term with a lifetime of $10^{10}$ years.
BOXDIFF's early form is a sum of power laws instead (see its [IRF coefficients](#irf-coefficients)).

The early form is scaled by $f$ = `OCEANCC_SCALE_IMPULSERESPONSE`, which keeps it between 0 and 1:

$$
S(x) = \frac{f x}{f x + 1 - x}
$$

The late form is multiplied by the ratio of the scaled to unscaled early form at the switch,
so the IRF is continuous at $t_s$:

$$
IRF(t) =
\begin{cases}
S\big(IRF_{early}(t)\big) & t < t_s \\[4pt]
IRF_{late}(t) \, \dfrac{S\big(IRF_{early}(t_s)\big)}{IRF_{early}(t_s)} & t \ge t_s
\end{cases}
$$

### Change in dissolved inorganic carbon

The mixed-layer DIC change convolves the monthly flux history with the IRF,
as a left-hand Riemann sum over the $n$ months so far:

$$
\Delta DIC_n = \frac{\mu}{h A} \sum_{i=1}^{n} \frac{F_i}{12} \, IRF\!\left( \frac{n - i}{12} \right)
$$

where:

- $\Delta DIC$ is in µmol kg⁻¹ and $F_i$ is month $i$'s flux (ppm yr⁻¹),
- $h$ is the mixed layer depth (m) and $A$ the ocean surface area (m²),
- $\mu$ = `OCEAN_MICROMOL_PER_PPM_M3_PER_KG` in µmol ppm⁻¹ m³ kg⁻¹,
  from `OCEAN_PPM_PER_MOL` and the seawater density `OCEANCC_DENSITY_KG_PER_M3`:
  $\mu = 10^6 / (5.65770 \times 10^{-15} \times 1026.5) \approx 1.72187 \times 10^{17}$.

### Ocean surface pCO2

The surface pCO2 follows Joos et al. (2001), equations A24 and A25.
The DIC effect (A24) is a fifth-order polynomial:

$$
\Delta pCO_2^{DIC} = a_1 D + 10^{-3} a_2 D^2 - 10^{-5} a_3 D^3 + 10^{-7} a_4 D^4 - 10^{-10} a_5 D^5,
\qquad a_i = b_i + c_i T_0
$$

The temperature effect (A25) is exponential:

$$
pCO_2^{ocn} = \left( pCO_2^{0} + \Delta pCO_2^{DIC} \right) e^{\alpha_T \Delta T}
$$

where:

- $D = \Delta DIC$ (µmol kg⁻¹),
- $b_i$ and $c_i$ are the [pCO2-DIC coefficients](#pco2-dic-coefficients),
- $T_0$ is the model's preindustrial sea surface temperature (°C),
- $pCO_2^{0}$ is the first-year atmospheric CO2, so the ocean starts in equilibrium,
- $\alpha_T$ is `OCEANCC_TEMPFEEDBACK`, 0.03717879 K⁻¹ by default,
- $\Delta T$ is the global-mean surface warming since `CO2_TEMPFEEDBACK_YRSTART`,
  zero before that year and floored at zero.

### Revelle factor

The polynomial in A24 carries the Revelle (buffer) factor implicitly:

$$
R = \frac{\partial \ln pCO_2}{\partial \ln DIC} \approx 10 \text{ to } 15
$$

Ocean pCO2 rises 10 to 15 times faster than DIC in relative terms,
so the ocean takes up carbon less efficiently as it absorbs more.

## State variables

| Variable             | Fortran name                                    | Units | Initial value                  |
| -------------------- | ----------------------------------------------- | ----- | ------------------------------ |
| Ocean surface pCO2   | `pco2s_currentstep`, `DAT_PCO2S_CONC%DATGLOBE`  | ppm   | `DAT_CO2_CONC%DATGLOBE(1)`     |
| Air-sea flux history | `CO2_AIR2OCNFLX_ALLT_PPMPERYR`                  | ppm/yr | 0                             |
| Warming for feedback | `co2_feed_deltatemp`                            | K     | 0                              |
| Atmospheric CO2      | `c_atm_currentstep`                             | ppm   | From `DAT_CO2_CONC`            |
| Tabulated IRF        | `delta_dioc_calculator%irf`                     | 1     | Model-dependent                |

## Inputs and outputs

| Direction | Variable                     | Units  | Fortran variable                             |
| --------- | ---------------------------- | ------ | -------------------------------------------- |
| Input     | Atmospheric CO2              | ppm    | `DAT_CO2_CONC%DATGLOBE(current_year_idx)`    |
| Input     | Other carbon sources         | GtC/yr | `CO2_TERRBIO_AND_FOSSIL_EMIS` (CO2 budget)   |
| Input     | Warming for feedback         | K      | `co2_feed_deltatemp`                         |
| Output    | Monthly air-sea flux         | ppm/yr | `CO2_AIR2OCNFLX_ALLT_PPMPERYR(currentstep)`  |
| Output    | Ocean surface pCO2           | ppm    | `pco2s_nextstep`, `DAT_PCO2S_CONC%DATGLOBE`  |
| Output    | Annual air-sea flux          | GtC/yr | `CO2_AIR2OCEAN_FLUX(year)`                   |
| Output    | End-of-year atmospheric CO2  | ppm    | `DAT_CO2_CONC%DATGLOBE(next_year_idx)`       |

The other carbon sources are fossil emissions, CO2 from fossil CH4 oxidation and permafrost CO2,
minus the net land uptake.
[CO2 budget](module_11_co2_budget.md#budget-closure) assembles them.

## Parameters

### Configurable parameters

| Parameter          | Fortran name                    | RSCM field              | Units | Default    | Description                              |
| ------------------ | ------------------------------- | ----------------------- | ----- | ---------- | ---------------------------------------- |
| Ocean model        | `OCEANCC_MODEL`                 | constructor             | -     | "3D-GFDL"  | "3D-GFDL", "2D-BERN", "HILDA", "BOXDIFF" |
| Gas exchange scale | `OCEANCC_SCALE_GASXCHANGE`      | `gas_exchange_scale`    | 1     | 1.833492   | Scales the gas exchange rate $k$         |
| IRF scale          | `OCEANCC_SCALE_IMPULSERESPONSE` | `irf_scale`             | 1     | 0.9492864  | Scales the early IRF ($f$)               |
| Temperature effect | `OCEANCC_TEMPFEEDBACK`          | `temp_sensitivity`      | K⁻¹   | 0.03717879 | $\alpha_T$ in A25                        |
| Feedback start     | `CO2_TEMPFEEDBACK_YRSTART`      | `tempfeedback_yrstart`  | year  | 1900       | Start of the warming that drives A25     |
| Radiative setting  | `OCEANCC_RAD_SETTING`           | not implemented         | flag  | 0          | If 1, the ocean sees preindustrial CO2   |
| Flux change limit  | `OCEANCC_STABILITY_LIMIT_DIFFLUX` | `flux_change_limit`   | ppm/yr | 0.04      | Largest flux change between months       |

The Takahashi et al. value for the temperature effect is about 0.0423 K⁻¹.
MAGICC7's default of 0.03717879 K⁻¹ is a calibrated value.
The flux change limit is a hard-coded constant in MAGICC7.

### Model constants

`OCEANCC_MODEL` selects one row of hard-coded constants and the model's [IRF coefficients](#irf-coefficients).

| Model   | Switch time $t_s$ (yr) | Gas exchange $\tau_k$ (yr) | Mixed layer $h$ (m) | Area $A$ (m²) | $T_0$ (°C) |
| ------- | ---------------------- | -------------------------- | ------------------- | ------------- | ---------- |
| 3D-GFDL | 1.0                    | 7.66                       | 50.9                | 3.55e14       | 17.7       |
| 2D-BERN | 9.9                    | 7.46                       | 50.0                | 3.5375e14     | 18.2997    |
| HILDA   | 2.0                    | 9.06                       | 75.0                | 3.62e14       | 18.1716    |
| BOXDIFF | 3.2                    | 7.8                        | 75.0                | 3.62e14       | 17.7       |

### pCO2-DIC coefficients

All models share the A24 coefficients.

| $i$                              | 1         | 2        | 3        | 4        | 5        |
| -------------------------------- | --------- | -------- | -------- | -------- | -------- |
| $b_i$ (`delta_ospp_offsets`)     | 1.5568    | 7.4706   | 1.2748   | 2.4491   | 1.5468   |
| $c_i$ (`delta_ospp_coefficients`) | -0.013993 | -0.20207 | -0.12015 | -0.12639 | -0.15326 |

### IRF coefficients

MAGICC7's coefficients carry more digits than the tables in Joos et al. (2001), and these are MAGICC7's.
Lifetimes are in years, and the $10^{10}$-year lifetime is the constant $a_0$.

=== "3D-GFDL"

    Before 1 year, a polynomial in $t$ (years):

    | $i$   | 0   | 1         | 2        | 3         | 4        | 5         | 6       |
    | ----- | --- | --------- | -------- | --------- | -------- | --------- | ------- |
    | $c_i$ | 1.0 | -2.261725 | 14.00231 | -48.77024 | 82.98591 | -67.52663 | 21.0369 |

    After 1 year:

    | Coefficient | 0.01481883 | 0.0194392 | 0.0383437 | 0.0664848 | 0.249664 | 0.70367   |
    | ----------- | ---------- | --------- | --------- | --------- | -------- | --------- |
    | Lifetime    | 1e10       | 347.547   | 65.35929  | 15.28116  | 2.348814 | 0.7017658 |

=== "2D-BERN"

    Before 9.9 years:

    | Coefficient | 0.058648 | 0.07515 | 0.079338 | 0.41413 | 0.24845 | 0.12429  |
    | ----------- | -------- | ------- | -------- | ------- | ------- | -------- |
    | Lifetime    | 1e10     | 9.62180 | 9.23640  | 0.7603  | 0.16294 | 0.032825 |

    After 9.9 years:

    | Coefficient | 0.013691 | 0.012456 | 0.026933 | 0.026994 | 0.036608 | 0.06738 |
    | ----------- | -------- | -------- | -------- | -------- | -------- | ------- |
    | Lifetime    | 1e10     | 331.54   | 107.57   | 38.946   | 11.677   | 10.515  |

    The last early lifetime is 0.032825, as in the v7.5.3 binary.

=== "HILDA"

    Before 2 years:

    | Coefficient | 0.12935 | 0.24093 | 0.24071 | 0.17003 | 0.21898  |
    | ----------- | ------- | ------- | ------- | ------- | -------- |
    | Lifetime    | 1e10    | 4.9792  | 0.96083 | 0.26936 | 0.034569 |

    After 2 years:

    | Coefficient | 0.022936 | 0.035549 | 0.037820 | 0.089318 | 0.13963 | 0.24278 |
    | ----------- | -------- | -------- | -------- | -------- | ------- | ------- |
    | Lifetime    | 1e10     | 232.30   | 68.736   | 18.601   | 5.2528  | 1.2679  |

=== "BOXDIFF"

    Before 3.2 years, a sum of two power laws in $t$ (years):

    $$
    IRF_{early}(t) = \frac{0.1476804}{(t + 0.026540147)^{0.3881032}}
      + \frac{0.3439660}{(t + 0.7751384)^{0.5519552}}
    $$

    After 3.2 years:

    | Coefficient | 0.0197368421 | 0.0315281 | 0.0104691 | 0.0504693 | 0.076817 | 0.118034 | 0.168507 |
    | ----------- | ------------ | --------- | --------- | --------- | -------- | -------- | -------- |
    | Lifetime    | 1e10         | 215.7122  | 148.7718  | 43.50592  | 14.17156 | 4.870225 | 1.63876  |

## Algorithm

### Initialisation

`oceancc_init` sets the [pCO2-DIC coefficients](#pco2-dic-coefficients)
and builds the calculator for `OCEANCC_MODEL`, which takes its row of [model constants](#model-constants).
It then tabulates the [IRF](#impulse-response-function) at every monthly lag of the run.
An unknown model name is a fatal error.

### Monthly loop

The loop runs twelve times inside each annual step of `MAGICC7.f90`.
It is the one place where atmospheric CO2 changes in an emissions-driven year.
Values marked `prev` are one month before `cur`.

```text
sources = CO2_TERRBIO_AND_FOSSIL_EMIS            # GtC/yr, constant over the year
for month in 1..12:
    step = (year - 1) * 12 + month

    c_atm = c_first_year if OCEANCC_RAD_SETTING == 1 else (c_atm_prev + c_atm_cur) / 2
    flux = k * (c_atm - (pco2_prev + pco2_cur) / 2)               # ppm/yr

    # Stability limiter on the change from the previous month
    change = flux - FLUX[step - 1]
    if change > 0.04:  flux = FLUX[step - 1] + 0.04
    if change < -0.04: flux = FLUX[step - 1] - 0.04
    FLUX[step] = flux

    pco2_next = A25(A24(delta_dic(FLUX[1:step])), delta_T)

    if year >= CO2_SWITCHFROMCONC2EMIS_YEAR:
        c_atm_next = c_atm_cur + sources / GTC_PER_PPM / 12 - flux / 12
    else:
        c_atm_next = c_start + (c_end - c_start) * month / 12
    if CO2_CAPCONC_APPLY: c_atm_next = min(c_atm_next, CO2_CAPCONC_PPM)

    c_atm_prev, c_atm_cur = c_atm_cur, c_atm_next
    pco2_prev, pco2_cur = pco2_cur, pco2_next

CO2_AIR2OCEAN_FLUX(year) = mean(FLUX[this year]) * GTC_PER_PPM
next year's CO2 = c_atm_cur
```

In concentration-driven years the interpolation runs from the concentration at the start of the year
to the prescribed value at its end.
`CARBONCYCLE.OUT` labels `AIR2OCEAN_FLUX` with the end of the year in these years,
one year later than in emissions-driven years.

## Numerical considerations

- **Monthly stepping**: the module runs twelve sub-steps a year for stability,
  which costs twelve times the annual work and stores the flux in ppm yr⁻¹ while stepping by months.
- **Stability limiter**: the 0.04 ppm yr⁻¹ cap on the change between months is a numerical device,
  not physics.
- **Two-step averages**: the flux uses the mean of the last two months of atmospheric and ocean pCO2.
- **Convolution cost**: month $n$ sums over all $n$ earlier months, so a run costs $O(N^2)$ in its length.
- **Polynomial range**: A24 is a fit for rising DIC.
  A falling DIC (outgassing) takes it outside the range it was fitted to.

## Issues in MAGICC7

The MAGICC7 source flags these design problems with TODO comments and its own issue numbers.

| MAGICC7 issue | What MAGICC7 does                                                         | Consequence                                                       |
| ------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| None          | Hard-codes the pCO2-DIC coefficients in `oceancc_init`                    | They cannot be configured or differ between models                |
| #206          | Selects whole models by a name string, with per-name hard-coded constants | The IRF coefficients and model constants are not parameters       |
| #207          | Runs the monthly loop in `MAGICC7.f90`, not in the ocean calculator       | Ocean state is split between `mod_ocean_carbon_cycle` and the driver |
| #208          | Averages the last two months and limits the flux change                   | Stability comes from ad hoc devices rather than the integrator    |
| #209          | Steps monthly "because of issues with the stability"                      | The comment doubts this is needed with a better integrator        |
| #210          | Handles the radiative-only setting inside the physics                     | The switch is entangled with the flux calculation                 |
| #211          | Applies the CO2 concentration cap inside the physics                      | The cap is entangled with the atmospheric update                  |

Beyond these:

- The calculator class `CarbonCycleOceanCalculatorMAGICCConstantStepTimeAxis` holds three sub-calculators,
  each with its own build and finalise steps.
  The `*_manager.f90` files keep arrays of 2048 instances for Python interoperability.
- The models are swappable at run time, but they are all a single global mixed layer.
  They differ only in calibration, not in physics.
- There is no biological pump, no circulation change with warming, no change in alkalinity
  and no regional variation.

## Differences from MAGICC7

- RSCM has the 3D-GFDL, 2D-BERN and HILDA models
  (`OceanCarbonParameters::gfdl_3d`, `bern_2d` and `hilda` in Rust), but not BOXDIFF.
  The Python builder defaults to 3D-GFDL, and another model needs its constants and IRF forms passed in.
- There is no radiative-only mode (`OCEANCC_RAD_SETTING`).
- The concentration cap is not implemented
  (see [CO2 budget](module_11_co2_budget.md#differences-from-magicc7)).
- The flux history keeps the last `max_history_months` sub-steps (6000, or 500 years).
  Longer runs drop the oldest months from the convolution, where MAGICC7 keeps them all.

## References

| File                                                            | Lines     | Contents                                         |
| --------------------------------------------------------------- | --------- | ------------------------------------------------ |
| `physics/carbon_cycle_ocean/carbon_cycle_ocean.f90`             | 19-49     | Main calculator type                             |
|                                                                 | 82-209    | `build_from_name_and_parameters` (model choice)  |
|                                                                 | 351-364   | `calc_atm_to_ocn_c_flux`                         |
|                                                                 | 366-394   | `calc_ospp`                                      |
|                                                                 | 396-454   | 3D-GFDL IRF (polynomial, then exponential)       |
|                                                                 | 456-504   | 2D-BERN IRF                                      |
|                                                                 | 506-554   | HILDA IRF                                        |
|                                                                 | 556-597   | BOXDIFF IRF                                      |
| `physics/carbon_cycle_ocean/atmosphere_to_ocean_carbon_flux.f90` | 39-48     | Air-sea flux                                     |
| `physics/carbon_cycle_ocean/ocean_surface_partial_pressure.f90` | 86-115    | Joos A24                                         |
|                                                                 | 117-137   | Joos A25                                         |
| `physics/carbon_cycle_ocean/delta_dissolved_inorganic_carbon.f90` | 130-143 | IRF scaling                                      |
|                                                                 | 211-269   | DIC convolution                                  |
| `physics/carbon_cycle.f90`                                      | 149-156   | `OCEANCC_*` parameters                           |
|                                                                 | 163-189   | `oceancc_init`                                   |
| `MAGICC7.f90`                                                   | 4849-4939 | Monthly loop                                     |
|                                                                 | 4887-4898 | Stability limiter                                |
| `utils/units.f90`                                               | 8-14      | Unit constants                                   |
| `utils/maths.f90`                                               | -         | `exponential_decay`, `integrate_lh_riemann_sum`  |

- Joos et al. (2001), appendix equations A24 and A25, for the ocean surface pCO2.
