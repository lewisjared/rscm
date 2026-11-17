Added input validation to `rscm.fair`.
`setup_model`, `components`, `run_ensemble`, `read_calibration` and `from_fair` raise one `rscm.validation.InputValidationError` listing every problem before a model is built.
It covers unknown keys, non-finite parameters and inputs, fewer than two ocean layers, missing species, input lengths and FaIR options that `from_fair` cannot reproduce.

Fixed several differences from FaIR:

- `run_ensemble` inverted concentrations to emissions over the wrong steps.
- `run_ensemble` failed when CO2 was concentration-driven.
- `from_fair` returned views of FaIR's arrays and ignored `properties_df`.
- ERFari, ERFaci and the minor linear forcings did not skip a NaN term as FaIR does.
- The energy balance started from zero forcing, which changed the second temperature when the first prescribed forcing was nonzero.
