`rscm.magicc.full.from_magicc7` no longer runs a two-year model to set the F-gas forcing at the first time bound.
`rscm.magicc.model.setup` now calculates it from HalocarbonChemistry's first step,
so it follows parameters changed on the configuration.
Configurations no longer carry an F-gas entry in `initial_forcing`.

With `ncycle_apply_ndep_from_emis`, `TerrestrialCarbon` now calculates nitrogen deposition from the NOx and NH3 emissions
through its new `nitrogen.deposition_from_emissions` parameters.
`rscm.magicc.full.nitrogen_inputs` now returns those emissions with their units instead of the deposition,
and its `bugs` argument moved to `rscm.magicc.full.nitrogen_parameters`.
