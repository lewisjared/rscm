Moved MAGICC7 aerosol, land-use and BC on snow preparation that depends on parameters into the components,
so the AR6 ensemble's aerosol parameters take effect when changed on a built configuration.

- `ForcingAgent` gained `harmonisation`, `sum_of_scale_from`, `spreading_pattern`, `constant_after_year`,
  `relative_to_year`, `pattern_terms` and `pattern_year`.
  `from_magicc7` uses them for the direct SOx, BC and OC, mineral dust, land-use and BC on snow agents,
  so their `Effective Radiative Forcing|...` inputs are now unscaled,
  and their regional patterns come from the scaled forcing in `rf_regions_normyear`.
- `ForcingAggregationBuilder.aggregate` takes the prescribed inputs and time axis,
  which harmonised, held and relative agents read.
  `variable_scales` and `variable_box_forcing` give the scales and scaled box forcing.
- Adds the `NitrateCloudForcing` component, required by the MAGICC models.
  It calculates nitrate and cloud albedo forcing in the run from emissions and the
  `Optical Thickness|...` inputs, which `full.nitrate_cloud_inputs` provides.
  `full.prescribed_forcing` no longer returns them.
  `rscm.magicc.prescribed.nitrate`, `nitrate_pattern` and `cloud_albedo` were removed.
