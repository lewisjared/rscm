`from_magicc7` now checks its configuration and inputs before it builds a model,
and raises every problem together as one `rscm.validation.InputValidationError`.

- An unknown configuration name is rejected with the closest known name, rather than ignored.
- A MAGICC7 switch for a feature RSCM lacks is rejected when its value would change the run.
  `rscm.magicc.checks` classifies every MAGICC7 name RSCM does not read as ignorable or unsupported.
- Non-finite emissions, concentrations and prescribed forcing are rejected,
  as are negative concentrations, non-positive sensitivities, depths and lifetimes,
  all-zero regional patterns and input files that end before the years they must cover.

It also fixes these differences from MAGICC7:

- `OceanCarbon` keeps the whole flux history by default, as MAGICC7 does,
  so runs past 2250 no longer drift.
- After 2014 the F-gas forcing pattern splits between the hemispheres like 2014,
  so it no longer turns global when fossil CO2 emissions reach zero.
- Nitrate forcing is zero rather than NaN without anthropogenic NOx and NH3.
- Runs start from the concentration files' values in their first year, as MAGICC7 does.
- `slr_expansion_scaling` and `file_slr_thexp_params` set thermal expansion.
- The CH4, N2O and halocarbon lifetime, feedback and meridional flux parameters reach their components.
- Feedback start years and harmonisation years outside the run move onto its first or last year.
