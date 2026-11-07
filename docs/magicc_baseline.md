# Offline MAGICC concentration baseline

The baseline report compares `rscm.magicc.run_magicc` with a checked-in MAGICC7 SSP2-4.5 run
that is driven by concentrations and uses IPCCTAR GHG forcing.
It reports the differences without a pass or fail threshold.
The full chain's parity test is separate,
and [Agreement with MAGICC7](magicc_runs.md#agreement-with-magicc7) describes it.

The command runs locally without an external MAGICC executable or network access
once the RSCM environment is installed.

```sh
make build-dev
uv run python scripts/regression/compare_magicc_baseline.py --output-dir /tmp/rscm-baseline
```

Use a new output directory each time.
Existing directories are rejected.
The optional `--data-dir` selects a directory containing `01_concentration_driven.csv`
and `01_concentration_driven_config.json`.
The default is resolved relative to the repository even when invoking the script from elsewhere.

The command writes three files only after the run and serialisation succeed:

- `comparison.csv` has 1,755 rows: five observables at all 351 annual labels from 1750 through 2100.
  These are CO2, CH4 and N2O ERF, total ERF and surface temperature.
  Each row retains its raw reference timestamp, comparison year, units, actual and reference values,
  signed error, absolute error and relative error.
- `metadata.json` records fixture hashes, repository revision and source hashes,
  supplied and resolved Rust parameters, assumptions and unknown reference provenance.
- `summary.md` reports each observable's maximum absolute error and its year,
  plus signed errors at both endpoints.
  No early or final years are discarded.

Signed error is actual minus reference.
Relative error divides signed error by absolute reference.
At exactly zero reference it is blank in the CSV, and absolute error is still reported.
Small nonzero references are not masked.

## Interpretation

- A successful command means the inputs were complete and valid, and the run and its files succeeded.
- It does not mean scientific parity.
  Numerical disagreement is report data and does not change the exit status.
- Missing or ambiguous series, incomplete years, unexpected units, nonfinite values, invalid parameters
  and execution or write failures exit nonzero.

The residual is reference total ERF minus reference CO2, CH4 and N2O ERF.
It is added to the runner's calculated gas ERFs,
so total forcing does not independently validate the processes represented by that residual.

This case uses first-year concentrations as preindustrial references,
and unit rapid-adjustment factors for IPCCTAR when the configuration does not set them.
It sets the climate `erf_2xco2` to the configured `core_delq2xco2`
rather than the runner's default CO2-doubling ERF.
Climate begins with zero temperature and ocean anomalies, without spin-up.
The effective Rust defaults are recorded through model serialisation.
The reference JSON contains overrides only.
Its full resolved configuration, executable identity, consumed input hashes and initial climate state
are unknown.
Source hashes identify the checkout, not proof that an extension was rebuilt,
so run `make build-dev` before comparing after Rust changes.

Every value is compared at its original calendar label.
This is provisional, especially for temperature:
[time conventions](modules/time_conventions.md) describe the reference output timing
and the final year's substeps, where RSCM still differs from MAGICC7.
The command never searches for a shift that reduces error.
The comparison settings are provisional, and there is no temperature tolerance or parity verdict.

The baseline covers only this concentration-driven case.
It has no ocean diagnostics, verified reference regeneration or prescribed-forcing control.

## Verification

```sh
uv run pytest tests/regression/test_magicc_baseline.py tests/test_magicc_runner.py
```

The tests consume committed data and exercise invalid input, endpoint coverage,
zero-reference errors, residual reconstruction and incomplete-write cleanup.
