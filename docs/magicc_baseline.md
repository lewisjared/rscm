# Offline MAGICC concentration baseline

Compare the checked-in SSP245/IPCCTAR concentration reference with the partial
MAGICC runner. This command runs locally without an external MAGICC executable
or network access once the RSCM environment is installed.

```sh
make build-dev
uv run python scripts/regression/compare_magicc_baseline.py --output-dir /tmp/rscm-baseline
```

Use a new output directory each time. Existing directories are rejected. The
optional `--data-dir` selects a directory containing `01_concentration_driven.csv`
and `01_concentration_driven_config.json`; the default is resolved relative to
the repository even when invoking the script from elsewhere.

The command writes three files only after the run and serialization succeed:

- `comparison.csv` has 1,755 rows: five observables at all 351 annual labels from
  1750 through 2100. These are CO2, CH4 and N2O ERF, total ERF and surface temperature.
  Each row retains its raw reference timestamp, comparison year, units, actual
  and reference values, signed error, absolute error and relative error.
- `metadata.json` records fixture hashes, repository revision and source hashes,
  supplied and resolved Rust parameters, assumptions and unknown reference provenance.
- `summary.md` reports each observable's maximum absolute error and its year,
  plus signed errors at both endpoints. No early or final years are discarded.

Signed error is actual minus reference. Relative error divides signed error by
absolute reference. At exactly zero reference it is blank in CSV; absolute error
is still reported. Small nonzero references are not masked.

## Interpretation

A successful command means the data and execution contracts passed. It does not
mean scientific parity. Numerical disagreement is report data and does not change
the exit status. Missing or ambiguous series, incomplete years, unexpected units,
nonfinite values, invalid parameters and execution or write failures exit nonzero.

The residual is reference total ERF minus reference CO2, CH4 and N2O ERF. It is
added to the runner's calculated gas ERFs, so total forcing does not independently
validate the unfinished processes represented by that residual.

This case follows the existing reference test's use of first-year concentrations
as preindustrial references and unit rapid-adjustment factors for IPCCTAR when
unspecified. Climate begins with zero temperature and ocean anomalies, without
spin-up. The effective Rust defaults are recorded through model serialization.
The reference JSON contains overrides only; its full resolved configuration,
executable identity, consumed input hashes and initial climate state are unknown.
Source hashes identify the checkout, not proof that an extension was rebuilt;
run `make build-dev` before comparing after Rust changes.

Every value is compared at its original calendar label. This is provisional,
especially for temperature: [time conventions](modules/time_conventions.md)
describe differences in reference output timing and first/final substeps. The
command never searches for a shift that reduces error. All comparability fields
remain provisional and there is no temperature tolerance or parity verdict.

Ocean diagnostics, verified reference regeneration, a prescribed-forcing control,
and scientific review of regression expectations belong to later slices.

## Verification

```sh
uv run pytest tests/regression/test_magicc_baseline.py
uv run pytest tests/test_magicc_runner.py tests/regression/test_ghg_forcing.py
```

The tests consume committed data and exercise invalid input, endpoint coverage,
zero-reference errors, residual reconstruction and incomplete-write cleanup.
