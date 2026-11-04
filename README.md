<!--- --8<-- [start:description] -->
# Rust Simple Climate Model (RSCM)

RSCM is a framework for building reduced-complexity climate models in Rust and Python.
Models are assembled from components that are coupled through named variables,
and solved on a common time axis.
It is also a testbed for modularising MAGICC and reimplementing it in Rust.

## Design goals

* Fast
* Easy to extend
* Understandable by non-Rust developers
* Can be driven from Python

## What ships

* A two-layer energy balance model (`rscm.two_layer`).
* Example components for a one-box carbon cycle, CO2 forcing and four-box ocean heat uptake (`rscm.components`).
* MAGICC7 components (`rscm.magicc`) for:
  * CH4, N2O and halocarbon chemistry.
  * The terrestrial carbon cycle with nitrogen limitation, ocean carbon, CO2 budget and permafrost.
  * GHG, ozone and aerosol forcing, and forcing aggregation with efficacies.
  * The upwelling-diffusion climate and sea level rise.
* `rscm.magicc.full.run_magicc7`, which runs the full MAGICC7 chain from emissions.
  It matches MAGICC7 v7.5.3 on its default SSP245 run.
* Simpler MAGICC runners for concentration-driven and emissions-driven experiments.
* MCMC calibration and point estimation (`rscm.calibrate`).

<!--- --8<-- [end:description] -->
## Getting Started

<!--- --8<-- [start:getting-started-dev] -->

This project comprises a Rust workspace in `crates/` and a Python package (`rscm`)
which includes a Python extension module to provide access to the Rust library.

### Dependencies

The following dependencies are required to build the project:

* Rust
* [uv](https://github.com/astral-sh/uv) (Python package management)

After these dependencies have been installed the local Python environment can be initialised using:

```bash
make virtual-environment
```

Rust is a compiled language,
so the Python extension module must be recompiled after any changes to the Rust code.
This can be done using:

```bash
make build-dev
```

### Tests

Rust unit tests are embedded alongside the implementation files.
The `rscm` crate holds the Python bindings, so it is excluded when running the Rust tests directly:

```bash
cargo test --workspace --exclude rscm
```

`make test` rebuilds the extension and runs both the Python and Rust test suites.

### Documentation

A mkdocs-based documentation site is in the `docs/` directory.
These docs provide an overview of the whole project
and provide reference documentation for the Python interface with some example notebooks.

These docs can be built using:

```bash
make docs
```

The documentation for the Rust codebase can be built using:

```bash
make docs-rust
```
<!--- --8<-- [end:getting-started-dev] -->
