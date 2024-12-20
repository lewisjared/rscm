<!--- --8<-- [start:description] -->
# Rust Simple Climate Model (RSCM)

This is intended as a PoC of the use of Rust for simple climate models.
The aim is to provide a framework for building components in both Rust and Python.

## Design goals

* Fast
* Easy to extend
* Understandable by non-Rust developers
* Can be driven from Python

<!--- --8<-- [end:description] -->
## Getting Started

<!--- --8<-- [start:getting-started] -->

This project comprises of a Rust library and a Python package (`rcsm`)
which includes a Python extension module to provide access to the Rust library.

### Dependencies

The following dependencies are required to build the project:

* Rust
* [uv](https://github.com/astral-sh/uv) (Python package management)

after these dependencies have been installed the local Python environment can be initialised using:

```
make virtual-environment
```

Rust is a compiled language,
the Python extension module must be recompiled after any changes to the Rust code.
This can be done using:

```
make build-dev
```

### Tests

Rust unit tests are embedded alongside the implementation files.
These tests can be built and run using the following (or using RustRover):

```
cargo test
```


### Documentation

A mkdocs-based documentation site is in the `docs/` directory.
These docs provide an overview of the whole project
and provide reference documentation for the Python interface with some example notebooks.

These docs can be built using:

```
make docs
```

The documentation for the Rust codebase can be built using:

```
make docs-rust
```
