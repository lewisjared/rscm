# Developer Guide

This guide is for developers who want to extend RSCM with new components or contribute to the framework.

## Prerequisites

- **Rust 1.87+**: install from [rustup.rs](https://rustup.rs)
- **Python 3.13+**: uv can install it with `uv python install`
- **uv**: install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **pre-commit**: install with `uv tool install pre-commit` (`make virtual-environment` calls it)

## Development Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/lewisjared/rscm.git
cd rscm

# Create the virtual environment, install pre-commit hooks and build the extension
make virtual-environment
```

### Verify Installation

```bash
# Run the full test suite
make test

# Or run tests separately
cargo test --workspace --exclude rscm  # Rust tests only (rscm is the PyO3 crate)
uv run pytest -m "not slow"            # Python tests, skipping slow ones
```

### Rebuilding After Changes

If you modify Rust code, rebuild the extension:

```bash
make build-dev
```

For Python-only changes, no rebuild is needed.

## Project Structure

```
rscm/
├── crates/                   # Rust source code
│   ├── rscm/                 # PyO3 Python bindings (root extension module)
│   ├── rscm-core/            # Core traits (Component, Model, Timeseries)
│   ├── rscm-components/      # Pre-built components (CarbonCycle, CO2Erf)
│   ├── rscm-macros/          # Procedural macros (ComponentIO)
│   ├── rscm-two-layer/       # Two-layer climate model component
│   ├── rscm-magicc/          # MAGICC components
│   ├── rscm-calibrate/       # Parameter estimation and MCMC sampling
│   └── rscm-doc-gen/         # Component metadata generator for the docs
├── python/
│   └── rscm/                 # Python package
│       ├── core.py           # Re-exports from Rust
│       ├── component.py      # Python component base class
│       ├── components.py     # Builders for rscm-components
│       ├── two_layer/        # Two-layer model namespace
│       ├── magicc/           # MAGICC builders and runner
│       ├── calibrate/        # Calibration API
│       ├── config/           # Configuration layer
│       ├── _lib/             # Type stubs (.pyi) for the extension
│       └── _lib.abi3.so      # Compiled extension (built by maturin)
├── docs/                     # Documentation (mkdocs)
└── tests/                    # Python tests, with parity tests in tests/regression/
```

## Creating Components

Write a component in Python for rapid prototyping or integration with Python libraries.
Write it in Rust, with the `ComponentIO` derive macro, for performance or complex ODE systems.

=== "Python"

    ```python
    from rscm.component import Component, Input, Output

    class MyComponent(Component):
        x = Input("Input Variable", unit="unit")
        y = Output("Output Variable", unit="unit")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(y=inputs.x.get() * 2)
    ```

    See the [Python Components Tutorial](../notebooks/component_python.py) for details.

=== "Rust"

    ```rust
    // Imports are listed in the Rust Components Tutorial.
    #[derive(Debug, Serialize, Deserialize, ComponentIO)]
    #[inputs(
        forcing { name = "Effective Radiative Forcing", unit = "W/m^2" },
    )]
    #[outputs(
        temperature { name = "Surface Air Temperature Change", unit = "K" },
    )]
    pub struct MyComponent {
        pub sensitivity: FloatValue,
    }

    #[typetag::serde]
    impl Component for MyComponent {
        fn definitions(&self) -> Vec<VariableDefinition> {
            Self::generated_definitions()
        }

        fn solve(&self, _t_current: Time, _t_next: Time, input_state: &InputState) -> RscmResult<OutputState> {
            let inputs = MyComponentInputs::from_input_state(input_state);
            let outputs = MyComponentOutputs {
                temperature: inputs.forcing.get() * self.sensitivity,
            };
            Ok(outputs.into())
        }
    }
    ```

    See the [Rust Components Tutorial](../notebooks/component_rust.md) for details.

## Common Commands

| Command                                  | Description                                    |
| ---------------------------------------- | ---------------------------------------------- |
| `make virtual-environment`               | Set up the environment and build the extension |
| `make build-dev`                         | Rebuild the extension after `.rs` changes      |
| `make all`                               | Format, build, lint and test                   |
| `make test`                              | Rebuild, then run all tests (Rust and Python)  |
| `cargo test --workspace --exclude rscm`  | Run Rust tests only                            |
| `uv run pytest -m "not slow"`            | Run Python tests, skipping slow ones           |
| `make lint`                              | Run ruff, mypy, rustfmt and clippy             |
| `make format`                            | Format code (ruff and rustfmt)                 |
| `make validate-pyi`                      | Check `.pyi` stubs match the compiled module   |
| `make docs-metadata`                     | Regenerate component metadata for the docs     |
| `make docs`                              | Build documentation                            |
| `make docs-serve`                        | Serve documentation locally with live reload   |
| `cargo doc --workspace --no-deps --open` | Build and open Rust docs                       |

## Further Reading

- [Development Practices](development.md): Workflows, versioning, releasing
- [Rust Tips](rust_tips.md): Rust-specific guidance
- [Key Concepts](../key_concepts.md): Core abstractions (Component, Model, Timeseries)
- [ComponentIO Macro](../notebooks/component_rust.md): Detailed macro documentation
