# Developer Guide

This guide is for developers who want to extend RSCM with new components or contribute to the framework.

## Prerequisites

- **Rust 1.75+** - Install from [rustup.rs](https://rustup.rs)
- **Python 3.11+** - We recommend using [pyenv](https://github.com/pyenv/pyenv)
- **uv** - Fast Python package manager. Install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Development Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/lewisjared/rscm.git
cd rscm

# Create Python virtual environment and install dependencies
make virtual-environment

# Build the Rust extension module
make build-dev
```

### Verify Installation

```bash
# Run the full test suite
make test

# Or run tests separately
cargo test --workspace        # Rust tests only
uv run pytest -v              # Python tests only
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
│   ├── rscm-components/      # Pre-built components (CarbonCycle, CO2ERF)
│   ├── rscm-macros/          # Procedural macros (ComponentIO)
│   ├── rscm-two-layer/       # Two-layer climate model component
│   └── rscm-magicc/          # MAGICC components (in development)
├── python/
│   └── rscm/                 # Python package
│       ├── __init__.py
│       ├── core.py           # Re-exports from Rust
│       ├── component.py      # Python component base class
│       ├── two_layer/        # Two-layer model namespace
│       └── _lib/             # PyO3 extension module
├── docs/                     # Documentation (mkdocs)
└── tests/                    # Python tests
```

## Creating Components

### Python Components

For rapid prototyping or integration with Python libraries:

```python
from rscm.component import Component, Input, Output

class MyComponent(Component):
    x = Input("Input Variable", unit="unit")
    y = Output("Output Variable", unit="unit")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(y=inputs.x.current * 2)
```

See the [Python Components Tutorial](../notebooks/component_python.py) for details.

### Rust Components

For production performance or complex ODE systems, use the `ComponentIO` derive macro:

```rust
use rscm_core::{ComponentIO, component::{InputState, OutputState}};

#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    forcing { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[outputs(
    temperature { name = "Surface Temperature", unit = "K" },
)]
pub struct MyComponent {
    pub sensitivity: f64,
}

#[typetag::serde]
impl Component for MyComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let inputs = MyComponentInputs::from_input_state(input_state);
        let outputs = MyComponentOutputs {
            temperature: inputs.forcing.current() * self.sensitivity,
        };
        Ok(outputs.into())
    }
}
```

See the [Rust Components Tutorial](../notebooks/component_rust.md) for details.

## Common Commands

| Command | Description |
|---------|-------------|
| `make virtual-environment` | Set up Python environment |
| `make build-dev` | Build Rust extension (debug mode) |
| `make test` | Run all tests (Rust + Python) |
| `cargo test --workspace` | Run Rust tests only |
| `uv run pytest` | Run Python tests only |
| `make lint` | Run linters (ruff + clippy) |
| `make format` | Format code (ruff + rustfmt) |
| `make docs` | Build documentation |
| `cargo doc --workspace --open` | Build and open Rust docs |

## Further Reading

- [Development Practices](development.md): Workflows, versioning, releasing
- [Rust Tips](rust_tips.md): Rust-specific guidance
- [Key Concepts](../key_concepts.md): Core abstractions (Component, Model, Timeseries)
- [ComponentIO Macro](../notebooks/component_rust.md): Detailed macro documentation
