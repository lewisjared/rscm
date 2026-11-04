# External Modules

A new module does not need to live in the RSCM repository.
Package it on its own, depend on `rscm`, and install it next to RSCM.
This guide covers two layouts:

- [Python components](#python-components): the whole module is Python.
- [Python components with a Rust kernel](#python-components-with-a-rust-kernel):
  the physics is Rust in your own extension, wrapped in a thin Python component.

Both run in any model alongside the components that ship with RSCM.
Contribute the module to RSCM as a [Rust component](notebooks/component_rust.md)
once it should run at native speed with no Python in the loop.

## Python components

Create a package with uv:

```bash
uv init --lib rscm-decay
cd rscm-decay
uv add rscm
```

Write the component in `src/rscm_decay/__init__.py`.
This one is a carbon pool that takes up emissions and decays with a fixed timescale:

```python
import math

from rscm.component import Component, Input, State


class DecayingPool(Component):
    """Carbon pool that takes up emissions and decays with a fixed timescale."""

    inflow = Input("Emissions|CO2", unit="GtC/yr")
    stock = State("Carbon Pool", unit="GtC")

    def __init__(self, timescale: float):
        self.timescale = timescale

    def solve(self, t_current, t_next, inputs):
        retained = math.exp(-(t_next - t_current) / self.timescale)
        stock = inputs.stock.at_start() * retained + inputs.inflow.at_start() * (
            self.timescale * (1 - retained)
        )
        return self.Outputs(stock=stock)
```

See [Components in Python](notebooks/component_python.py) for the full component API.

### Register the components

Declare an entry point in the `rscm.components` group of `pyproject.toml`:

```toml
[project.entry-points."rscm.components"]
rscm_decay = "rscm_decay"
```

The value is the module that defines the components.
RSCM imports it the first time the component registry is queried,
so installed components are found by name without an import:

```python
from rscm.component import Component

DecayingPool = Component.get_component("DecayingPool")
```

`Component.get_registered_components()` lists them too,
and RSCM warns rather than fails when an entry point cannot be imported.
Components are keyed by class name,
so give them names that will not clash with other packages.

### Use the module

Install the package into the same environment as RSCM
and add its components to a model like any other Python component:

```python
from rscm.core import ModelBuilder, PythonComponent
from rscm_decay import DecayingPool

model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_py_component(PythonComponent.build(DecayingPool(timescale=50.0)))
    .with_exogenous_variable("Emissions|CO2", emissions)
    .with_initial_values({"Carbon Pool": 0.0})
    .build()
)
```

`model.to_toml()` saves the component's module and class name,
so `Model.from_toml()` restores it wherever the package is installed.
See [Saving Models](notebooks/component_python.py#saving-models) for what a component must do to be saved,
and load only model files you trust, because restoring one imports the modules it names.

## Python components with a Rust kernel

Move the physics into Rust when `solve()` itself is the slow part.
The component stays in Python and calls a function from your own compiled extension.

Create a package with a [maturin](https://www.maturin.rs) build:

```bash
uv init --lib --build-backend maturin rscm-decay
cd rscm-decay
uv add rscm
```

Write the kernel in `src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pymodule]
mod _core {
    use pyo3::prelude::*;

    /// Stock after one step of first-order decay with a constant inflow.
    #[pyfunction]
    fn decay_step(stock: f64, inflow: f64, timescale: f64, dt: f64) -> f64 {
        let retained = (-dt / timescale).exp();
        stock * retained + inflow * timescale * (1.0 - retained)
    }
}
```

Call it from the component in `src/rscm_decay/__init__.py`:

```python
from rscm.component import Component, Input, State

from rscm_decay._core import decay_step


class DecayingPool(Component):
    """Carbon pool that takes up emissions and decays with a fixed timescale."""

    inflow = Input("Emissions|CO2", unit="GtC/yr")
    stock = State("Carbon Pool", unit="GtC")

    def __init__(self, timescale: float):
        self.timescale = timescale

    def solve(self, t_current, t_next, inputs):
        stock = decay_step(
            inputs.stock.at_start(),
            inputs.inflow.at_start(),
            self.timescale,
            t_next - t_current,
        )
        return self.Outputs(stock=stock)
```

Register it with the same [entry point](#register-the-components).
`uv sync` builds the extension, and `uv build` produces wheels to publish.

Pass plain floats and NumPy arrays between the component and the kernel.
Your extension compiles its own copy of any RSCM crate it depends on,
so RSCM objects such as a `RustComponent` built there are a different type
and `ModelBuilder` rejects them.
The extension does not need the same PyO3 version as RSCM for the same reason.

### When a kernel helps

Every Python component pays a fixed cost per timestep
to build its inputs and read its outputs.
The kernel speeds up only the work inside `solve()`,
so it pays off when that work is large compared with the bridge,
for example an implicit solver or a loop over many layers.
Time the Python version first.
