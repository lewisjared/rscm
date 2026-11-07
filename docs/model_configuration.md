# Configuration Layer

The configuration layer describes RSCM models in TOML files.
The API is still a work in progress as we figure out how best to serialise and build models
and their configuration.

## Overview

The configuration layer provides:

- **TOML-based config files** for declarative model setup
- **Layered configuration** (defaults -> tuning -> experiment overrides)
- **Bidirectional mapping** with legacy formats (MAGICC .CFG files)
- **Structured parameter metadata** with validation, units, and documentation generation

## Quick Start

/// warning | Work in progress
`build_model(config)` builds the time axis and components and sets zero initial values,
but it does not yet read the `[inputs]` or `[initial_values]` tables.
A model built from `configs/two-layer/defaults.toml` therefore has no forcing, and its temperatures are NaN.
///

Until `build_model` reads inputs, take the parameters and time axis from the config
and assemble the model with `ModelBuilder`.
Run this from the repository root, where the example configs live:

```python
import numpy as np
from rscm.config import load_config
from rscm.core import InterpolationStrategy, ModelBuilder, TimeAxis, Timeseries
from rscm.two_layer import TwoLayerBuilder

config = load_config("configs/two-layer/defaults.toml")
params = config["components"]["climate"]["parameters"]
time_axis = TimeAxis.from_values(
    np.arange(config["time"]["start"], config["time"]["end"] + 1, dtype=float)
)
erf = Timeseries(np.linspace(0.0, 4.0, len(time_axis)), time_axis, "W/m^2", InterpolationStrategy.Linear)

model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_component(TwoLayerBuilder.from_parameters(params).build())
    .with_prescribed_variable("Effective Radiative Forcing", erf)
    .with_initial_values({"Surface Temperature": 0.0, "Deep Ocean Temperature": 0.0})
).build()
model.run()
```

## Configuration File Format

### Basic Structure

```toml
# configs/two-layer/defaults.toml

[model]
name = "two-layer-default"
type = "two-layer"
version = "1.0.0"
config_schema = "1.0.0"
description = "Two-layer energy balance model"

[time]
start = 1750
end = 2100

[components.climate]
type = "TwoLayer"

[components.climate.parameters]
lambda0 = 1.0        # Climate feedback (W/(m² K))
a = 0.0              # Nonlinear coefficient (W/(m² K²))
efficacy = 1.0       # Ocean heat uptake efficacy
eta = 0.7            # Heat exchange coefficient (W/(m² K))
heat_capacity_surface = 8.0   # Surface layer (W yr/(m² K))
heat_capacity_deep = 100.0    # Deep ocean (W yr/(m² K))

[inputs]
# "Effective Radiative Forcing" = { file = "data/erf.csv", unit = "W/m^2" }

[initial_values]
# "Surface Temperature" = 0.0
```

### Known Top-Level Keys

| Key | Description |
|-----|-------------|
| `schema` | Accepted, but nothing reads it |
| `model` | Model metadata (name, type, version) |
| `time` | Time axis configuration |
| `components` | Component definitions and parameters |
| `inputs` | Input data file specifications |
| `outputs` | Output specifications |

The shipped configs record their schema version as `config_schema` under `[model]`,
and `ModelConfig` has a matching `config_schema` field.
Unknown top-level keys are logged as a warning and ignored.
`initial_values` is not yet a known key, so the `[initial_values]` table in `defaults.toml` logs a warning.
`load_config` does not check the schema version.
Call `check_schema_version` yourself (see [Schema Versioning](#schema-versioning)).

## Layered Configuration

Configs can be layered to build up from defaults:

```python
from rscm.config import load_config_layers

# Later files override earlier ones (nested dicts merge recursively)
config = load_config_layers(
    "configs/two-layer/defaults.toml",      # Base defaults
    "configs/two-layer/tuning/high-ecs.toml" # Override specific params
)
```

**Example override file:**

```toml
# configs/two-layer/tuning/high-ecs.toml

[model]
name = "two-layer-high-ecs"
description = "Two-layer model tuned for high climate sensitivity"

[components.climate.parameters]
lambda0 = 0.7    # Lower feedback = higher sensitivity
efficacy = 1.3   # Higher efficacy
```

**Merge behaviour:**

- Nested dicts merge recursively
- Lists and scalar values are replaced (not concatenated)
- Override values take precedence

## Programmatic Configuration

The TOML examples above live in `configs/two-layer/`.
Configurations can also be built in Python and passed to `build_model`,
with the limitation described in the [Quick Start](#quick-start):

```python
from rscm.config.base import TimeConfig
from rscm.config.models.two_layer import TwoLayerConfig, TwoLayerParameters

config = TwoLayerConfig(
    name="my-experiment",
    time=TimeConfig(start=1850, end=2100),
    climate=TwoLayerParameters(
        lambda0=0.8,
        efficacy=1.2,
    ),
)
```

## Parameter Metadata System

Parameters can be annotated with rich metadata for validation and documentation.

### Defining Parameters

```python
from dataclasses import dataclass
from rscm.config.parameters import parameter, validate_parameters

@dataclass
class ExampleParameters:
    lambda0: float = parameter(
        default=1.0,
        unit="W/(m² K)",
        description="Climate feedback parameter at zero warming",
        range=(0.1, 5.0),           # Hard validation limits
        typical_range=(0.8, 1.5),    # Guidance for typical values
        source="Held et al. (2010)", # Citation
    )

    a: float = parameter(
        default=0.0,
        unit="W/(m² K²)",
        description="Nonlinear feedback coefficient",
        range=(0.0, 1.0),
        deprecated=False,
    )
```

### Parameter Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `default` | Any | Default value (required field if not provided) |
| `unit` | str | Physical unit (e.g., "K", "W/m^2") |
| `description` | str | Human-readable description |
| `range` | tuple[float, float] | Hard validation range (min, max) |
| `typical_range` | tuple[float, float] | Soft guidance range |
| `choices` | list | Valid enum-like choices |
| `source` | str | Citation or reference |
| `deprecated` | bool | Whether parameter is deprecated |
| `deprecated_message` | str | Deprecation warning message |

### Validation

```python
from rscm.config.parameters import validate_parameters

params = ExampleParameters(lambda0=10.0)  # Out of range!
errors = validate_parameters(params)
# errors = ["Parameter 'lambda0' value 10.0 is outside valid range [0.1, 5.0]"]
```

`rscm.config.models.two_layer.TwoLayerParameters` runs this validation in `__post_init__`,
so constructing it with an out-of-range value raises `ValueError`.

Validation checks:

- Values within `range` bounds
- Values in `choices` list (if specified)
- Deprecation warnings for deprecated parameters

### Documentation Generation

```python
from rscm.config import generate_parameter_docs, export_parameter_json
from rscm.config.models.two_layer import TwoLayerParameters

# Generate markdown documentation
markdown = generate_parameter_docs(TwoLayerParameters)
print(markdown)

# Export to JSON (for tooling/Rust schema compatibility)
json_data = export_parameter_json(TwoLayerParameters)
```

**Generated markdown example:**

```markdown
# TwoLayerParameters

Two-layer energy balance model parameters.

## Parameters

### `lambda0`

Climate feedback parameter at zero warming

- **Unit**: W/(m² K)
- **Valid range**: [0.1, 5.0]
- **Typical range**: [0.8, 1.5]
- **Source**: Held et al. (2010)
```

## Component Registry

Components are registered by name, enabling config files to reference them:

```python
from rscm.config.registry import component_registry, register_component
from rscm.two_layer import TwoLayerBuilder

# Register via decorator
@register_component("MyComponent")
class MyComponentBuilder:
    pass

# Or register directly
component_registry.register("TwoLayer", TwoLayerBuilder)

# Look up by name
builder_cls = component_registry.get("TwoLayer")

# List all registered components
names = component_registry.list()  # ["MyComponent", "TwoLayer"]
```

The two-layer model is automatically registered when importing `rscm.config.models.two_layer`.

## Model Building

```python
from rscm.config import load_config, build_model

# Dispatches on the model type
config = load_config("configs/two-layer/defaults.toml")
model = build_model(config)
```

The builder:

1. Extracts parameters from config
2. Looks up component builder in registry
3. Constructs components with parameters
4. Creates an annual time axis from `start` to `end` inclusive
5. Sets zero initial values for the state variables
6. Assembles and returns the model

It adds no prescribed inputs yet, as the [Quick Start](#quick-start) warns.

`build_model` only supports `type = "two-layer"` and raises `ValueError` for any other model type.
`MagiccConfig` in `rscm.config.models.magicc` describes MAGICC parameters,
but cannot yet be built into a model.
`rscm.magicc.full.run_magicc7` runs the full MAGICC7 chain without the config layer.
See [MAGICC Runs](magicc_runs.md#full-magicc7-chain).

## Legacy Format Support (MAGICC)

Bidirectional conversion between MAGICC .CFG format and RSCM config:

### Import from MAGICC

```python
from rscm.config.models.magicc.legacy import from_legacy_dict

# Flat MAGICC config dict
legacy = {
    "startyear": 1850,
    "endyear": 2100,
    "core_climatesensitivity": 3.0,
}

# Convert to nested RSCM config
rscm_config = from_legacy_dict(legacy)
# {
#     "time": {"start": 1850, "end": 2100},
#     "components": {"climate": {"parameters": {"climate_sensitivity": 3.0}}}
# }
```

The mapping also covers the permafrost `pf_*` parameters.
For example, `pf_apply` maps to `components.permafrost.enabled`.
`MagiccConfig` has no permafrost section yet, so these paths only exist in the dict form.

### Export to MAGICC

```python
from rscm.config.models.magicc.legacy import to_legacy_dict

rscm_config = {
    "time": {"start": 1850, "end": 2100},
}
legacy = to_legacy_dict(rscm_config)
# {"startyear": 1850, "endyear": 2100}
```

### Parameter Status Tracking

The MAGICC parameter registry tracks support status:

```python
from rscm.config.models.magicc.parameters import (
    MAGICC_PARAMETERS,
    ParameterStatus,
    get_coverage_report,
    get_coverage_stats,
)

# Check a parameter's status
param = MAGICC_PARAMETERS["core_climatesensitivity"]
param.status  # ParameterStatus.SUPPORTED
param.rscm_path  # "components.climate.parameters.climate_sensitivity"

# Generate coverage report
report = get_coverage_report()  # Markdown report

# Get statistics
stats = get_coverage_stats()
# {"SUPPORTED": 68, "NOT_IMPLEMENTED": 14, "NOT_NEEDED": 12, "DEPRECATED": 0, "total": 94}
```

**Parameter statuses:**

| Status | Description |
|--------|-------------|
| `SUPPORTED` | Mapped to RSCM config path |
| `NOT_IMPLEMENTED` | Feature not yet in RSCM |
| `NOT_NEEDED` | Output/file control handled differently |
| `DEPRECATED` | Superseded in MAGICC7 |

These statuses describe the config-layer mapping, not what `run_magicc7` supports.
For example, `ch4_switchfromconc2emis_year` is `NOT_IMPLEMENTED` here, but `run_magicc7` reads it.

## Schema Versioning

Configs specify a schema version for forward/backward compatibility:

```python
from rscm.config import check_schema_version

# Compatible (same major, any minor/patch)
check_schema_version("1.0.0", "1.0.0")  # OK
check_schema_version("1.1.0", "1.0.0")  # Logs a warning (forward-compatible)

# Incompatible (different major version)
check_schema_version("2.0.0", "1.0.0")  # Raises IncompatibleSchemaError
```

**Compatibility rules:**

- Major version mismatch -> `IncompatibleSchemaError`
- Minor version newer in config -> logged warning (forward-compatible)
- Otherwise -> Silent (compatible)

## Exception Hierarchy

```text
ConfigError (base)
├── ValidationError          # Reserved for config validation, not yet raised
├── IncompatibleSchemaError  # Schema version mismatch
└── ComponentNotFoundError   # Component not in registry
```

```python
from rscm.config import ComponentNotFoundError, component_registry

try:
    builder = component_registry.get("UnknownComponent")
except ComponentNotFoundError as e:
    print(e.name)       # "UnknownComponent"
    print(e.available)  # ["TwoLayer", ...]
```

## API Reference

The [`rscm.config` API](api/rscm/config/index.md) documents every function and class on this page.
`TwoLayerParameters` in [`rscm.config.models.two_layer`](api/rscm/config/models/two_layer/index.md)
holds the two-layer defaults and valid ranges,
and the [TwoLayer component page](components/twolayer.md) describes its physics.
