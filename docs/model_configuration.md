# RSCM Configuration

This document describes the TOML-based configuration layer for RSCM models.
The API is still a WIP as we figure out how best to serialise and build models
and their configuration.

## Overview

The configuration layer provides:

- **TOML-based config files** for declarative model setup
- **Layered configuration** (defaults -> tuning -> experiment overrides)
- **Bidirectional mapping** with legacy formats (MAGICC .CFG files)
- **Structured parameter metadata** with validation, units, and documentation generation

## Quick Start

```python
from rscm.config import load_config, build_model

# Load and build a model from TOML
config = load_config("configs/two-layer/defaults.toml")
model = build_model(config)
model.run()
```

## Architecture

```
rscm.config
├── __init__.py          # Public API exports
├── base.py              # Base classes: ModelConfig, TimeConfig, InputSpec
├── builder.py           # Model construction from config
├── docs.py              # Documentation generation
├── exceptions.py        # Custom exception hierarchy
├── loader.py            # TOML loading and merging
├── parameters.py        # Parameter metadata system
├── registry.py          # Component builder registry
├── validation.py        # Schema version and validation
└── models/
    ├── __init__.py
    ├── two_layer.py     # Two-layer model config
    └── magicc/
        ├── __init__.py
        ├── config.py    # MAGICC config classes
        ├── legacy.py    # Legacy format conversion
        └── parameters.py # MAGICC parameter registry
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
| `schema` | Configuration schema version |
| `model` | Model metadata (name, type, version) |
| `time` | Time axis configuration |
| `components` | Component definitions and parameters |
| `inputs` | Input data file specifications |
| `outputs` | Output specifications |

Unknown top-level keys trigger a warning and are ignored.

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

## Parameter Metadata System

Parameters can be annotated with rich metadata for validation and documentation.

### Defining Parameters

```python
from dataclasses import dataclass
from rscm.config.parameters import parameter, validate_parameters

@dataclass
class TwoLayerParameters:
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

params = TwoLayerParameters(lambda0=10.0)  # Out of range!
errors = validate_parameters(params)
# errors = ["Parameter 'lambda0' value 10.0 is outside valid range [0.1, 5.0]"]
```

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
from rscm.config import load_config, build_model, build_two_layer_model

# Generic builder (dispatches by model type)
config = load_config("configs/two-layer/defaults.toml")
model = build_model(config)

# Or build specific model type directly
model = build_two_layer_model(config)
```

The builder:

1. Extracts parameters from config
2. Looks up component builder in registry
3. Constructs components with parameters
4. Creates time axis from config
5. Assembles and returns the model

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
# {"SUPPORTED": 6, "NOT_IMPLEMENTED": 8, "NOT_NEEDED": 10, "DEPRECATED": 0, "total": 24}
```

**Parameter statuses:**

| Status | Description |
|--------|-------------|
| `SUPPORTED` | Mapped to RSCM config path |
| `NOT_IMPLEMENTED` | Feature not yet in RSCM |
| `NOT_NEEDED` | Output/file control handled differently |
| `DEPRECATED` | Superseded in MAGICC7 |

## Schema Versioning

Configs specify a schema version for forward/backward compatibility:

```python
from rscm.config import check_schema_version

# Compatible (same major, any minor/patch)
check_schema_version("1.0.0", "1.0.0")  # OK
check_schema_version("1.1.0", "1.0.0")  # Warning (forward-compatible)

# Incompatible (different major version)
check_schema_version("2.0.0", "1.0.0")  # Raises IncompatibleSchemaError
```

**Compatibility rules:**

- Major version mismatch -> `IncompatibleSchemaError`
- Minor version newer in config -> Warning (forward-compatible)
- Otherwise -> Silent (compatible)

## Exception Hierarchy

```
ConfigError (base)
├── ValidationError          # Type mismatches, out-of-range values
├── IncompatibleSchemaError  # Schema version mismatch
└── ComponentNotFoundError   # Component not in registry
```

```python
from rscm.config import (
    ConfigError,
    ValidationError,
    IncompatibleSchemaError,
    ComponentNotFoundError,
)

try:
    builder = component_registry.get("UnknownComponent")
except ComponentNotFoundError as e:
    print(e.name)       # "UnknownComponent"
    print(e.available)  # ["TwoLayer", ...]
```

## API Reference

### Loading Functions

| Function | Description |
|----------|-------------|
| `load_config(path)` | Load single TOML file |
| `load_config_layers(*paths)` | Load and merge multiple TOML files |
| `deep_merge(base, override)` | Deep merge two dicts |

### Building Functions

| Function | Description |
|----------|-------------|
| `build_model(config)` | Build model from config (dispatches by type) |
| `build_two_layer_model(config)` | Build two-layer model specifically |

### Parameter Functions

| Function | Description |
|----------|-------------|
| `parameter(...)` | Create dataclass field with metadata |
| `get_parameter_metadata(cls)` | Extract metadata from dataclass |
| `validate_parameters(instance)` | Validate instance against metadata |

### Documentation Functions

| Function | Description |
|----------|-------------|
| `generate_parameter_docs(cls)` | Generate markdown documentation |
| `export_parameter_json(cls)` | Export metadata as JSON |

### Registry Functions

| Function | Description |
|----------|-------------|
| `register_component(name)` | Decorator to register component |
| `component_registry.register(name, cls)` | Direct registration |
| `component_registry.get(name)` | Look up builder by name |
| `component_registry.list()` | List all registered names |
| `component_registry.is_registered(name)` | Check if registered |

### Validation Functions

| Function | Description |
|----------|-------------|
| `check_schema_version(config, loader)` | Check version compatibility |
| `parse_semver(version)` | Parse "MAJOR.MINOR.PATCH" string |
| `find_unknown_keys(data, known)` | Find keys not in known set |

## Two-Layer Model Parameters

The two-layer energy balance model (Held et al., 2010) uses these parameters:

| Parameter | Unit | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `lambda0` | W/(m² K) | 1.0 | [0.1, 5.0] | Climate feedback at zero warming |
| `a` | W/(m² K²) | 0.0 | [0.0, 1.0] | Nonlinear feedback coefficient |
| `efficacy` | - | 1.0 | [0.5, 3.0] | Ocean heat uptake efficacy |
| `eta` | W/(m² K) | 0.7 | [0.1, 2.0] | Heat exchange coefficient |
| `heat_capacity_surface` | W yr/(m² K) | 8.0 | [1.0, 50.0] | Surface layer heat capacity |
| `heat_capacity_deep` | W yr/(m² K) | 100.0 | [10.0, 500.0] | Deep ocean heat capacity |

## Example Configurations

### Default Two-Layer

```toml
[model]
name = "two-layer-default"
type = "two-layer"
version = "1.0.0"

[time]
start = 1750
end = 2100

[components.climate]
type = "TwoLayer"

[components.climate.parameters]
lambda0 = 1.0
a = 0.0
efficacy = 1.0
eta = 0.7
heat_capacity_surface = 8.0
heat_capacity_deep = 100.0
```

### High Climate Sensitivity Tuning

```toml
# Override file - use with load_config_layers()
[model]
name = "two-layer-high-ecs"

[components.climate.parameters]
lambda0 = 0.7    # Lower feedback = higher sensitivity
efficacy = 1.3
```

### Programmatic Configuration

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
