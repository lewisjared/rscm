# Variable Registry

The variable registry provides a central source of truth for variable metadata in RSCM.
It ensures consistency across components by defining standard names, units, and time conventions.

## Overview

Variables in RSCM have three key properties:

- **Name**: A hierarchical identifier using `|` as separator (e.g., `Emissions|CO2`)
- **Unit**: The canonical unit for the variable (e.g., `GtC / yr`)
- **Time Convention**: When the value applies within a time period

### Time Conventions

| Convention      | Description            | Use Case                                       |
| --------------- | ---------------------- | ---------------------------------------------- |
| `StartOfYear`   | Value applies at Jan 1 | Stock variables (concentrations, temperatures) |
| `MidYear`       | Value applies at Jul 1 | Flow variables (emissions)                     |
| `Instantaneous` | No temporal averaging  | Derived quantities (radiative forcing)         |

## Using Standard Variables (Rust)

Standard variables are pre-defined in `rscm_core::standard_variables`.
These are commonly used variables that will be used in a range of components.
Use them to ensure consistency across your components.

```rust
use rscm_core::standard_variables::{VAR_CO2_EMISSIONS, VAR_CO2_CONCENTRATION};
use rscm_core::component::RequirementDefinition;

// Reference standard variable names and units in component definitions
let input = RequirementDefinition::scalar_input(
    VAR_CO2_EMISSIONS.name,  // "Emissions|CO2"
    VAR_CO2_EMISSIONS.unit,  // "GtC / yr"
);

// Time convention is automatically available from the registry
assert_eq!(
    input.time_convention(),
    Some(TimeConvention::MidYear)
);
```

### Available Standard Variables

#### Emissions

- `VAR_CO2_EMISSIONS` - CO2 emissions (`Emissions|CO2`, `GtC / yr`, MidYear)
- `VAR_CH4_EMISSIONS` - CH4 emissions (`Emissions|CH4`, `MtCH4 / yr`, MidYear)
- `VAR_N2O_EMISSIONS` - N2O emissions (`Emissions|N2O`, `MtN2O / yr`, MidYear)

#### Concentrations

- `VAR_CO2_CONCENTRATION` - CO2 concentration (`Atmospheric Concentration|CO2`, `ppm`, StartOfYear)
- `VAR_CH4_CONCENTRATION` - CH4 concentration (`Atmospheric Concentration|CH4`, `ppb`, StartOfYear)
- `VAR_N2O_CONCENTRATION` - N2O concentration (`Atmospheric Concentration|N2O`, `ppb`, StartOfYear)

#### Radiative Forcing

- `VAR_CO2_ERF` - CO2 ERF (`Effective Radiative Forcing|CO2`, `W / m^2`, Instantaneous)
- `VAR_CH4_ERF` - CH4 ERF (`Effective Radiative Forcing|CH4`, `W / m^2`, Instantaneous)
- `VAR_N2O_ERF` - N2O ERF (`Effective Radiative Forcing|N2O`, `W / m^2`, Instantaneous)
- `VAR_TOTAL_ERF` - Total ERF (`Effective Radiative Forcing|Total`, `W / m^2`, Instantaneous)

#### Climate

- `VAR_GLOBAL_TEMPERATURE` - Global temperature (`Surface Temperature|Global`, `K`, StartOfYear)
- `VAR_OCEAN_HEAT_UPTAKE` - Ocean heat uptake (`Ocean Heat Uptake`, `W / m^2`, Instantaneous)

## Defining New Variables (Rust)

Use the `define_static_variable!` macro to register new variables at compile time:

```rust
use rscm_core::define_static_variable;
use rscm_core::variable::TimeConvention;

define_static_variable!(
    VAR_MY_VARIABLE,
    name = "Category|Subcategory|Name",
    unit = "kg / yr",
    time_convention = TimeConvention::MidYear,
    description = "Description of what this variable represents",
);

// The variable is now available in the global registry
use rscm_core::variable::VARIABLE_REGISTRY;
let var = VARIABLE_REGISTRY.get_with_static("Category|Subcategory|Name");
assert!(var.is_some());
```

## Using the Registry from Python

In addition to staticlly defined variables in Rust,
users can register variables at run-time.
This is particularly important for Python components.

### Looking Up Variables

```python
from rscm.core import get_variable, list_variables, is_variable_registered

# Check if a variable exists
if is_variable_registered("Emissions|CO2"):
    var = get_variable("Emissions|CO2")
    print(f"Unit: {var.unit}")
    print(f"Time convention: {var.time_convention}")

# List all registered variables
for var in list_variables():
    print(f"{var.name}: {var.unit}")
```

### Registering Variables at Runtime

```python
from rscm.core import (
    VariableDefinition,
    TimeConvention,
    register_variable,
)

# Create a new variable definition
my_var = VariableDefinition(
    name="Custom|Variable",
    unit="units",
    time_convention=TimeConvention.mid_year(),
    description="A custom variable registered from Python",
)

# Register it (will raise ValueError if already exists)
register_variable(my_var)
```

## Preindustrial Values

Variables can have associated preindustrial reference values for computing anomalies.
These can be scalar (global), four-box regional, or hemispheric.

### In Rust

```rust
use rscm_core::variable::PreindustrialValue;

// Scalar (global) preindustrial value
let pi_scalar = PreindustrialValue::Scalar(278.0);  // ppm for CO2

// Four-box regional values [N.Ocean, N.Land, S.Ocean, S.Land]
let pi_fourbox = PreindustrialValue::FourBox([278.0, 278.0, 278.0, 278.0]);

// Hemispheric values [Northern, Southern]
let pi_hemispheric = PreindustrialValue::Hemispheric([278.0, 278.0]);

// Convert any variant to scalar (uses area-weighted average)
let global_value = pi_fourbox.to_scalar();
```

### In Python

```python
from rscm.core import PreindustrialValue

# Create preindustrial values
pi_scalar = PreindustrialValue.scalar(278.0)
pi_fourbox = PreindustrialValue.four_box((278.0, 278.0, 278.0, 278.0))
pi_hemispheric = PreindustrialValue.hemispheric((278.0, 278.0))

# Convert to scalar
global_value = pi_fourbox.to_scalar()
```

### Accessing Preindustrial in Components

```rust
// In a component's solve() method
fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
    // Get preindustrial value for a variable
    if let Some(pi) = input_state.get_preindustrial("Atmospheric Concentration|CO2") {
        let pi_scalar = pi.to_scalar();
        // Use for anomaly calculations...
    }

    // Or get directly as scalar
    let pi = input_state.get_preindustrial_scalar("Atmospheric Concentration|CO2")
        .unwrap_or(278.0);  // Default if not provided

    // ...
}
```

## Variable Naming Conventions

Follow the MAGICC naming convention:

1. Use `|` as the hierarchical separator
2. Use title case for each segment
3. Be specific but concise

Examples:

- `Emissions|CO2` (not `CO2 Emissions` or `emissions_co2`)
- `Atmospheric Concentration|CO2` (not `CO2 Concentration`)
- `Effective Radiative Forcing|CO2` (not `ERF CO2`)
- `Surface Temperature|Global` (not `Global Surface Temperature`)

## Best Practices

1. **Always use standard variables** when they exist rather than defining your own
2. **Check the registry** before creating new variables to avoid duplicates
3. **Use consistent units** - the registry's canonical units are authoritative
4. **Set time conventions correctly** - this affects temporal interpolation
5. **Document new variables** with clear descriptions
6. **Use preindustrial values** when computing anomalies for proper baseline handling
