Fixed model building and configuration edge cases:

- Prescribing a variable twice, or after a collection holding it, now replaces the earlier data instead of panicking.
- `TimeseriesCollection.add_timeseries` raises `ValueError` for a duplicate name instead of panicking, and accepts four-box and hemispheric timeseries.
- `TimeseriesCollection` supports `collection[name]`, which raises `KeyError` with close matches, as well as `in` and `len`.
- `ModelBuilder.build` and `build_models` warn when prescribed data has to be extrapolated to cover the time axis, naming the variable and the years.
- Component builders' `from_parameters` errors name the component, list every missing parameter with its type, suggest close matches for unknown parameters and name the parameter with a wrong type.
- `build_model` reads `[initial_values]` and constant `[inputs]` entries, and the shipped `configs/two-layer/defaults.toml` loads without a warning and runs to finite temperatures.
- The incompatible schema error links to the schema versioning docs.
