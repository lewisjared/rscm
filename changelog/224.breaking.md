`ModelBuilder.build` now raises instead of falling back to silent defaults:

- A builder with no time axis raises, where it used to run on 2000 to 2099.
- A builder with no components raises.
- `with_initial_values` and `with_initial_grid_values` raise for a name no component reads or produces, suggesting close matches.
- `with_prescribed_variable` raises for a variable no component reads, suggesting close matches, and for a variable a component produces. Collections passed to `with_prescribed_collection` still ignore variables the model does not read.
- `TwoLayerBuilder.from_parameters` rejects parameters that are not finite, a negative `a` or `ocean_heat_transfer`, and a feedback, efficacy or heat capacity that is not positive. In Rust, `TwoLayer::from_parameters` returns a `Result`.
- `build_model` in `rscm.config` needs a `[time]` table with integer `start` and `end` years, rejects unknown keys in `[time]` and `[components]`, checks the climate component type, and raises if no forcing is given instead of returning NaN temperatures. Give the forcing as a constant `[inputs]` entry or pass it with the new `prescribed` argument.
- `rscm.config.ValidationError` is now also a `ValueError`.
