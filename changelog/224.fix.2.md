Common mistakes in the core Python API raise `ValueError`, `IndexError` or `TypeError` instead of a `PanicException`, which escaped `except Exception`:

- `TimeAxis.from_values` and `TimeAxis.from_bounds` reject fewer than 2 points and unsorted, duplicate, NaN or infinite times, naming the first bad index.
- `Timeseries.set` raises `IndexError` out of range, and `Timeseries.at_time(nan)` raises `ValueError`.
- Linear interpolation of a single-step `Timeseries` holds its one value instead of panicking.
- Window `interpolate(nan)` raises `ValueError` rather than returning the first value.
- Passing a `Component` class to `ModelBuilder.with_component` or `PythonComponent` says to pass an instance.

`TimeAxis`, `Timeseries`, `FourBoxTimeseries` and `HemisphericTimeseries` accept any array-like of reals, such as `np.arange(2000, 2010)` or a list, and name the shape of an array with the wrong dimensions.
`TimeAxis.at`, `TimeAxis.at_bounds`, `Timeseries.at`, `Timeseries.set` and slice indexing count negative indices from the end.
`FourBoxSlice` and `HemisphericSlice` iterate, and compare equal by value, as does `TimeAxis`.
A `KeyboardInterrupt` or `SystemExit` raised in a Python component's `solve` reaches the caller unchanged, rather than as a `RuntimeError`.
`TimeAxis`, `Model`, `ModelBuilder`, `VariableDefinition`, `RustComponent` and `PythonComponent` have concise reprs.
