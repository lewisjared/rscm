Tightened checks in the core Python API that may reject code that ran before:

- `Unit` raises `ValueError` for an unknown unit symbol, such as `Unit("Gt CO2/yer")`, which used to parse.
- A Python component that declares a unit with an unknown symbol raises `ValueError` when added to a `ModelBuilder`. An empty unit is still allowed.
- Indexing a `FourBoxSlice` or `HemisphericSlice` out of range raises `IndexError` rather than `ValueError`.
- `TimeAxis`, `FourBoxSlice` and `HemisphericSlice` compare by value, so they are no longer hashable.
