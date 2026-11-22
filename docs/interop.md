# Pandas and xarray

`rscm.interop` converts timeseries to and from pandas and xarray.
Every model's results and every kind of prescribed data go through the same conversion,
so the code below works for MAGICC, FaIR, the two-layer model and models you build yourself.

| Function                         | From                             | To                                       |
| -------------------------------- | -------------------------------- | ---------------------------------------- |
| `to_dataframe(timeseries)`       | a collection or a dict of series | a wide `pandas.DataFrame`                |
| `from_dataframe(df)`             | a wide DataFrame                 | a `TimeseriesCollection`                 |
| `to_xarray(timeseries)`          | a collection or a dict of series | an `xarray.Dataset`                      |
| `from_xarray(ds)`                | a Dataset                        | a `TimeseriesCollection`                 |
| `to_inputs(data, time_axis)`     | a DataFrame or a collection      | `(values, unit)` inputs of `setup_model` |
| `from_inputs(inputs, time_axis)` | `(values, unit)` inputs          | a `TimeseriesCollection`                 |

`TimeseriesCollection` has the first four as methods:
`collection.to_dataframe()`, `collection.to_xarray()`,
`TimeseriesCollection.from_dataframe(df)` and `TimeseriesCollection.from_xarray(ds)`.

xarray is optional. Install it with the `xarray` extra, `pip install rscm[xarray]`.

## Results as a DataFrame

```python
results = model.timeseries().to_dataframe()
results.loc["Atmospheric Concentrations|CO2"]
```

The DataFrame has a row for each variable and region and a column for each step start, in decimal years.
Its index levels are `variable`, `unit` and `region`, the IAMC layout that scmdata and pandas-indexing read.
A scalar variable has the region `World`.
A grid variable has a row per region, labelled as MAGICC7 output labels them:

- Scalar: `World`.
- Four-box: `World|Northern Hemisphere|Ocean`, `World|Northern Hemisphere|Land`,
  `World|Southern Hemisphere|Ocean` and `World|Southern Hemisphere|Land`.
- Hemispheric: `World|Northern Hemisphere` and `World|Southern Hemisphere`.

`rscm.interop.REGIONS` holds these labels in grid order.
Modelled values a run has not reached yet are NaN.

## Global means of grid variables

`to_dataframe(global_mean=True)` adds a `World` row to each grid variable:
the mean over its regions, weighted by the grid weights.

```python
results = model.timeseries().to_dataframe(global_mean=True)
results.loc[("Surface Air Temperature Change", "K", "World")]
```

A grid series computes the same mean on its own with `global_mean()`,
and `grid.weights()` gives the weights it uses.
A model's grid series carry the model's grid weights,
so a MAGICC run's four-box series are weighted by `rscm.magicc.prescribed.area_fractions()`.

```python
temperature = model.timeseries()["Surface Air Temperature Change"]
temperature.global_mean().values()
temperature.grid.weights()
```

## Results as an xarray Dataset

```python
ds = model.timeseries().to_xarray(global_mean=True)
ds["Surface Air Temperature Change"].sel(region="World")
```

The Dataset holds the same values as the DataFrame.
Each variable is a data variable on the `time` and `region` dimensions,
NaN in the regions of other grids.
Its attributes hold its `unit`, its `grid`, its `interpolation_strategy` and,
for a grid variable, its `grid_weights`.
A `time_bounds` coordinate holds the time axis.

## Reading data back

`from_dataframe` and `from_xarray` return a `TimeseriesCollection`.
A variable is read on the finest grid it has every region of,
so the `World` rows that `global_mean=True` adds are ignored.

```python
from rscm.interop import from_dataframe

collection = from_dataframe(results)
```

The DataFrame's `attrs` and the Dataset's attributes keep the time axis, interpolation strategies
and grid weights, so converting to pandas or xarray and back gives the same timeseries.
A file such as a CSV keeps none of them.
Pass them when reading one, or the last step is as long as the one before,
interpolation is linear and grids have their default weights:

```python
import pandas as pd

from rscm.core import GridType, InterpolationStrategy
from rscm.magicc.prescribed import area_fractions

df = pd.read_csv("results.csv", index_col=["variable", "unit", "region"])
collection = from_dataframe(
    df,
    time_axis=time_axis,
    interpolation_strategy={"Emissions|CO2": InterpolationStrategy.Previous},
    grid_weights={GridType.FourBox: area_fractions()},
)
```

`variable`, `unit` and `region` may be columns rather than index levels.
Any other index level, such as `scenario` or `model`, must hold one value,
so select one scenario before converting.
`from_dataframe` raises `ValueError` for a variable with two units,
a region on no grid or a grid with regions missing.
Long data, with one row per time, becomes wide with
`long.pivot_table(index=["variable", "unit", "region"], columns="time", values="value")`.

## Scenarios as prescribed data

A `ModelBuilder` takes the collection as prescribed data,
interpolating each series onto the model's time axis with its strategy:

```python
scenario = pd.read_csv("scenario.csv", index_col=["variable", "unit", "region"])
prescribed = from_dataframe(
    scenario, interpolation_strategy=InterpolationStrategy.Previous
)
model = (
    ModelBuilder()
    .with_time_axis(time_axis)
    .with_component(component)
    .with_prescribed_collection(prescribed)
    .build()
)
```

`rscm.magicc` and `rscm.fair` `setup_model` take `(values, unit)` pairs on the model's time axis instead.
`to_inputs` interpolates each series onto `time_axis` and returns them in that form:

```python
from rscm.fair import setup_model
from rscm.interop import from_inputs, to_inputs

inputs = inputs | to_inputs(scenario, time_axis)
model = setup_model(config, inputs, time_axis).run()
```

`from_inputs(inputs, time_axis)` goes the other way,
so `from_inputs(inputs, time_axis).to_dataframe()` shows the inputs that `from_magicc7` or `from_fair` translated.
`to_inputs` keeps each series' unit, so the DataFrame must use the units the model's components declare.
`from_magicc7` takes its emissions keyed by species and sector, such as `CO2|Fossil`,
so a scenario of MAGICC emissions goes in as its `emissions` argument rather than through `to_inputs`.
