"""`rscm.fair.from_fair` against a small `fair.FAIR`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rscm import names
from rscm.fair import from_fair, naming, setup_model
from rscm.validation import InputValidationError, IssueKind

fair = pytest.importorskip("fair")
from fair.interface import fill, initialise  # noqa: E402
from fair.io import read_properties  # noqa: E402

SPECIES = ("CO2 FFI", "CO2 AFOLU", "CO2", "CH4", "N2O", "Solar")
EMISSIONS = {"CO2 FFI": 10.0, "CO2 AFOLU": 1.0, "CH4": 300.0, "N2O": 10.0}


def _fair(**kwargs: Any) -> Any:
    """A FAIR with FaIR's default initial state, ready to run."""
    _, properties = read_properties()
    properties = {s: properties[s] for s in SPECIES}
    f = fair.FAIR(ch4_method="leach2021", **kwargs)
    f.define_time(1750, 1770, 1)
    f.define_scenarios(["s"])
    f.define_configs(["c"])
    f.define_species(list(SPECIES), properties)
    f.allocate()
    f.fill_species_configs()
    for s, value in EMISSIONS.items():
        fill(f.emissions, value, specie=s)
    fill(f.forcing, 0.1, specie="Solar")
    fill(f.species_configs["forcing_temperature_feedback"], -0.2, specie="Solar")
    fill(f.climate_configs["ocean_heat_capacity"], [8.0, 14.0, 100.0])
    fill(f.climate_configs["ocean_heat_transfer"], [1.1, 1.6, 0.9])
    fill(f.climate_configs["deep_ocean_efficacy"], 1.1)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    for state in (
        f.temperature,
        f.cumulative_emissions,
        f.airborne_emissions,
        f.ocean_heat_content_change,
    ):
        initialise(state, 0)
    return f


def _issues(f: Any) -> set[tuple[IssueKind, str]]:
    with pytest.raises(InputValidationError) as info:
        from_fair(f)
    return {(issue.kind, issue.location) for issue in info.value.issues}


def _temperature(config: Any, inputs: Any, time_axis: Any) -> np.ndarray[Any, Any]:
    outputs = setup_model(config, inputs, time_axis).run().timeseries()
    series = outputs.get_timeseries_by_name(names.SURFACE_AIR_TEMPERATURE_CHANGE)
    assert series is not None
    return np.asarray(series.values())


def test_inputs_are_copies() -> None:
    """FaIR adds the forcing feedback in place, which a view would count twice."""
    f = _fair()
    config, inputs, time_axis = from_fair(f)
    f.run(progress=False)
    solar = naming.prescribed_forcing({"name": "Solar", "type": "solar"})
    np.testing.assert_array_equal(inputs[solar][0], 0.1)
    np.testing.assert_allclose(
        _temperature(config, inputs, time_axis),
        f.temperature.values[:, 0, 0, 0],
        rtol=1e-9,
        atol=1e-12,
    )


def test_converting_after_the_run_is_rejected() -> None:
    f = _fair()
    f.run(progress=False)
    assert _issues(f) == {(IssueKind.UNSUPPORTED, "Solar.forcing")}


def test_properties_df_is_read() -> None:
    """FaIR runs from `properties_df`, so a mode switched only there is followed."""
    f = _fair()
    f.properties_df.loc["CH4", "input_mode"] = "concentration"
    fill(f.concentration, 800.0, specie="CH4")
    config, inputs, _ = from_fair(f)
    entry = next(s for s in config["species"] if s["name"] == "CH4")
    assert entry["input_mode"] == "concentration"
    assert naming.concentration("CH4") in inputs


def test_stochastic_run_is_unsupported() -> None:
    f = _fair()
    fill(f.climate_configs["stochastic_run"], True)
    assert _issues(f) == {(IssueKind.UNSUPPORTED, "climate.stochastic_run")}


def test_prescribed_temperature_is_unsupported() -> None:
    f = _fair(temperature_prescribed=True)
    assert _issues(f) == {(IssueKind.UNSUPPORTED, "temperature_prescribed")}


def test_ghg_forcing_offset_is_unsupported() -> None:
    f = _fair()
    f.ghg_forcing_offset = np.zeros((1, 1, 1, len(SPECIES)))
    assert _issues(f) == {(IssueKind.UNSUPPORTED, "ghg_forcing_offset")}


@pytest.mark.parametrize(
    ("state", "specie", "location"),
    [
        ("temperature", None, "temperature"),
        ("concentration", "CO2", "CO2.concentration"),
        ("cumulative_emissions", "CO2 FFI", "CO2 FFI.cumulative_emissions"),
        ("airborne_emissions", "CH4", "CH4.airborne_emissions"),
        ("ocean_heat_content_change", None, "ocean_heat_content_change"),
    ],
)
def test_initial_state_is_unsupported(
    state: str, specie: str | None, location: str
) -> None:
    f = _fair()
    initialise(getattr(f, state), 0.5, **({} if specie is None else {"specie": specie}))
    assert _issues(f) == {(IssueKind.UNSUPPORTED, location)}


def test_initial_gas_partitions_are_unsupported() -> None:
    f = _fair()
    f.gas_partitions.loc[{"specie": "CO2"}] = 0.5
    assert _issues(f) == {(IssueKind.UNSUPPORTED, "CO2.gas_partitions")}


def test_uninitialised_state_is_rejected() -> None:
    f = _fair()
    initialise(f.temperature, np.nan)
    assert _issues(f) == {(IssueKind.NON_FINITE, "temperature")}


def test_nan_emissions_are_rejected() -> None:
    f = _fair()
    f.emissions[5, 0, 0, SPECIES.index("CH4")] = np.nan
    assert _issues(f) == {(IssueKind.NON_FINITE, "Emissions|CH4")}
