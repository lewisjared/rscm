import numpy as np
import pytest

from rscm._lib.core import InterpolationStrategy, Model, TimeAxis, Timeseries
from rscm.core import ModelBuilder
from rscm.fair import GasCycleBuilder

AXIS = TimeAxis.from_values(np.arange(2000.0, 2010.0))


def gas(name, lifetime=8.25):
    return {
        "driver": "emissions",
        "emissions": [f"Emissions|{name}"],
        "emissions_unit": "Mt CH4/yr",
        "amount_unit": "Mt CH4",
        "concentration": f"Atmospheric Concentrations|{name}",
        "concentration_unit": "ppb",
        "partition_fraction": [1.0, 0.0, 0.0, 0.0],
        "unperturbed_lifetime": [lifetime] * 4,
        "baseline_concentration": 729.2,
        "baseline_emissions": 0.0,
        "concentration_per_emission": 0.35,
        "iirf_0": 8.25,
        "iirf_airborne": 0.00032,
        "iirf_uptake": 0.0,
        "iirf_temperature": -0.3,
        "g0": 0.368,
        "g1": 8.25,
    }


def gas_cycle(*species):
    return GasCycleBuilder.from_parameters({"species": list(species)}).build()


def series(value, unit):
    return Timeseries(
        np.full(len(AXIS), value), AXIS, unit, InterpolationStrategy.Previous
    )


@pytest.fixture
def record():
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component_as("Gases", gas_cycle(gas("A"), gas("B")))
        .with_prescribed_variable("Emissions|A", series(10.0, "Mt CH4/yr"))
        .with_prescribed_variable("Emissions|B", series(100.0, "Mt CH4/yr"))
        .with_prescribed_variable("Surface Air Temperature Change", series(0.5, "K"))
        .build()
    )
    model.step()
    model.step()
    return model.checkpoint()


def test_reordered_species_fail_branch(record):
    with pytest.raises(
        ValueError,
        match="saved state of 'Gases' does not fit its parameters, species 0 is "
        "'Atmospheric Concentrations\\|B' in the parameters",
    ):
        Model.branch(record, parameters={"Gases": gas_cycle(gas("B"), gas("A"))})


def test_a_new_lifetime_branches_and_runs(record):
    branched = Model.branch(
        record, parameters={"Gases": gas_cycle(gas("A"), gas("B", lifetime=4.0))}
    )
    branched.run()
    restored = Model.restore(record)
    restored.run()

    def last(model):
        return model.timeseries()["Atmospheric Concentrations|B"].values()[-1]

    assert last(branched) < last(restored)
