"""The coupling report lists each read and write conversion with its component."""

import json

from regression import full_run
from regression.test_order_independence import _input_files
from rscm.magicc import emissions_driven, full
from rscm.names import SURFACE_AIR_TEMPERATURE_CHANGE


def _magicc7():
    run = full_run.load()
    config, inputs = full.from_magicc7(
        run.time_axis,
        full_run.emissions(run),
        parameters=run.parameters,
        input_files=_input_files(),
    )
    return emissions_driven.setup_model(config, inputs, run.time_axis).build()


def test_magicc7_reports_each_global_reader_of_regional_temperature():
    coupling = json.loads(_magicc7().coupling("json"))
    readers = {
        c["name"]
        for c in coupling["components"]
        for v in c.get("inputs", [])
        if v["name"] == SURFACE_AIR_TEMPERATURE_CHANGE and v.get("grid") is None
    }
    reads = [
        c
        for c in coupling["conversions"]
        if c["variable"] == SURFACE_AIR_TEMPERATURE_CHANGE and c["direction"] == "read"
    ]

    assert len(readers) > 1
    assert sorted(c["component"] for c in reads) == sorted(readers)
    assert all(c["grid"] == {"from": "four_box", "to": "scalar"} for c in reads)
