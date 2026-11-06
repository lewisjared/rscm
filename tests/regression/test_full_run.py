"""Component parity against the full emissions-driven MAGICC run.

Each test drives one RSCM component with the reference run's own inputs
and compares its outputs with MAGICC7's, so differences point at that
component alone. See ``regression/full_run.py`` for the fixture.
"""

from __future__ import annotations

from functools import partial

import numpy.testing as npt
import pytest

from regression import full_run
from rscm._lib.magicc import (
    ClimateUDEBBuilder,
    GhgForcingBuilder,
    HalocarbonChemistryBuilder,
    OzoneForcingBuilder,
    PermafrostBuilder,
)
from rscm.magicc import full
from rscm.magicc import prescribed as pr


def test_halocarbons():
    """Concentrations, forcing and EESC of all 41 halocarbons from emissions."""
    run = full_run.load()
    builder = HalocarbonChemistryBuilder.from_parameters({})
    component = builder.build()
    species = [
        d.name.removeprefix("Emissions|")
        for d in component.definitions()
        if d.name.startswith("Emissions|")
    ]
    exogenous = {
        f"Emissions|{s}": (run.series(f"Emissions|{s}"), "kt/yr") for s in species
    }
    exogenous["Lifetime|CH4|OH"] = (run.series("CH4_TAUOH"), "yr")
    exogenous["Surface Temperature"] = (run.series("Surface Temperature"), "K")
    # MAGICC7 reports an F-gas's second-year concentration in its first year
    fgases = set(full.halocarbon_species()[: run.parameters["fgas_n"]])
    defaults = full.default_initial_concentrations()
    initial = {
        f"Atmospheric Concentration|{s}": defaults[s]
        if s in fgases
        else run.series(f"Atmospheric Concentrations|{s}")[0]
        for s in species
    }
    model = full_run.run_components(
        run.years, [component], exogenous=exogenous, initial=initial
    )

    for s in species:
        full_run.assert_close(
            full_run.output(model, f"Atmospheric Concentration|{s}")[1:],
            run.series(f"Atmospheric Concentrations|{s}")[1:],
            err_msg=s,
        )
    for group in ("F-Gases", "Montreal Protocol Halogen Gases"):
        reported = [
            full_run.reported().erf(group, full_run.output(model, name))
            for name in full.halocarbon_forcing(group)
        ]
        full_run.assert_close(
            sum(reported)[1:], run.series(f"{full_run.ERF}|{group}")[1:], err_msg=group
        )
    # MAGICC7 reports no EESC in its first year
    full_run.assert_close(
        full_run.output(model, "EESC")[1:], run.series("EESC_CONC")[1:]
    )


def test_ozone():
    """Stratospheric, tropospheric, N2O and temperature ozone forcing."""
    run = full_run.load()
    component = OzoneForcingBuilder.from_parameters(
        full.ozone_parameters(run.parameters)
    ).build()
    exogenous = {
        "EESC": (run.series("EESC_CONC"), "ppt"),
        "Atmospheric Concentration|CH4": (
            run.series("Atmospheric Concentrations|CH4"),
            "ppb",
        ),
        "Atmospheric Concentration|N2O": (
            run.series("Atmospheric Concentrations|N2O"),
            "ppb",
        ),
        **{
            f"Emissions|{gas}|{ours}": (
                run.series(f"Emissions|{gas}|MAGICC {theirs}"),
                unit,
            )
            for gas, unit in (
                ("NOx", "Mt N/yr"),
                ("CO", "Mt CO/yr"),
                ("NMVOC", "Mt NMVOC/yr"),
            )
            for ours, theirs in (
                ("Fossil", "Fossil and Industrial"),
                ("AFOLU", "AFOLU"),
            )
        },
        "Surface Temperature": (run.series("Surface Temperature"), "K"),
    }
    model = full_run.run_components(run.years, [component], exogenous=exogenous)

    erf = "Effective Radiative Forcing|"
    for ours, magicc in (
        ("Stratospheric Ozone", "Stratospheric Ozone"),
        ("Ozone due to N2O", "OZDUETOEffective Radiative Forcing|N2O"),
        ("Ozone due to Temperature", "OZDUETOTEMPERATURE_ERF"),
    ):
        expected = run.series(magicc if magicc.startswith("OZ") else erf + magicc)
        full_run.assert_close(
            full_run.reported().erf(ours, full_run.output(model, erf + ours))[1:],
            expected[1:],
            err_msg=ours,
        )
    trop = "Tropospheric Ozone"
    boxes = model.timeseries().get_fourbox_timeseries_by_name(erf + trop).values()
    full_run.assert_close(
        full_run.reported().erf(trop, boxes)[1:], run.boxes(erf + trop)[1:]
    )
    full_run.assert_close(
        full_run.reported().erf(trop, pr.area_mean(boxes))[1:],
        run.series(erf + trop)[1:],
    )


def test_ch4_oxidation_strat_h2o():
    """Stratospheric H2O forcing from CH4 oxidation."""
    run = full_run.load()
    ch4 = run.series("Atmospheric Concentrations|CH4")
    n2o = run.series("Atmospheric Concentrations|N2O")
    builder = GhgForcingBuilder.from_parameters(
        {
            "ch4_pi": ch4[0],
            "n2o_pi": n2o[0],
            "ch4_strat_h2o_fraction": run.parameters["ch4_addedstrath2o_percent"],
        }
    )
    exogenous = {
        "Atmospheric Concentration|CO2": (
            run.series("Atmospheric Concentrations|CO2"),
            "ppm",
        ),
        "Atmospheric Concentration|CH4": (ch4, "ppb"),
        "Atmospheric Concentration|N2O": (n2o, "ppb"),
    }
    model = full_run.run_components(run.years, [builder.build()], exogenous=exogenous)
    agent = "CH4 Oxidation Stratospheric H2O"
    name = f"Effective Radiative Forcing|{agent}"
    full_run.assert_close(
        full_run.reported().erf(agent, full_run.output(model, name))[1:],
        run.series(name)[1:],
    )


def test_climate():
    """Temperature from MAGICC7's own box forcing and concentrations."""
    run = full_run.load()
    conc = {
        gas: run.series(f"Atmospheric Concentrations|{gas}")
        for gas in ("CO2", "CH4", "N2O")
    }
    ghg = GhgForcingBuilder.from_parameters(
        {f"{gas.lower()}_pi": values[0] for gas, values in conc.items()}
    )
    climate = ClimateUDEBBuilder.from_parameters(full_run.climate_parameters(run))
    boxes = run.boxes("TOTAL_EFFRF")
    units = {"CO2": "ppm", "CH4": "ppb", "N2O": "ppb"}
    model = full_run.run_components(
        run.years,
        [ghg.build(), climate.build()],
        exogenous={
            f"Atmospheric Concentration|{gas}": (values, units[gas])
            for gas, values in conc.items()
        },
        four_box={full_run.REGIONAL_FORCING: (boxes, "W/m^2")},
        initial=climate.initial_values(float(boxes[0] @ full_run.AREA)),
    )
    temperature = model.timeseries().get_fourbox_timeseries_by_name(
        "Surface Temperature"
    )
    full_run.assert_close(temperature.values(), run.boxes("Surface Temperature"))


@pytest.mark.parametrize("agent", list(full_run.PRESCRIBED))
def test_prescribed_forcing(agent):
    """Aerosol, land-use and solar forcing from input files and emissions."""
    run = full_run.load()
    expected = run.boxes("Effective Radiative Forcing|" + full_run.PRESCRIBED[agent])
    full_run.assert_close(
        full_run.reported().erf(agent, full_run.prescribed_forcing()[agent]), expected
    )


PERMAFROST_FLUX = "Net Land to Atmosphere Flux|{}|Earth System Feedbacks|Permafrost"


def test_permafrost():
    """Permafrost thaw and emissions driven by MAGICC7's own temperature."""
    run = full_run.load_permafrost()
    component = PermafrostBuilder.from_parameters(
        full.permafrost_parameters(run.parameters)
    ).build()
    model = full_run.run_components(
        run.years,
        [component],
        exogenous={"Surface Temperature": (run.series("Surface Temperature"), "K")},
    )
    full_run.assert_close(
        full_run.output(model, "Permafrost Thawed Area")[1:],
        run.series("PFTHAWED_AREA")[1:],
    )
    for gas in ("CO2", "CH4"):
        full_run.assert_close(
            full_run.output(model, f"Emissions|{gas}|Permafrost")[:-1],
            run.series(PERMAFROST_FLUX.format(gas))[:-1],
            err_msg=gas,
        )


#: Permafrost emissions and their units.
PERMAFROST_EMISSIONS = {
    "Emissions|CO2|Permafrost": "GtC/yr",
    "Emissions|CH4|Permafrost": "Mt CH4/yr",
}


def _all_outputs(model):
    ts = model.timeseries()
    getters = (ts.get_timeseries_by_name, ts.get_fourbox_timeseries_by_name)
    return {
        name: next(s for s in (get(name) for get in getters) if s is not None)
        .values()
        .copy()
        for name in ts.names()
    }


def test_prescribed_permafrost_emissions_match_modelled(monkeypatch):
    """Prescribing permafrost emissions from a run reproduces that run bit for bit."""
    run = full_run.load_permafrost()
    modelled = full_run.run_coupled(run)
    emissions = {name: full_run.output(modelled, name) for name in PERMAFROST_EMISSIONS}
    run_components = full.run_components

    def without_permafrost(years, components, *, exogenous, **kwargs):
        n = len(years)
        prescribed = {
            name: (values[:n], PERMAFROST_EMISSIONS[name])
            for name, values in emissions.items()
        }
        kept = [
            c
            for c in components
            if "Permafrost Thawed Area" not in {d.name for d in c.definitions()}
        ]
        assert len(kept) == len(components) - 1
        return run_components(
            years, kept, exogenous={**exogenous, **prescribed}, **kwargs
        )

    monkeypatch.setattr(full, "run_components", without_permafrost)
    prescribed = _all_outputs(full_run.run_coupled(run))
    expected = _all_outputs(modelled)
    assert set(expected) - set(prescribed) == {"Permafrost Thawed Area"}
    for name, values in prescribed.items():
        npt.assert_array_equal(values, expected[name], err_msg=name)
        assert values.tobytes() == expected[name].tobytes(), name


def _assert_temperature(model, run):
    temperature = model.timeseries().get_fourbox_timeseries_by_name(
        "Surface Temperature"
    )
    full_run.assert_close(temperature.values(), run.boxes("Surface Temperature"))


def _assert_concentrations(model, run, gases):
    for gas in gases:
        full_run.assert_close(
            full_run.output(model, f"Atmospheric Concentration|{gas}"),
            run.series(f"Atmospheric Concentrations|{gas}"),
            err_msg=gas,
        )


def _assert_total_erf(model, run):
    total = full_run.Reported(run, model).total()
    full_run.assert_close(total[1:], run.series(full_run.ERF)[1:])


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(full_run.load, id="default"),
        pytest.param(full_run.load_permafrost, id="permafrost"),
        *(
            pytest.param(partial(full_run.load_variant, name), id=name)
            for name in full_run.VARIANTS
        ),
    ],
)
def test_coupled_run(load):
    """The full RSCM chain from emissions against MAGICC7's coupled run."""
    run = load()
    model = full_run.run_coupled(run)
    _assert_temperature(model, run)
    _assert_concentrations(model, run, ("CO2", "CH4", "N2O"))
    _assert_total_erf(model, run)


#: RSCM nitrogen averages and their CARBONCYCLE.OUT columns.
NITROGEN_OUTPUTS = {
    "Nitrogen Limitation Factor": ("NCYCLE_LIMIT_FACTOR", 1e-4),
    "Net Primary Production": ("CURRENT_NPP", 1e-4),
}


def test_nitrogen_coupled_run():
    """Nitrogen limitation, with deposition from emissions, in the coupled chain.

    MAGICC7 writes the nitrogen columns only to CARBONCYCLE.OUT, at ASCII precision.
    """
    run = full_run.load_nitrogen()
    model = full_run.run_coupled(run)
    for name, (theirs, rtol) in NITROGEN_OUTPUTS.items():
        npt.assert_allclose(
            full_run.output(model, name)[:-1],
            run.series(theirs)[:-1],
            rtol=rtol,
            err_msg=name,
        )
    # MAGICC7 labels the pool at the end of a year by the year's start
    npt.assert_allclose(
        full_run.output(model, "Nitrogen Pool")[1:],
        run.series("NCYCLE_NPOOL")[:-1],
        rtol=1e-4,
    )
    _assert_temperature(model, run)
    _assert_concentrations(model, run, ("CO2", "CH4", "N2O"))


def test_run_magicc7_defaults():
    """The packaged defaults reproduce MAGICC7's default run.

    It prescribes concentrations up to 2015 and has permafrost.
    """
    run = full_run.load_default()
    model = full_run.default_model()
    _assert_temperature(model, run)
    _assert_concentrations(
        model, run, ("CO2", "CH4", "N2O", *full.halocarbon_species())
    )
    _assert_total_erf(model, run)


def test_run_magicc7_reads_no_files_given_inputs(monkeypatch):
    """Loaded inputs reproduce the default run without reading a file."""
    run = full_run.load_default()
    inputs = full.load_inputs()
    expected = full_run.default_model()

    def no_reads(path):
        pytest.fail(f"read {path}")

    monkeypatch.setattr(full.pr, "read_input", no_reads)
    model = full.run_magicc7(run.years, full_run.emissions(run), inputs=inputs)
    for name in ("Atmospheric Concentration|CO2", "Effective Radiative Forcing"):
        npt.assert_array_equal(
            full_run.output(model, name),
            full_run.output(expected, name),
            err_msg=name,
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda e: e.pop("CO2|Fossil"), "missing emissions"),
        (lambda e: e.update({"CO3|Fossil": e["CO2|Fossil"]}), "unknown emissions"),
        (lambda e: e.update({"CFC11": e["CFC11"][:-1]}), "one per year"),
    ],
)
def test_run_magicc7_rejects_bad_emissions(change, match):
    run = full_run.load()
    emissions = full_run.emissions(run)
    change(emissions)
    with pytest.raises(ValueError, match=match):
        full.run_magicc7(run.years, emissions)
