import pytest

from rscm.core import Unit


@pytest.mark.parametrize(
    "spaced, fused",
    [
        ("Gt C", "GtC"),
        ("Gt C/yr", "GtC/yr"),
        ("Gt C / yr", "GtC / yr"),
        ("Mt CO2/yr", "MtCO2/yr"),
        ("Pg C/yr", "PgC/yr"),
    ],
)
def test_spaced_and_fused_carbon_are_equal(spaced, fused):
    assert Unit(spaced) == Unit(fused)
    assert hash(Unit(spaced)) == hash(Unit(fused))


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("W / m ^ 2", "W/m^2"),
        ("m^-2 W", "W/m^2"),
        ("Gt C / yr", "GtC/yr"),
        ("GtC/yr", "GtC/yr"),
        ("MtCO2/yr", "Mt CO2/yr"),
        ("Mt CO2 / yr", "Mt CO2/yr"),
        ("Mt CH4 / yr", "Mt CH4/yr"),
        ("Mt SO2/yr", "Mt SO2/yr"),
        ("Mt NO2/yr", "Mt NO2/yr"),
        ("kt N2O/yr", "kt N2O/yr"),
        ("Mt NMVOC/yr", "Mt VOC/yr"),
        ("kt HFC4310/yr", "kt HFC4310mee/yr"),
        ("kt HFC134a / yr", "kt HFC134a/yr"),
        ("1 / yr", "1/yr"),
    ],
)
def test_normalized_is_compact(unit, expected):
    assert Unit(unit).normalized() == expected
    assert str(Unit(unit)) == expected
    assert Unit(expected).normalized() == expected


@pytest.mark.parametrize(
    "source, target, factor",
    [
        ("GtC/yr", "Mt CO2/yr", 1000 * 44 / 12),
        ("Gt CO2/yr", "Gt C/yr", 12 / 44),
        ("Mt S/yr", "Mt SO2/yr", 64 / 32),
        ("Mt N/yr", "Mt NO2/yr", 46 / 14),
        ("Mt N/yr", "Mt NOx/yr", 46 / 14),
        ("Mt NOx/yr", "Mt NO2/yr", 1.0),
        ("Mt N2ON/yr", "kt N2O/yr", 1000 * 44 / 28),
        ("Mt CH4/yr", "kt CH4/yr", 1000.0),
        ("Mt NMVOC/yr", "Mt VOC/yr", 1.0),
        ("kt HFC4310/yr", "kt HFC4310mee/yr", 1.0),
    ],
)
def test_species_family_conversion(source, target, factor):
    source_unit = Unit(source)
    target_unit = Unit(target)
    assert source_unit.is_compatible(target_unit)
    forward = source_unit.conversion_factor(target_unit)
    backward = target_unit.conversion_factor(source_unit)
    assert forward == pytest.approx(factor, rel=1e-12)
    assert backward == pytest.approx(1 / factor, rel=1e-12)


@pytest.mark.parametrize(
    "source, target",
    [
        ("Mt CH4/yr", "Mt CO2/yr"),
        ("Mt N/yr", "kt N2O/yr"),
        ("Mt NH3/yr", "Mt N/yr"),
        ("Mt S/yr", "GtC/yr"),
        ("kt CFC11/yr", "kt CFC12/yr"),
        ("Mt CH4/yr", "Mt/yr"),
    ],
)
def test_different_species_never_convert(source, target):
    source_unit = Unit(source)
    target_unit = Unit(target)
    assert not source_unit.is_compatible(target_unit)
    with pytest.raises(ValueError, match="different species"):
        source_unit.conversion_factor(target_unit)


def test_unknown_species_names_token():
    with pytest.raises(ValueError, match="unknown species: 'XYZ'"):
        Unit("Mt XYZ/yr")


@pytest.mark.parametrize("unit", ["CO2", "W CH4", "Mt * CH4"])
def test_species_must_follow_mass(unit):
    with pytest.raises(ValueError, match="must follow a mass unit"):
        Unit(unit)


def test_original_is_preserved():
    unit = Unit("Gt C / yr")
    assert unit.original == "Gt C / yr"
    assert repr(unit) == "Unit('GtC/yr')"
