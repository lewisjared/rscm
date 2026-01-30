"""Type stubs for the RSCM units module.

This module provides unit parsing, normalization, and conversion functionality
for climate model variables. It supports flexible unit string syntax and
automatic conversion factor calculation.
"""

from typing import final

@final
class Unit:
    """A physical unit with parsing, normalization, and conversion support.

    This class provides comprehensive support for working with physical units
    in climate models. It handles parsing of unit strings with flexible syntax,
    dimensional analysis, and conversion factor calculation.

    Parsing
    -------
    The parser accepts several equivalent notations:

    - Exponents: `m^2`, `m**2`, `m2`
    - Division: `W/m^2`, `W m^-2`, `W per m^2`
    - Multiplication: `kg m`, `kg*m`
    - Whitespace: `W/m^2` == `W / m ^ 2`

    Examples
    --------
    >>> from rscm._lib.core import Unit
    >>>
    >>> # Parse and compare units
    >>> u1 = Unit("W/m^2")
    >>> u2 = Unit("W / m ^ 2")
    >>> assert u1 == u2  # Same normalized form
    >>>
    >>> # Convert between compatible units
    >>> gtc = Unit("GtC/yr")
    >>> mtco2 = Unit("MtCO2/yr")
    >>> factor = gtc.conversion_factor(mtco2)  # ~3666.67
    """

    def __init__(self, unit_str: str) -> None:
        """Create a new Unit from a unit string.

        Parameters
        ----------
        unit_str
            The unit string to parse. Accepts flexible syntax including:
            - Exponents: `m^2`, `m**2`, `m2`
            - Division: `W/m^2`, `W m^-2`, `W per m^2`
            - Multiplication: `kg m`, `kg*m`

        Raises
        ------
        ValueError
            If the unit string cannot be parsed.

        Examples
        --------
        >>> u = Unit("W/m^2")
        >>> u = Unit("GtC / yr")
        >>> u = Unit("kg m^-2 s^-1")
        """

    @property
    def original(self) -> str:
        """Return the original input string used to create this unit.

        Returns
        -------
        str
            The original input string.

        Examples
        --------
        >>> u = Unit("W / m ^ 2")
        >>> u.original
        'W / m ^ 2'
        """

    def normalized(self) -> str:
        """Return the normalized string representation of this unit.

        The normalized form is canonical: units with positive exponents
        first (alphabetically), then `/`, then units with negative exponents.

        Returns
        -------
        str
            The normalized unit string.

        Examples
        --------
        >>> Unit("W / m ^ 2").normalized()
        'W / m^2'
        >>> Unit("m^-2 W").normalized()
        'W / m^2'
        """

    def is_dimensionless(self) -> bool:
        """Check if this unit is dimensionless.

        Units like "ppm", "ppb", and "1" are dimensionless.

        Returns
        -------
        bool
            True if the unit is dimensionless, False otherwise.

        Examples
        --------
        >>> Unit("ppm").is_dimensionless()
        True
        >>> Unit("W/m^2").is_dimensionless()
        False
        """

    def is_compatible(self, other: Unit) -> bool:
        """Check if this unit can be converted to another unit.

        Units are compatible if they have the same physical dimension.
        For example, GtC/yr and MtCO2/yr are compatible because they
        both represent mass flux.

        Parameters
        ----------
        other
            The target unit to check compatibility with.

        Returns
        -------
        bool
            True if conversion is possible, False otherwise.

        Examples
        --------
        >>> gtc = Unit("GtC/yr")
        >>> mtco2 = Unit("MtCO2/yr")
        >>> gtc.is_compatible(mtco2)
        True
        >>> flux = Unit("W/m^2")
        >>> gtc.is_compatible(flux)
        False
        """

    def conversion_factor(self, target: Unit) -> float:
        """Calculate the conversion factor from this unit to another unit.

        The factor is the multiplier to convert a value in this unit to a
        value in the target unit. For example, if this unit is `GtC/yr` and
        the target is `MtCO2/yr`, the factor is approximately 3666.67.

        Parameters
        ----------
        target
            The target unit to convert to.

        Returns
        -------
        float
            The conversion factor.

        Raises
        ------
        ValueError
            If the units have incompatible dimensions.

        Examples
        --------
        >>> gtc = Unit("GtC/yr")
        >>> mtco2 = Unit("MtCO2/yr")
        >>> factor = gtc.conversion_factor(mtco2)
        >>> round(factor, 2)
        3666.67
        """

    def convert(self, value: float, target: Unit) -> float:
        """Convert a value from this unit to another unit.

        This is a convenience method equivalent to:
        `value * self.conversion_factor(target)`

        Parameters
        ----------
        value
            The value to convert.
        target
            The target unit to convert to.

        Returns
        -------
        float
            The converted value.

        Raises
        ------
        ValueError
            If the units have incompatible dimensions.

        Examples
        --------
        >>> gtc = Unit("GtC/yr")
        >>> mtco2 = Unit("MtCO2/yr")
        >>> gtc.convert(0.34, mtco2)
        1246.666...
        """

    def __repr__(self) -> str:
        """Return a string representation of this unit."""

    def __str__(self) -> str:
        """Return the normalized string representation."""

    def __eq__(self, other: Unit) -> bool:  # type: ignore[override]
        """Check equality with another unit.

        Two units are equal if they have the same normalized representation.
        """

    def __hash__(self) -> int:
        """Compute hash for use in sets and dicts."""
