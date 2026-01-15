class ScalarRegion:
    """Region enum for scalar (global) grid"""

    GLOBAL: int = 0

class FourBoxRegion:
    """Region enum for four-box grid"""

    NORTHERN_OCEAN: int = 0
    NORTHERN_LAND: int = 1
    SOUTHERN_OCEAN: int = 2
    SOUTHERN_LAND: int = 3

class HemisphericRegion:
    """Region enum for hemispheric grid"""

    NORTHERN: int = 0
    SOUTHERN: int = 1

class ScalarGrid:
    """
    Single global region (scalar grid).

    Used for backwards compatibility with scalar timeseries and for
    variables that are truly spatially uniform (e.g., atmospheric CO₂).
    """

    def __init__(self) -> ScalarGrid: ...
    def grid_name(self) -> str: ...
    def size(self) -> int: ...
    def region_names(self) -> list[str]: ...
    def aggregate_global(self, values: list[float]) -> float:
        """
        Aggregate all regional values to a single global value.

        Parameters
        ----------
        values
            Regional values to aggregate (must have length equal to size())

        Returns
        -------
        Aggregated global value
        """

class FourBoxGrid:
    """
    MAGICC standard four-box grid structure.

    Divides the world into:
    - Northern Ocean
    - Northern Land
    - Southern Ocean
    - Southern Land
    """

    def __init__(
        self, weights: tuple[float, float, float, float] | None = None
    ) -> FourBoxGrid: ...
    @staticmethod
    def magicc_standard() -> FourBoxGrid:
        """Create a four-box grid with MAGICC standard (equal) weights"""
    @staticmethod
    def with_weights(weights: tuple[float, float, float, float]) -> FourBoxGrid:
        """Create a four-box grid with custom weights"""
    def grid_name(self) -> str: ...
    def size(self) -> int: ...
    def region_names(self) -> list[str]: ...
    def weights(self) -> tuple[float, float, float, float]: ...
    def aggregate_global(self, values: list[float]) -> float:
        """
        Aggregate all regional values to a single global value using weights.

        Parameters
        ----------
        values
            Regional values to aggregate (must have length 4)

        Returns
        -------
        Weighted global average
        """

class HemisphericGrid:
    """
    Simple north-south hemispheric split.

    Divides the world into:
    - Northern Hemisphere
    - Southern Hemisphere
    """

    def __init__(
        self, weights: tuple[float, float] | None = None
    ) -> HemisphericGrid: ...
    @staticmethod
    def equal_weights() -> HemisphericGrid:
        """Create a hemispheric grid with equal weights (0.5 each)"""
    @staticmethod
    def with_weights(weights: tuple[float, float]) -> HemisphericGrid:
        """Create a hemispheric grid with custom weights"""
    def grid_name(self) -> str: ...
    def size(self) -> int: ...
    def region_names(self) -> list[str]: ...
    def weights(self) -> tuple[float, float]: ...
    def aggregate_global(self, values: list[float]) -> float:
        """
        Aggregate all regional values to a single global value using weights.

        Parameters
        ----------
        values
            Regional values to aggregate (must have length 2)

        Returns
        -------
        Weighted global average
        """
