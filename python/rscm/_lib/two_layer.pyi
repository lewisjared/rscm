"""Two-layer energy balance climate model.

This module provides the two-layer energy balance model following Held et al. (2010).
The model represents the climate system as two coupled thermal reservoirs:

- **Surface layer**: Fast-responding mixed layer ocean + atmosphere
- **Deep layer**: Slow-responding deep ocean

The model solves coupled ODEs describing heat exchange between layers and
radiative response to forcing.

References
----------
Held, I. M., Winton, M., Takahashi, K., Delworth, T., Zeng, F., & Vallis, G. K. (2010).
Probing the fast and slow components of global warming by returning abruptly to
preindustrial forcing. Journal of Climate, 23(9), 2418-2427.
"""

from rscm._lib.core import ComponentBuilder, RustComponent

class TwoLayerBuilder(ComponentBuilder):
    """Builder for the two-layer energy balance model component.

    The two-layer model solves the following coupled ODEs:

        C_s * dT_s/dt = F - lambda(T_s) * T_s - epsilon * eta * (T_s - T_d)
        C_d * dT_d/dt = eta * (T_s - T_d)

    where lambda(T_s) = lambda0 - a * T_s allows for state-dependent feedbacks.

    # Parameters
    # ----------
    # lambda0 : float
    #     Climate feedback parameter at zero warming (W/(m^2 K)).
    #     Controls the strength of radiative feedback. Higher values mean stronger
    #     negative feedback and lower climate sensitivity. Typical values: 0.8-1.5
    # a : float
    #     Nonlinear feedback coefficient (W/(m^2 K^2)).
    #     Represents state-dependence of climate feedbacks. Positive values indicate
    #     that feedback weakens (sensitivity increases) as temperature rises.
    #     Set to 0 for a linear model. Typical values: 0-0.1
    # efficacy : float
    #     Ocean heat uptake efficacy (dimensionless).
    #     Ratio of the feedback parameter for ocean heat uptake to the equilibrium
    #     feedback parameter. Values > 1 indicate that ocean heat uptake is more
    #     effective at reducing surface warming. Typical values: 1.0-1.8
    # eta : float
    #     Heat exchange coefficient between surface and deep layers (W/(m^2 K)).
    #     Controls the rate of heat transfer from surface to deep ocean.
    #     Higher values mean faster equilibration. Typical values: 0.5-1.0
    # heat_capacity_surface : float
    #     Heat capacity of the surface layer (W yr/(m^2 K)).
    #     Determines thermal inertia of the fast-responding layer.
    #     Typical values: 5-15 (corresponding to ~50-150m mixed layer depth)
    # heat_capacity_deep : float
    #     Heat capacity of the deep ocean layer (W yr/(m^2 K)).
    #     Determines thermal inertia of the slow-responding deep ocean.
    #     Typical values: 50-200 (much larger than surface layer)

    Inputs
    ------
    Effective Radiative Forcing : float
        Total effective radiative forcing (W/m^2)

    Outputs
    -------
    Surface Temperature : float
        Global mean surface temperature anomaly (K)

    Examples
    --------
    >>> builder = TwoLayerBuilder.from_parameters(
    ...     {
    ...         "lambda0": 1.0,
    ...         "a": 0.0,
    ...         "efficacy": 1.0,
    ...         "eta": 0.7,
    ...         "heat_capacity_surface": 8.0,
    ...         "heat_capacity_deep": 100.0,
    ...     }
    ... )
    >>> component = builder.build()

    For a model with state-dependent feedbacks:

    >>> builder = TwoLayerBuilder.from_parameters(
    ...     {
    ...         "lambda0": 1.2,
    ...         "a": 0.05,  # Sensitivity increases with warming
    ...         "efficacy": 1.3,
    ...         "eta": 0.65,
    ...         "heat_capacity_surface": 10.0,
    ...         "heat_capacity_deep": 150.0,
    ...     }
    ... )
    """

    @staticmethod
    def from_parameters(parameters: dict[str, float]) -> TwoLayerBuilder:
        """Create a builder from a parameter dictionary."""
    def build(self) -> RustComponent:
        """Build the two-layer model component."""
