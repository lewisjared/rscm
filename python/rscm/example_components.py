"""
Example Python components for RSCM

These components demonstrate the Python component API and serve as examples
for implementing custom components.
"""

from rscm.component import Component, Input, Output, State


class LinearConcentration(Component):
    r"""
    Linear concentration component

    A simple component that calculates atmospheric CO2 concentration
    from emissions using a linear accumulation model with decay.

    This component demonstrates the Python component API with:

    - Input variables
    - State variables (read previous, write new)
    - Output variables
    - Parameters passed to __init__

    $$
    C_{t+1} = C_t + E \cdot \Delta t \cdot \alpha - (C_t - C_0) \cdot \beta
    $$

    Where:

    - $C$ is the atmospheric CO2 concentration
    - $E$ is the emissions rate
    - $\alpha$ is the airborne fraction
    - $\beta$ is the decay rate
    - $C_0$ is the pre-industrial concentration
    """

    # Declare inputs
    emissions = Input("Emissions|CO2", unit="GtCO2 / yr")

    # Declare state variables (read previous value, write new value)
    concentration = State("Atmospheric Concentration|CO2", unit="ppm")

    # Declare outputs
    airborne_emissions = Output("Airborne Emissions|CO2", unit="GtCO2 / yr")

    def __init__(
        self,
        airborne_fraction: float = 0.5,
        decay_rate: float = 0.01,
        preindustrial_concentration: float = 278.0,
    ):
        """
        Initialise the linear concentration component.

        Parameters
        ----------
        airborne_fraction
            Fraction of emissions that remain in the atmosphere (default: 0.5)
        decay_rate
            Rate at which excess CO2 decays per year (default: 0.01)
        preindustrial_concentration
            Pre-industrial CO2 concentration in ppm (default: 278.0)
        """
        self.airborne_fraction = airborne_fraction
        self.decay_rate = decay_rate
        self.preindustrial_concentration = preindustrial_concentration

    def solve(
        self,
        t_current: float,
        t_next: float,
        inputs: "LinearConcentration.Inputs",
    ) -> "LinearConcentration.Outputs":
        """
        Calculate concentration at the next timestep.

        Parameters
        ----------
        t_current
            Current time (years)
        t_next
            Next timestep time (years)
        inputs
            Typed inputs with emissions and previous concentration

        Returns
        -------
        Outputs with new concentration and airborne emissions
        """
        dt = t_next - t_current

        # Get current values from inputs
        emissions = inputs.emissions.current
        conc_prev = inputs.concentration.current

        # Calculate airborne emissions
        airborne = emissions * self.airborne_fraction

        # Calculate new concentration with decay
        excess = conc_prev - self.preindustrial_concentration
        decay = excess * self.decay_rate * dt
        new_conc = conc_prev + airborne * dt - decay

        return self.Outputs(
            concentration=new_conc,
            airborne_emissions=airborne,
        )
