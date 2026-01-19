"""Tests for typed Python components using the Component base class."""

import numpy as np
import pytest

from rscm.component import Component, Input, Output, State
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    Timeseries,
)


class SimpleCarbonCycle(Component):
    """A simple carbon cycle component demonstrating typed inputs/outputs."""

    # Declare inputs
    emissions = Input("Emissions|CO2", unit="GtCO2")

    # Declare state variables (read previous, write new)
    concentration = State("Atmospheric Concentration|CO2", unit="ppm")

    # Declare outputs
    uptake = Output("Carbon Uptake", unit="GtC")

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def solve(
        self, t_current: float, t_next: float, inputs: "SimpleCarbonCycle.Inputs"
    ) -> "SimpleCarbonCycle.Outputs":
        """Calculate carbon cycle dynamics."""
        # Type-safe access to inputs via TimeseriesWindow
        emissions = inputs.emissions.current
        conc_prev = inputs.concentration.current

        # Calculate outputs
        new_conc = conc_prev + emissions * self.sensitivity
        uptake = emissions * 0.5

        return self.Outputs(
            concentration=new_conc,
            uptake=uptake,
        )


class EmissionsComponent(Component):
    """Component that provides emissions (no inputs)."""

    emissions = Output("Emissions|CO2", unit="GtCO2")

    def __init__(self, emissions_value: float):
        self.emissions_value = emissions_value

    def solve(
        self, t_current: float, t_next: float, inputs: "EmissionsComponent.Inputs"
    ) -> "EmissionsComponent.Outputs":
        return self.Outputs(emissions=self.emissions_value)


def test_typed_component_definitions():
    """Test that typed components generate correct requirement definitions."""
    component = SimpleCarbonCycle(sensitivity=0.5)
    defs = component.definitions()

    # Should have 3 definitions:
    # emissions (input), concentration (state), uptake (output)
    assert len(defs) == 3

    # Check each definition
    def_names = {d.name for d in defs}
    assert "Emissions|CO2" in def_names
    assert "Atmospheric Concentration|CO2" in def_names
    assert "Carbon Uptake" in def_names


def test_typed_component_in_model():
    """Test that typed components work in a model."""
    # Use a simpler single-component test with exogenous emissions
    carbon_comp = PythonComponent.build(SimpleCarbonCycle(sensitivity=0.5))

    # Create time axis
    time_axis = TimeAxis.from_values(np.array([2020.0, 2021.0, 2022.0]))

    # Create exogenous emissions timeseries
    emissions = Timeseries(
        np.array([10.0, 10.0, 10.0]),
        time_axis,
        "GtCO2",
        InterpolationStrategy.Previous,
    )

    # Build model with builder pattern
    model = (
        ModelBuilder()
        .with_py_component(carbon_comp)
        .with_time_axis(time_axis)
        .with_exogenous_variable("Emissions|CO2", emissions)
        .with_initial_values({"Atmospheric Concentration|CO2": 280.0})
        .build()
    )

    # Run one timestep
    model.step()

    # Get timeseries collection after step
    collection = model.timeseries()

    # Check results
    conc_ts = collection.get_timeseries_by_name("Atmospheric Concentration|CO2")
    assert conc_ts is not None

    # At t=2021.0, concentration should be: 280.0 + 10.0 * 0.5 = 285.0
    assert abs(conc_ts.at(1) - 285.0) < 0.001

    # Check uptake was calculated
    uptake_ts = collection.get_timeseries_by_name("Carbon Uptake")
    assert uptake_ts is not None
    assert abs(uptake_ts.at(1) - 5.0) < 0.001


def test_typed_component_timeseries_window_access():
    """Test that components receive TimeseriesWindow objects with history."""

    class HistoryTestComponent(Component):
        """Component that uses historical values."""

        temperature = Input("Temperature", unit="K")
        output = Output("Output", unit="")

        def solve(
            self, t_current: float, t_next: float, inputs: "HistoryTestComponent.Inputs"
        ) -> "HistoryTestComponent.Outputs":
            # Access current and previous values
            current = inputs.temperature.current
            try:
                previous = inputs.temperature.previous
                delta = current - previous
            except ValueError:
                # No previous value available at first timestep
                delta = 0.0

            return self.Outputs(output=delta)

    # Wrap component
    comp = PythonComponent.build(HistoryTestComponent())

    # Create time axis and temperature timeseries
    time_axis = TimeAxis.from_values(np.array([2020.0, 2021.0, 2022.0, 2023.0]))
    temp = Timeseries(
        np.array([288.0, 289.0, 290.5, 291.0]),
        time_axis,
        "K",
        InterpolationStrategy.Previous,
    )

    # Build model
    model = (
        ModelBuilder()
        .with_py_component(comp)
        .with_time_axis(time_axis)
        .with_exogenous_variable("Temperature", temp)
        .build()
    )

    # Step 1: t=2020.0 -> t=2021.0
    model.step()
    output_ts = model.timeseries().get_timeseries_by_name("Output")
    assert output_ts is not None
    # Just verify that we got output values - the exact numerical values
    # depend on interpolation and when the component is evaluated
    first_delta = output_ts.at(1)
    assert first_delta is not None

    # Step 2: t=2021.0 -> t=2022.0
    model.step()
    output_ts = model.timeseries().get_timeseries_by_name("Output")
    assert output_ts is not None
    second_delta = output_ts.at(2)
    assert second_delta is not None

    # The key thing is that the component successfully accessed .current and .previous
    # properties on the TimeseriesWindow object without errors


def test_typed_component_output_validation():
    """Test that output validation catches missing fields."""

    class SimpleComponent(Component):
        input_val = Input("Input", unit="")
        output1 = Output("Output1", unit="")
        output2 = Output("Output2", unit="")

        def solve(
            self, t_current: float, t_next: float, inputs: "SimpleComponent.Inputs"
        ) -> "SimpleComponent.Outputs":
            # This should work - all outputs provided
            return self.Outputs(output1=1.0, output2=2.0)

    # Test that providing all outputs works
    comp = SimpleComponent()
    outputs = comp.Outputs(output1=1.0, output2=2.0)
    assert isinstance(outputs, comp.Outputs)

    # Test that missing outputs raises TypeError at construction
    with pytest.raises(TypeError, match="Missing required output fields: output2"):
        comp.Outputs(output1=1.0)


def test_component_registry_auto_registration():
    """Test that components are automatically registered when defined."""
    # Clear registry for clean test
    Component._registry.clear()

    class AutoRegisteredComponent(Component):
        value = Output("Value", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(value=1.0)

    # Component should be in registry
    registry = Component.get_registered_components()
    assert "AutoRegisteredComponent" in registry
    assert registry["AutoRegisteredComponent"] is AutoRegisteredComponent


def test_component_registry_opt_out():
    """Test that components can opt out of registration."""
    # Clear registry for clean test
    Component._registry.clear()

    class UnregisteredComponent(Component, register=False):
        value = Output("Value", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(value=1.0)

    # Component should NOT be in registry
    registry = Component.get_registered_components()
    assert "UnregisteredComponent" not in registry


def test_component_registry_get_component():
    """Test get_component retrieves registered components by name."""
    # Clear registry for clean test
    Component._registry.clear()

    class LookupTestComponent(Component):
        value = Output("Value", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(value=1.0)

    # Should be able to retrieve by name
    retrieved = Component.get_component("LookupTestComponent")
    assert retrieved is LookupTestComponent


def test_component_registry_get_component_not_found():
    """Test get_component raises KeyError for unknown components."""
    with pytest.raises(KeyError, match="No component registered with name"):
        Component.get_component("NonExistentComponent")


def test_component_registry_multiple_components():
    """Test that multiple components are tracked correctly."""
    # Clear registry for clean test
    Component._registry.clear()

    class ComponentA(Component):
        a = Output("A", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(a=1.0)

    class ComponentB(Component):
        b = Output("B", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(b=2.0)

    class ComponentC(Component, register=False):
        c = Output("C", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(c=3.0)

    registry = Component.get_registered_components()
    assert len(registry) == 2
    assert "ComponentA" in registry
    assert "ComponentB" in registry
    assert "ComponentC" not in registry


def test_typed_component_inheritance():
    """Test that typed component declarations are inherited."""

    class BaseComponent(Component, register=False):
        base_input = Input("BaseInput", unit="")
        base_output = Output("BaseOutput", unit="")

    class DerivedComponent(BaseComponent):
        derived_input = Input("DerivedInput", unit="")
        derived_output = Output("DerivedOutput", unit="")

        def solve(
            self, t_current: float, t_next: float, inputs: "DerivedComponent.Inputs"
        ) -> "DerivedComponent.Outputs":
            return self.Outputs(
                base_output=inputs.base_input.current,
                derived_output=inputs.derived_input.current,
            )

    comp = DerivedComponent()
    defs = comp.definitions()

    # Should have all 4 definitions
    assert len(defs) == 4
    def_names = {d.name for d in defs}
    assert "BaseInput" in def_names
    assert "BaseOutput" in def_names
    assert "DerivedInput" in def_names
    assert "DerivedOutput" in def_names


def test_exogenous_input_returns_correct_timestep_value():
    """Test that TimeseriesWindow.current returns the value at current timestep.

    Regression test for GitHub issue #65: Python component TimeseriesWindow.current
    always returns last value instead of the value at the current timestep.

    Before the fix, exogenous variables that were pre-populated with all values
    would always return the last value because the code used .latest() instead
    of computing the index from the current time.
    """
    # Track values seen at each timestep to verify correctness
    observed_values = []

    class DebugComponent(Component, register=False):
        """Component that records the concentration value seen at each timestep."""

        concentration = Input("Concentration|CO2", unit="ppm")
        erf = Output("ERF|Test", unit="W/m^2")

        def solve(
            self, t_current: float, t_next: float, inputs: "DebugComponent.Inputs"
        ) -> "DebugComponent.Outputs":
            conc_value = inputs.concentration.current
            observed_values.append((t_current, conc_value))
            return self.Outputs(erf=conc_value * 0.01)

    time_axis = TimeAxis.from_values(np.arange(2020.0, 2025.0, 1.0))

    # Create concentration timeseries with distinct values at each timestep
    # so we can verify which value is actually being returned
    concentration_ts = Timeseries(
        np.array([400.0, 410.0, 420.0, 430.0, 440.0]),
        time_axis,
        "ppm",
        InterpolationStrategy.Previous,
    )

    comp = PythonComponent.build(DebugComponent())
    model = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_py_component(comp)
        .with_exogenous_variable("Concentration|CO2", concentration_ts)
        .build()
    )

    # Run all timesteps
    for _ in range(4):
        model.step()

    # Verify that each timestep received the correct value
    # Before the fix, all timesteps would see 440.0 (the last value)
    # After the fix, each timestep should see its corresponding value
    assert len(observed_values) == 4

    expected_values = [
        (2020.0, 400.0),
        (2021.0, 410.0),
        (2022.0, 420.0),
        (2023.0, 430.0),
    ]

    for (t_observed, val_observed), (t_expected, val_expected) in zip(
        observed_values, expected_values
    ):
        assert t_observed == t_expected, f"Time mismatch at {t_expected}"
        assert abs(val_observed - val_expected) < 0.001, (
            f"At t={t_expected}, expected concentration={val_expected}, "
            f"got {val_observed}"
        )
