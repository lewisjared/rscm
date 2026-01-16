"""Tests for StateValue class and grid output support."""

import numpy as np
import pytest

from rscm._lib.core import (
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    Timeseries,
)
from rscm._lib.core.state import (
    FourBoxSlice,
    HemisphericSlice,
    StateValue,
)
from rscm.component import Component, Input, Output


class TestStateValueBasics:
    """Test basic StateValue functionality."""

    def test_scalar_creation(self):
        """Test creating a scalar StateValue."""
        sv = StateValue.scalar(42.0)
        assert sv.is_scalar()
        assert not sv.is_four_box()
        assert not sv.is_hemispheric()
        assert sv.as_scalar() == 42.0
        assert sv.as_four_box() is None
        assert sv.as_hemispheric() is None
        assert sv.to_scalar() == 42.0

    def test_four_box_creation(self):
        """Test creating a FourBox StateValue."""
        slice = FourBoxSlice(
            northern_ocean=1.0, northern_land=2.0, southern_ocean=3.0, southern_land=4.0
        )
        sv = StateValue.four_box(slice)
        assert not sv.is_scalar()
        assert sv.is_four_box()
        assert not sv.is_hemispheric()
        assert sv.as_scalar() is None
        fb = sv.as_four_box()
        assert fb is not None
        assert fb.northern_ocean == 1.0
        assert fb.northern_land == 2.0
        assert fb.southern_ocean == 3.0
        assert fb.southern_land == 4.0
        assert sv.as_hemispheric() is None
        # to_scalar should return weighted mean
        assert sv.to_scalar() == pytest.approx(2.5, rel=0.1)

    def test_hemispheric_creation(self):
        """Test creating a Hemispheric StateValue."""
        slice = HemisphericSlice(northern=10.0, southern=20.0)
        sv = StateValue.hemispheric(slice)
        assert not sv.is_scalar()
        assert not sv.is_four_box()
        assert sv.is_hemispheric()
        assert sv.as_scalar() is None
        assert sv.as_four_box() is None
        hs = sv.as_hemispheric()
        assert hs is not None
        assert hs.northern == 10.0
        assert hs.southern == 20.0
        # to_scalar should return mean of hemispheres
        assert sv.to_scalar() == pytest.approx(15.0, rel=0.01)

    def test_repr_scalar(self):
        """Test repr for scalar StateValue."""
        sv = StateValue.scalar(42.0)
        assert "StateValue.scalar(42" in repr(sv)

    def test_repr_four_box(self):
        """Test repr for FourBox StateValue."""
        slice = FourBoxSlice.uniform(1.0)
        sv = StateValue.four_box(slice)
        assert "StateValue.four_box" in repr(sv)
        assert "FourBoxSlice" in repr(sv)

    def test_repr_hemispheric(self):
        """Test repr for Hemispheric StateValue."""
        slice = HemisphericSlice.uniform(1.0)
        sv = StateValue.hemispheric(slice)
        assert "StateValue.hemispheric" in repr(sv)
        assert "HemisphericSlice" in repr(sv)


class TestStateValueInComponents:
    """Test StateValue integration with typed Python components."""

    def test_typed_component_with_scalar_output(self):
        """Test that typed components can return scalar outputs via StateValue."""

        class ScalarComponent(Component):
            input_var = Input("Input", unit="W/m^2")
            output_var = Output("Output", unit="K")

            def solve(self, t_current, t_next, inputs):
                return self.Outputs(output_var=inputs.input_var.current * 2.0)

        component = ScalarComponent()

        # Create dummy inputs
        outputs = component.Outputs(output_var=5.0)
        result = outputs.to_dict()

        assert "Output" in result
        sv = result["Output"]
        assert isinstance(sv, StateValue)
        assert sv.is_scalar()
        assert sv.as_scalar() == 5.0

    def test_typed_component_with_four_box_output(self):
        """Test that typed components can return FourBox outputs via StateValue."""

        class FourBoxComponent(Component):
            input_var = Input("Input", unit="W/m^2")
            output_var = Output("Regional Output", unit="K", grid="FourBox")

            def solve(self, t_current, t_next, inputs):
                return self.Outputs(
                    output_var=FourBoxSlice(
                        northern_ocean=1.0,
                        northern_land=2.0,
                        southern_ocean=3.0,
                        southern_land=4.0,
                    )
                )

        component = FourBoxComponent()

        outputs = component.Outputs(
            output_var=FourBoxSlice(
                northern_ocean=1.0,
                northern_land=2.0,
                southern_ocean=3.0,
                southern_land=4.0,
            )
        )
        result = outputs.to_dict()

        assert "Regional Output" in result
        sv = result["Regional Output"]
        assert isinstance(sv, StateValue)
        assert sv.is_four_box()
        fb = sv.as_four_box()

        assert fb is not None
        assert fb.northern_ocean == 1.0
        assert fb.southern_land == 4.0

    def test_typed_component_with_hemispheric_output(self):
        """Test that typed components can return Hemispheric outputs via StateValue."""

        class HemisphericComponent(Component):
            input_var = Input("Input", unit="W/m^2")
            output_var = Output("Hemispheric Output", unit="K", grid="Hemispheric")

            def solve(self, t_current, t_next, inputs):
                return self.Outputs(
                    output_var=HemisphericSlice(northern=10.0, southern=20.0)
                )

        component = HemisphericComponent()

        outputs = component.Outputs(
            output_var=HemisphericSlice(northern=10.0, southern=20.0)
        )
        result = outputs.to_dict()

        assert "Hemispheric Output" in result
        sv = result["Hemispheric Output"]
        assert isinstance(sv, StateValue)
        assert sv.is_hemispheric()
        hs = sv.as_hemispheric()

        assert hs is not None
        assert hs.northern == 10.0
        assert hs.southern == 20.0

    def test_typed_component_mixed_outputs(self):
        """Test that typed components can mix scalar and grid outputs."""

        class MixedComponent(Component):
            input_var = Input("Input", unit="W/m^2")
            scalar_out = Output("Scalar Output", unit="K")
            four_box_out = Output("FourBox Output", unit="K", grid="FourBox")

            def solve(self, t_current, t_next, inputs):
                return self.Outputs(
                    scalar_out=42.0, four_box_out=FourBoxSlice.uniform(10.0)
                )

        component = MixedComponent()

        outputs = component.Outputs(
            scalar_out=42.0, four_box_out=FourBoxSlice.uniform(10.0)
        )
        result = outputs.to_dict()

        assert result["Scalar Output"].is_scalar()
        assert result["FourBox Output"].is_four_box()


class TestStateValueInModel:
    """Test StateValue with full model integration."""

    def test_python_component_grid_output_in_model(self):
        """Test Python component with FourBox output runs in model."""

        class GridProducerComponent(Component):
            input_var = Input("Forcing", unit="W/m^2")
            output_var = Output("Regional Temperature", unit="K", grid="FourBox")

            def solve(self, t_current, t_next, inputs):
                forcing = inputs.input_var.current
                # Different response in each region
                return self.Outputs(
                    output_var=FourBoxSlice(
                        northern_ocean=forcing * 0.8,
                        northern_land=forcing * 1.2,
                        southern_ocean=forcing * 0.7,
                        southern_land=forcing * 1.1,
                    )
                )

        time_axis = TimeAxis.from_values(np.array([2000.0, 2001.0, 2002.0]))
        forcing_ts = Timeseries(
            np.array([1.0, 2.0, 3.0]),
            time_axis,
            "W/m^2",
            InterpolationStrategy.Linear,
        )

        component = GridProducerComponent()
        py_component = PythonComponent.build(component)

        model = (
            ModelBuilder()
            .with_time_axis(time_axis)
            .with_exogenous_variable("Forcing", forcing_ts)
            .with_py_component(py_component)
            .build()
        )

        model.run()

        # The model should complete without error
        assert model.finished()
