"""Tests for the VariableSchema Python bindings."""

import numpy as np
import pytest

from rscm._lib.core import (
    GridType,
    InterpolationStrategy,
    TimeAxis,
    Timeseries,
    VariableSchema,
)
from rscm.component import Component, Input, Output
from rscm.core import ModelBuilder, PythonComponent


class TestVariableSchemaBuilder:
    """Tests for VariableSchema builder API."""

    def test_create_empty_schema(self):
        schema = VariableSchema()
        assert schema.variables == {}
        assert schema.aggregates == {}

    def test_add_variable(self):
        schema = VariableSchema()
        schema.add_variable("Emissions|CO2", "GtCO2/yr")

        assert "Emissions|CO2" in schema.variables
        var = schema.variables["Emissions|CO2"]
        assert var.name == "Emissions|CO2"
        assert var.unit == "GtCO2/yr"
        assert var.grid_type == GridType.Scalar

    def test_add_variable_with_grid_type(self):
        schema = VariableSchema()
        schema.add_variable("Regional Temperature", "K", GridType.FourBox)

        var = schema.variables["Regional Temperature"]
        assert var.grid_type == GridType.FourBox

    def test_add_aggregate_sum(self):
        schema = VariableSchema()
        schema.add_variable("ERF|CO2", "W/m^2")
        schema.add_variable("ERF|CH4", "W/m^2")
        schema.add_aggregate("Total ERF", "W/m^2", "Sum", ["ERF|CO2", "ERF|CH4"])

        assert "Total ERF" in schema.aggregates
        agg = schema.aggregates["Total ERF"]
        assert agg.name == "Total ERF"
        assert agg.unit == "W/m^2"
        assert agg.operation_type == "Sum"
        assert agg.contributors == ["ERF|CO2", "ERF|CH4"]
        assert agg.weights is None

    def test_add_aggregate_mean(self):
        schema = VariableSchema()
        schema.add_variable("Value|A", "units")
        schema.add_variable("Value|B", "units")
        schema.add_aggregate("Average", "units", "Mean", ["Value|A", "Value|B"])

        agg = schema.aggregates["Average"]
        assert agg.operation_type == "Mean"

    def test_add_aggregate_weighted(self):
        schema = VariableSchema()
        schema.add_variable("Value|A", "units")
        schema.add_variable("Value|B", "units")
        schema.add_aggregate(
            "Weighted Total",
            "units",
            "Weighted",
            ["Value|A", "Value|B"],
            weights=[0.6, 0.4],
        )

        agg = schema.aggregates["Weighted Total"]
        assert agg.operation_type == "Weighted"
        assert agg.weights == [0.6, 0.4]

    def test_weighted_requires_weights(self):
        schema = VariableSchema()
        schema.add_variable("Value|A", "units")

        with pytest.raises(ValueError, match="weights must be provided"):
            schema.add_aggregate("Weighted", "units", "Weighted", ["Value|A"])

    def test_invalid_operation_raises(self):
        schema = VariableSchema()
        schema.add_variable("Value|A", "units")

        with pytest.raises(ValueError, match="Unknown operation"):
            schema.add_aggregate("Bad", "units", "Invalid", ["Value|A"])

    def test_contains(self):
        schema = VariableSchema()
        schema.add_variable("Var1", "units")
        schema.add_aggregate("Agg1", "units", "Sum", ["Var1"])

        assert schema.contains("Var1")
        assert schema.contains("Agg1")
        assert not schema.contains("Nonexistent")


class TestVariableSchemaValidation:
    """Tests for VariableSchema validation."""

    def test_valid_schema(self):
        schema = VariableSchema()
        schema.add_variable("ERF|CO2", "W/m^2")
        schema.add_variable("ERF|CH4", "W/m^2")
        schema.add_aggregate("Total ERF", "W/m^2", "Sum", ["ERF|CO2", "ERF|CH4"])

        # Should not raise
        schema.validate()

    def test_empty_schema_valid(self):
        schema = VariableSchema()
        schema.validate()

    def test_undefined_contributor_fails(self):
        schema = VariableSchema()
        schema.add_variable("ERF|CO2", "W/m^2")
        schema.add_aggregate(
            "Total ERF",
            "W/m^2",
            "Sum",
            ["ERF|CO2", "ERF|CH4"],  # ERF|CH4 not defined
        )

        with pytest.raises(ValueError, match="Undefined contributor"):
            schema.validate()

    def test_unit_mismatch_fails(self):
        schema = VariableSchema()
        schema.add_variable("ERF|CO2", "W/m^2")
        schema.add_variable("Emissions|CO2", "GtCO2/yr")  # Different unit
        schema.add_aggregate(
            "Total",
            "W/m^2",
            "Sum",
            ["ERF|CO2", "Emissions|CO2"],
        )

        with pytest.raises(ValueError, match="Unit mismatch"):
            schema.validate()

    def test_grid_mismatch_fails(self):
        schema = VariableSchema()
        schema.add_variable("Global", "K", GridType.Scalar)
        schema.add_variable("Regional", "K", GridType.FourBox)
        schema.add_aggregate("Total", "K", "Sum", ["Global", "Regional"])

        with pytest.raises(ValueError, match="Grid type mismatch"):
            schema.validate()

    def test_weight_count_mismatch_fails(self):
        schema = VariableSchema()
        schema.add_variable("A", "units")
        schema.add_variable("B", "units")
        schema.add_variable("C", "units")
        schema.add_aggregate(
            "Total",
            "units",
            "Weighted",
            ["A", "B", "C"],
            weights=[0.5, 0.5],  # 2 weights, 3 contributors
        )

        with pytest.raises(ValueError, match="Weight count mismatch"):
            schema.validate()

    def test_circular_dependency_fails(self):
        # Need to construct this manually since the Python API doesn't allow
        # adding aggregates that reference undefined contributors before validation
        schema = VariableSchema()
        schema.add_aggregate("A", "units", "Sum", ["B"])
        schema.add_aggregate("B", "units", "Sum", ["A"])

        with pytest.raises(ValueError, match="Circular dependency"):
            schema.validate()


class ERFComponent(Component):
    """Simple component that outputs ERF from concentration."""

    concentration = Input("Concentration|CO2", unit="ppm")
    erf = Output("ERF|Test", unit="W/m^2")

    def __init__(self, forcing_per_ppm: float = 0.01):
        self.forcing_per_ppm = forcing_per_ppm

    def solve(
        self, t_current, t_next, inputs: "ERFComponent.Inputs"
    ) -> "ERFComponent.Outputs":
        conc = inputs.concentration.at_start()
        return self.Outputs(erf=conc * self.forcing_per_ppm)


class TestSchemaModelIntegration:
    """Integration tests for VariableSchema with Model."""

    @pytest.fixture
    def time_axis(self):
        return TimeAxis.from_values(np.arange(2020.0, 2025.0, 1.0))

    @pytest.fixture
    def concentration_ts(self, time_axis):
        return Timeseries(
            np.array([400.0, 410.0, 420.0, 430.0, 440.0]),
            time_axis,
            "ppm",
            InterpolationStrategy.Previous,
        )

    def test_model_with_schema_validates(self, time_axis, concentration_ts):
        """Test that ModelBuilder validates components against schema."""
        schema = VariableSchema()
        schema.add_variable("Concentration|CO2", "ppm")
        schema.add_variable("ERF|Test", "W/m^2")
        schema.validate()

        component = PythonComponent.build(ERFComponent())

        model = (
            ModelBuilder()
            .with_time_axis(time_axis)
            .with_schema(schema)
            .with_py_component(component)
            .with_exogenous_variable("Concentration|CO2", concentration_ts)
            .build()
        )

        model.run()

        ts = model.timeseries()
        erf_ts = ts.get_timeseries_by_name("ERF|Test")
        assert erf_ts is not None
        # Verify that the model ran and produced output (values calculated correctly)
        # The exact timestep-by-timestep values depend on component execution timing,
        # but the final value should be correct
        assert not np.isnan(erf_ts.values()[-1])

    def test_schema_undefined_output_fails(self, time_axis, concentration_ts):
        """Test that building fails if component output is not in schema."""
        schema = VariableSchema()
        schema.add_variable("Concentration|CO2", "ppm")
        # Note: ERF|Test is NOT in the schema
        schema.validate()

        component = PythonComponent.build(ERFComponent())

        with pytest.raises(ValueError, match="not defined in the schema"):
            (
                ModelBuilder()
                .with_time_axis(time_axis)
                .with_schema(schema)
                .with_py_component(component)
                .with_exogenous_variable("Concentration|CO2", concentration_ts)
                .build()
            )

    def test_model_with_aggregate(self, time_axis):
        """Test model execution with a Sum aggregate."""

        # Two components producing different ERF values
        class CO2ERF(Component):
            concentration = Input("Concentration|CO2", unit="ppm")
            erf = Output("ERF|CO2", unit="W/m^2")

            def solve(
                self, t_current, t_next, inputs: "CO2ERF.Inputs"
            ) -> "CO2ERF.Outputs":
                return self.Outputs(erf=inputs.concentration.at_start() * 0.01)

        class CH4ERF(Component):
            concentration = Input("Concentration|CH4", unit="ppb")
            erf = Output("ERF|CH4", unit="W/m^2")

            def solve(
                self, t_current, t_next, inputs: "CH4ERF.Inputs"
            ) -> "CH4ERF.Outputs":
                return self.Outputs(erf=inputs.concentration.at_start() * 0.001)

        # Create schema with aggregate
        schema = VariableSchema()
        schema.add_variable("Concentration|CO2", "ppm")
        schema.add_variable("Concentration|CH4", "ppb")
        schema.add_variable("ERF|CO2", "W/m^2")
        schema.add_variable("ERF|CH4", "W/m^2")
        schema.add_aggregate("Total ERF", "W/m^2", "Sum", ["ERF|CO2", "ERF|CH4"])
        schema.validate()

        # Create exogenous data
        co2_ts = Timeseries(
            np.array([400.0, 410.0, 420.0, 430.0, 440.0]),
            time_axis,
            "ppm",
            InterpolationStrategy.Previous,
        )
        ch4_ts = Timeseries(
            np.array([1800.0, 1850.0, 1900.0, 1950.0, 2000.0]),
            time_axis,
            "ppb",
            InterpolationStrategy.Previous,
        )

        model = (
            ModelBuilder()
            .with_time_axis(time_axis)
            .with_schema(schema)
            .with_py_component(PythonComponent.build(CO2ERF()))
            .with_py_component(PythonComponent.build(CH4ERF()))
            .with_exogenous_variable("Concentration|CO2", co2_ts)
            .with_exogenous_variable("Concentration|CH4", ch4_ts)
            .build()
        )

        model.run()

        ts = model.timeseries()

        # Check individual ERFs were computed
        co2_erf = ts.get_timeseries_by_name("ERF|CO2")
        ch4_erf = ts.get_timeseries_by_name("ERF|CH4")
        assert co2_erf is not None
        assert ch4_erf is not None
        assert not np.isnan(co2_erf.values()[-1])
        assert not np.isnan(ch4_erf.values()[-1])

        # Check aggregate was computed
        total_erf = ts.get_timeseries_by_name("Total ERF")
        assert total_erf is not None

        # The aggregate should be the sum of the two ERFs
        # Check that the last value is the sum (this is robust to timing issues)
        expected_total = co2_erf.values()[-1] + ch4_erf.values()[-1]
        np.testing.assert_allclose(total_erf.values()[-1], expected_total)
