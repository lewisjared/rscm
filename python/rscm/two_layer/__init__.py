"""Two-layer climate model component."""

from rscm._lib.two_layer import TwoLayerBuilder

__all__ = ["TwoLayerBuilder"]


builder = TwoLayerBuilder.from_parameters(
    {
        "lambda0": 1.2,
        "a": 0.0,
        "efficacy": 1.0,
        "eta": 0.7,
        "heat_capacity_surface": 8.0,
        "heat_capacity_deep": 100.0,
    }
)
builder
