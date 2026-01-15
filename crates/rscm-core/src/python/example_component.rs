use crate::create_component_builder;
use crate::example_components::{TestComponent, TestComponentParameters};
use pyo3::prelude::*;
use pyo3::pyclass;

use crate::python::PyRustComponent;

create_component_builder!(TestComponentBuilder, TestComponent, TestComponentParameters);
